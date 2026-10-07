"""Reconhecimento facial em fotos ou camera usando OpenCV e DeepFace.

Execute: python main.py
Camera: python main.py --camera
As pastas padrao sao relativas a este script, independentemente do diretorio atual.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic

# Deve ser definido antes de importar TensorFlow/DeepFace.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("DEEPFACE_HOME", str(Path(__file__).resolve().parent))

import cv2
import numpy as np
from deepface import DeepFace

BASE = Path(__file__).resolve().parent
EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp"}
VERDE = (0, 255, 0)  # OpenCV usa BGR.
VERMELHO = (0, 0, 255)
LOG = logging.getLogger(__name__)


def listar_imagens(pasta: Path) -> list[Path]:
    return sorted(p for p in pasta.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSOES)


def ler_imagem(caminho: Path) -> np.ndarray:
    """imdecode permite caminhos com acentos no Windows."""
    imagem = cv2.imdecode(np.fromfile(str(caminho), dtype=np.uint8), cv2.IMREAD_COLOR)
    if imagem is None:
        raise ValueError(f"Imagem invalida ou ilegivel: {caminho}")
    return imagem


def salvar_imagem(caminho: Path, imagem: np.ndarray) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    sucesso, dados = cv2.imencode(".png", imagem)
    if not sucesso:
        raise OSError(f"Nao foi possivel codificar a imagem: {caminho}")
    dados.tofile(str(caminho))


def representar(imagem: np.ndarray, modelo: str) -> list[dict]:
    return DeepFace.represent(
        img_path=imagem,
        model_name=modelo,
        detector_backend="opencv",
        enforce_detection=True,
        align=True,
    )


def carregar_autorizados(pasta: Path, modelo: str) -> list[dict]:
    arquivos = listar_imagens(pasta)
    if not arquivos:
        raise ValueError(f"Adicione fotos de pessoas autorizadas em: {pasta}")
    referencias = []
    for caminho in arquivos:
        try:
            rostos = representar(ler_imagem(caminho), modelo)
            if len(rostos) != 1:
                raise ValueError("A foto de cadastro deve conter exatamente um rosto.")
            relativo = caminho.relative_to(pasta)
            referencias.append({
                "pessoa": relativo.parts[0] if len(relativo.parts) > 1 else caminho.stem,
                "foto": str(relativo),
                "embedding": [float(v) for v in rostos[0]["embedding"]],
            })
        except Exception as erro:
            raise ValueError(f"Cadastro invalido em {caminho}: {erro}") from erro
    LOG.info("Carregadas %d fotos de pessoas autorizadas.", len(referencias))
    return referencias


def identificar(embedding: list[float], referencias: list[dict], modelo: str,
                limiar: float | None) -> dict:
    """Compara vetores ja extraidos; os cadastros ficam somente em memoria."""
    melhor = None
    for referencia in referencias:
        resultado = DeepFace.verify(
            img1_path=[float(v) for v in embedding],
            img2_path=referencia["embedding"],
            model_name=modelo,
            distance_metric="cosine",
            threshold=limiar,
            silent=True,
        )
        distancia = float(resultado["distance"])
        limite = float(resultado["threshold"])
        if not math.isfinite(distancia) or not math.isfinite(limite):
            raise ValueError("A comparacao retornou uma distancia ou limiar invalido.")
        candidato = {
            "pessoa": referencia["pessoa"],
            "foto_cadastro": referencia["foto"],
            "distancia": distancia,
            "limiar": limite,
            "liberado": bool(resultado["verified"]) and distancia <= limite,
        }
        if melhor is None or distancia < melhor["distancia"]:
            melhor = candidato
    if melhor is None:
        return {"liberado": False, "pessoa": None, "motivo": "Sem cadastros"}
    if not melhor["liberado"]:
        melhor["pessoa"] = None
        melhor["foto_cadastro"] = None
        melhor["motivo"] = "Pessoa nao reconhecida na base de autorizados"
    return melhor


def desenhar_caixa(imagem: np.ndarray, area: dict, liberado: bool) -> None:
    altura, largura = imagem.shape[:2]
    x = max(0, min(int(area["x"]), largura - 1))
    y = max(0, min(int(area["y"]), altura - 1))
    direita = max(x, min(int(area["x"]) + int(area["w"]), largura - 1))
    fundo = max(y, min(int(area["y"]) + int(area["h"]), altura - 1))
    cor = VERDE if liberado else VERMELHO
    texto = "ACESSO LIBERADO" if liberado else "ACESSO NEGADO"
    cv2.rectangle(imagem, (x, y), (direita, fundo), cor, 2)
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    escala = 0.6
    (texto_largura, texto_altura), base = cv2.getTextSize(texto, fonte, escala, 2)
    if texto_largura + 8 > largura:
        escala *= max(1, largura - 8) / (texto_largura + 8)
        (texto_largura, texto_altura), base = cv2.getTextSize(texto, fonte, escala, 2)
    texto_x = max(0, min(x, largura - texto_largura - 8))
    topo = y - texto_altura - base - 8
    if topo < 0:
        topo = y
    topo = max(0, min(topo, altura - texto_altura - base - 8))
    cv2.rectangle(imagem, (texto_x, topo),
                  (texto_x + texto_largura + 8, topo + texto_altura + base + 8), cor, -1)
    cv2.putText(imagem, texto, (texto_x + 4, topo + texto_altura + 4),
                fonte, escala, (0, 0, 0), 2, cv2.LINE_AA)


def processar_imagem(imagem: np.ndarray, referencias: list[dict], modelo: str,
                     limiar: float | None, registrar_erros: bool = True
                     ) -> tuple[np.ndarray, list[dict], str | None]:
    anotada = imagem.copy()
    try:
        rostos = representar(imagem, modelo)
        if not rostos:
            raise ValueError("Nenhum rosto detectado")
    except Exception as erro:
        # Sem deteccao confiavel, nenhuma identidade recebe acesso.
        altura, largura = imagem.shape[:2]
        desenhar_caixa(anotada, {"x": 0, "y": 0, "w": largura, "h": altura}, False)
        if registrar_erros:
            LOG.warning("Deteccao falhou: %s", erro)
        return anotada, [], str(erro)
    decisoes = []
    for rosto in rostos:
        area = {chave: int(rosto["facial_area"][chave]) for chave in ("x", "y", "w", "h")}
        try:
            decisao = identificar(rosto["embedding"], referencias, modelo, limiar)
        except Exception as erro:
            if registrar_erros:
                LOG.warning("Comparacao falhou: %s", erro)
            decisao = {"liberado": False, "pessoa": None, "motivo": str(erro), "erro": True}
        decisao["bounding_box"] = area
        decisao["status"] = "ACESSO LIBERADO" if decisao["liberado"] else "ACESSO NEGADO"
        desenhar_caixa(anotada, area, decisao["liberado"])
        decisoes.append(decisao)
    return anotada, decisoes, None


def abrir_camera(indice: int):
    # DirectShow costuma iniciar a webcam mais rapidamente no Windows.
    captura = cv2.VideoCapture(indice, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(indice)
    if not captura.isOpened() and os.name == "nt":
        captura.release()
        captura = cv2.VideoCapture(indice)
    return captura


def montar_visualizacao_camera(quadro: np.ndarray, analisada: np.ndarray | None,
                              fps: float, idade: float = 0.0) -> np.ndarray:
    """Mostra video fluido e o resultado no quadro exato que foi reconhecido."""
    altura, largura = quadro.shape[:2]
    tela = np.full((altura + 40, largura * 2, 3), 24, dtype=np.uint8)
    tela[40:, :largura] = quadro
    if analisada is not None:
        tela[40:, largura:] = cv2.resize(analisada, (largura, altura))
    else:
        cv2.putText(tela, "ANALISANDO...", (largura + 20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(tela, f"AO VIVO | {fps:.1f} FPS", (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    titulo = f"ULTIMA ANALISE | {idade:.1f}s atras" if analisada is not None else "RECONHECIMENTO"
    cv2.putText(tela, titulo, (largura + 10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return tela


def executar_camera(referencias: list[dict], modelo: str, limiar: float | None,
                    indice: int = 0, fps_camera: int = 30,
                    intervalo_analise: float = 0.25, largura_analise: int = 480) -> int:
    captura = None
    executor = None
    futuro = None
    quadro_pendente = None
    instante_pendente = 0.0
    analisada = None
    instante_resultado = 0.0
    ultimo_envio = float("-inf")
    tempos = deque(maxlen=30)
    janela = "Reconhecimento facial - Q ou ESC para sair"
    janela_criada = False
    ultimo_erro = None
    try:
        captura = abrir_camera(indice)
        if not captura.isOpened():
            LOG.error("Nao foi possivel abrir a camera %d. Feche outros aplicativos que usam a camera, "
                      "confira a permissao do Windows ou tente --indice-camera 1.", indice)
            return 1
        captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        captura.set(cv2.CAP_PROP_FPS, fps_camera)
        captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reconhecimento")
        cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
        janela_criada = True
        LOG.info("Camera %d aberta. Pressione Q ou ESC na janela para sair.", indice)
        while True:
            sucesso, quadro = captura.read()
            if not sucesso or quadro is None:
                LOG.error("Nao foi possivel receber imagens da camera.")
                return 1
            # Alguns drivers ignoram a resolucao solicitada. Limita o custo da deteccao.
            altura, largura = quadro.shape[:2]
            if largura > 640:
                quadro = cv2.resize(quadro, (640, max(1, round(altura * 640 / largura))))
            agora = monotonic()
            tempos.append(agora)
            if futuro is not None and futuro.done():
                try:
                    analisada, rostos, erro = futuro.result()
                except Exception as falha:
                    analisada = quadro_pendente.copy()
                    h, w = analisada.shape[:2]
                    desenhar_caixa(analisada, {"x": 0, "y": 0, "w": w, "h": h}, False)
                    rostos, erro = [], str(falha)
                instante_resultado = instante_pendente
                erro_atual = erro or "; ".join(r["motivo"] for r in rostos if r.get("erro")) or None
                if erro_atual and erro_atual != ultimo_erro:
                    LOG.warning("Analise da camera: %s", erro_atual)
                ultimo_erro = erro_atual
                futuro = None
            # No maximo uma analise em andamento, sem fila de quadros antigos.
            if futuro is None and agora - ultimo_envio >= intervalo_analise:
                h, w = quadro.shape[:2]
                quadro_pendente = cv2.resize(quadro, (largura_analise, max(1, round(h * largura_analise / w)))) \
                    if w > largura_analise else quadro.copy()
                instante_pendente = agora
                ultimo_envio = agora
                futuro = executor.submit(processar_imagem, quadro_pendente, referencias, modelo, limiar,
                                         registrar_erros=False)
            duracao = agora - tempos[0]
            fps = (len(tempos) - 1) / duracao if duracao > 0 else 0.0
            tela = montar_visualizacao_camera(quadro, analisada, fps, max(0.0, agora - instante_resultado))
            cv2.imshow(janela, tela)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), ord("Q"), 27):
                return 0
            try:
                visivel = cv2.getWindowProperty(janela, cv2.WND_PROP_VISIBLE)
            except cv2.error:
                return 0  # Alguns drivers removem a janela imediatamente ao clicar no X.
            if visivel < 1:
                return 0
    except KeyboardInterrupt:
        LOG.info("Camera encerrada pelo terminal.")
        return 0
    except Exception as erro:
        LOG.error("Falha no modo de camera: %s", erro)
        return 1
    finally:
        if captura is not None:
            captura.release()
        if janela_criada:
            try:
                cv2.destroyWindow(janela)
            except cv2.error:
                pass  # A janela pode ter sido fechada pelo usuario.
        if executor is not None:
            # Libera a camera e a janela antes de aguardar a ultima inferencia.
            executor.shutdown(wait=True, cancel_futures=True)


def pastas_sobrepostas(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconhecimento facial em fotos ou camera: OpenCV + DeepFace")
    parser.add_argument("--camera", action="store_true", help="Abrir a webcam e reconhecer rostos continuamente")
    parser.add_argument("--indice-camera", type=int, default=0, help="Indice da camera: 0 (padrao), 1, 2...")
    parser.add_argument("--fps-camera", type=int, default=30, help="FPS solicitado a webcam (padrao: 30)")
    parser.add_argument("--intervalo-analise", type=float, default=0.25,
                        help="Intervalo minimo entre analises, em segundos (padrao: 0.25)")
    parser.add_argument("--largura-analise", type=int, default=480,
                        help="Largura maxima da imagem analisada (padrao: 480 pixels)")
    parser.add_argument("--autorizados", type=Path, default=BASE / "fotos" / "autorizados")
    parser.add_argument("--testes", type=Path, default=BASE / "fotos" / "testes")
    parser.add_argument("--saida", type=Path, default=BASE / "resultados")
    parser.add_argument("--modelo", choices=["Facenet512", "Facenet", "VGG-Face", "ArcFace", "SFace"],
                        default="Facenet512")
    parser.add_argument("--limiar", type=float, default=None,
                        help="Limiar de distancia cosseno entre 0 e 2; padrao: valor do DeepFace")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.limiar is not None and (not math.isfinite(args.limiar) or not 0 < args.limiar <= 2):
        parser.error("--limiar deve ser um numero finito maior que 0 e menor ou igual a 2")
    if args.indice_camera < 0:
        parser.error("--indice-camera deve ser maior ou igual a 0")
    if not 1 <= args.fps_camera <= 120:
        parser.error("--fps-camera deve estar entre 1 e 120")
    if not math.isfinite(args.intervalo_analise) or args.intervalo_analise < 0:
        parser.error("--intervalo-analise deve ser um numero finito maior ou igual a 0")
    if not 160 <= args.largura_analise <= 640:
        parser.error("--largura-analise deve estar entre 160 e 640 pixels")
    if args.camera:
        try:
            autorizados = args.autorizados.resolve()
            autorizados.mkdir(parents=True, exist_ok=True)
            referencias = carregar_autorizados(autorizados, args.modelo)
        except Exception as erro:
            LOG.error("Nao foi possivel iniciar: %s", erro)
            return 1
        return executar_camera(referencias, args.modelo, args.limiar, args.indice_camera,
                               fps_camera=args.fps_camera, intervalo_analise=args.intervalo_analise,
                               largura_analise=args.largura_analise)
    autorizados, testes, saida = (p.resolve() for p in (args.autorizados, args.testes, args.saida))
    if any(pastas_sobrepostas(a, b) for a, b in
           [(autorizados, testes), (autorizados, saida), (testes, saida)]):
        parser.error("As pastas de autorizados, testes e saida devem ser separadas e nao conter umas as outras")
    try:
        for pasta in (autorizados, testes, saida):
            pasta.mkdir(parents=True, exist_ok=True)
        arquivos = listar_imagens(testes)
        if not arquivos:
            raise ValueError(f"Adicione fotos de teste em: {testes}")
        referencias = carregar_autorizados(autorizados, args.modelo)
    except Exception as erro:
        LOG.error("Nao foi possivel iniciar: %s", erro)
        return 1
    registros = []
    houve_erro = False
    for caminho in arquivos:
        relativo = caminho.relative_to(testes)
        try:
            anotada, rostos, erro = processar_imagem(ler_imagem(caminho), referencias, args.modelo, args.limiar)
            destino = saida / relativo.parent / (relativo.name + ".png")
            salvar_imagem(destino, anotada)
            falha = erro is not None or any(r.get("erro", False) for r in rostos)
            houve_erro |= falha
            registros.append({"imagem": str(relativo), "resultado": str(destino.relative_to(saida)),
                              "rostos": rostos, "erro": erro})
            LOG.info("%s: %d rosto(s), %d liberado(s). Resultado: %s", relativo, len(rostos),
                     sum(r["liberado"] for r in rostos), destino)
        except Exception as erro:
            houve_erro = True
            LOG.error("Falha em %s: %s", caminho, erro)
            registros.append({"imagem": str(relativo), "resultado": None, "rostos": [], "erro": str(erro)})
    relatorio = {"modelo": args.modelo, "metrica": "cosine", "limiar_personalizado": args.limiar,
                 "fotos_cadastradas": len(referencias), "imagens": registros}
    try:
        (saida / "relatorio.json").write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as erro:
        LOG.error("Nao foi possivel salvar o relatorio: %s", erro)
        return 1
    LOG.info("Relatorio salvo em %s", saida / "relatorio.json")
    return 1 if houve_erro else 0


if __name__ == "__main__":
    raise SystemExit(main())
