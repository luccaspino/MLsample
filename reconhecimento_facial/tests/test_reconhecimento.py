"""Testes de decisoes e desenho, sem inferencia ou download de modelos."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main as app


class TestReconhecimento(unittest.TestCase):
    def setUp(self):
        self.referencias = [{"pessoa": "Ana", "foto": "Ana/foto.jpg", "embedding": [1.0, 0.0]}]
        self.imagem = np.zeros((240, 400, 3), dtype=np.uint8)
        self.area = {"x": 70, "y": 80, "w": 100, "h": 100}

    @patch.object(app.DeepFace, "verify")
    def test_correspondencia_libera_e_encaminha_limiar(self, verificar):
        verificar.return_value = {"verified": True, "distance": 0.1, "threshold": 0.25}
        resultado = app.identificar([1.0, 0.0], self.referencias, "Facenet512", 0.25)
        self.assertTrue(resultado["liberado"])
        self.assertEqual(resultado["pessoa"], "Ana")
        self.assertEqual(verificar.call_args.kwargs["threshold"], 0.25)

    @patch.object(app.DeepFace, "verify")
    def test_desconhecido_nega(self, verificar):
        verificar.return_value = {"verified": False, "distance": 0.8, "threshold": 0.3}
        resultado = app.identificar([0.0, 1.0], self.referencias, "Facenet512", None)
        self.assertFalse(resultado["liberado"])
        self.assertIsNone(resultado["pessoa"])

    @patch.object(app.DeepFace, "verify")
    def test_escolhe_cadastro_mais_proximo(self, verificar):
        outra = {"pessoa": "Bruno", "foto": "Bruno/foto.jpg", "embedding": [0.0, 1.0]}
        verificar.side_effect = [
            {"verified": True, "distance": 0.2, "threshold": 0.3},
            {"verified": True, "distance": 0.1, "threshold": 0.3},
        ]
        resultado = app.identificar([0.0, 1.0], self.referencias + [outra], "Facenet512", None)
        self.assertEqual(resultado["pessoa"], "Bruno")

    @patch.object(app.DeepFace, "verify")
    def test_distancia_nao_finita_nao_libera(self, verificar):
        verificar.return_value = {"verified": True, "distance": float("nan"), "threshold": 0.3}
        with self.assertRaises(ValueError):
            app.identificar([1.0, 0.0], self.referencias, "Facenet512", None)

    def test_caixas_tem_as_cores_exigidas(self):
        for liberado, cor, texto in [(True, app.VERDE, "ACESSO LIBERADO"),
                                    (False, app.VERMELHO, "ACESSO NEGADO")]:
            with self.subTest(liberado=liberado):
                imagem = self.imagem.copy()
                with patch.object(app.cv2, "putText", wraps=app.cv2.putText) as escrever:
                    app.desenhar_caixa(imagem, self.area, liberado)
                self.assertEqual(tuple(imagem[120, 70]), cor)
                self.assertEqual(escrever.call_args.args[1], texto)

    @patch.object(app, "identificar")
    @patch.object(app, "representar", side_effect=ValueError("Nenhum rosto detectado"))
    def test_sem_rosto_nega_sem_comparar(self, representar, identificar):
        anotada, rostos, erro = app.processar_imagem(self.imagem, self.referencias, "Facenet512", None)
        self.assertEqual(rostos, [])
        self.assertIsNotNone(erro)
        self.assertEqual(tuple(anotada[120, 0]), app.VERMELHO)
        identificar.assert_not_called()
        self.assertFalse(np.any(self.imagem))

    @patch.object(app, "representar")
    @patch.object(app, "identificar")
    def test_decisoes_independentes_por_rosto(self, identificar, representar):
        outra_area = {"x": 220, "y": 80, "w": 100, "h": 100}
        representar.return_value = [
            {"embedding": [1.0, 0.0], "facial_area": self.area},
            {"embedding": [0.0, 1.0], "facial_area": outra_area},
        ]
        identificar.side_effect = [{"liberado": True, "pessoa": "Ana"},
                                   {"liberado": False, "pessoa": None}]
        anotada, rostos, erro = app.processar_imagem(self.imagem, self.referencias, "Facenet512", None)
        self.assertIsNone(erro)
        self.assertEqual([r["status"] for r in rostos], ["ACESSO LIBERADO", "ACESSO NEGADO"])
        self.assertEqual(tuple(anotada[120, 70]), app.VERDE)
        self.assertEqual(tuple(anotada[120, 220]), app.VERMELHO)

    @patch.object(app, "representar")
    @patch.object(app, "identificar", side_effect=RuntimeError("Falha no modelo"))
    def test_falha_na_comparacao_nega(self, identificar, representar):
        representar.return_value = [{"embedding": [1.0, 0.0], "facial_area": self.area}]
        anotada, rostos, _ = app.processar_imagem(self.imagem, self.referencias, "Facenet512", None)
        self.assertFalse(rostos[0]["liberado"])
        self.assertTrue(rostos[0]["erro"])
        self.assertEqual(tuple(anotada[120, 70]), app.VERMELHO)

    @patch.object(app, "representar", return_value=[{}, {}])
    @patch.object(app, "ler_imagem")
    def test_cadastro_com_duas_pessoas_e_rejeitado(self, ler, representar):
        with tempfile.TemporaryDirectory() as temporario:
            pasta = Path(temporario)
            (pasta / "grupo.jpg").touch()
            with self.assertRaisesRegex(ValueError, "exatamente um rosto"):
                app.carregar_autorizados(pasta, "Facenet512")

    def test_impede_sobreposicao_das_pastas(self):
        pasta = Path("base").resolve()
        self.assertTrue(app.pastas_sobrepostas(pasta, pasta / "resultados"))
        self.assertTrue(app.pastas_sobrepostas(pasta, pasta))
        self.assertFalse(app.pastas_sobrepostas(pasta / "testes", pasta / "resultados"))


if __name__ == "__main__":
    unittest.main()
