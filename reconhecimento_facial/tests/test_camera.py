"""Verifica o ciclo da camera sem acessar hardware ou abrir janelas nos testes."""

import sys
import unittest
from concurrent.futures import Future, ThreadPoolExecutor as ExecutorReal
from contextlib import ExitStack
from pathlib import Path
from threading import Event
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main as app


class TestCamera(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.captura = MagicMock()
        self.captura.isOpened.return_value = True
        self.quadro = np.zeros((240, 400, 3), dtype=np.uint8)
        self.captura.read.return_value = (True, self.quadro)
        self.stack.enter_context(patch.object(app, "abrir_camera", return_value=self.captura))
        self.processar = self.stack.enter_context(patch.object(app, "processar_imagem",
                                                              return_value=(self.quadro.copy(), [], None)))
        self.executor = MagicMock()
        self.futuro = Future()
        self.executor.submit.return_value = self.futuro
        self.stack.enter_context(patch.object(app, "ThreadPoolExecutor", return_value=self.executor))
        self.janela = self.stack.enter_context(patch.object(app.cv2, "namedWindow"))
        self.exibir = self.stack.enter_context(patch.object(app.cv2, "imshow"))
        self.tecla = self.stack.enter_context(patch.object(app.cv2, "waitKey", return_value=ord("q")))
        self.visibilidade = self.stack.enter_context(patch.object(app.cv2, "getWindowProperty", return_value=1))
        self.fechar = self.stack.enter_context(patch.object(app.cv2, "destroyWindow"))

    def executar(self):
        return app.executar_camera([{"pessoa": "Ana"}], "Facenet512", None)

    def test_tecla_q_encerra_e_libera_camera(self):
        self.assertEqual(self.executar(), 0)
        self.captura.release.assert_called_once()
        self.fechar.assert_called_once()
        self.executor.submit.assert_called_once()
        self.assertIs(self.executor.submit.call_args.args[0], app.processar_imagem)
        self.assertFalse(self.executor.submit.call_args.kwargs["registrar_erros"])
        self.executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)

    def test_escape_encerra(self):
        self.tecla.return_value = 27
        self.assertEqual(self.executar(), 0)
        self.captura.release.assert_called_once()

    def test_fechar_janela_encerra(self):
        self.tecla.return_value = -1
        self.visibilidade.return_value = 0
        self.assertEqual(self.executar(), 0)
        self.captura.release.assert_called_once()

    def test_camera_indisponivel_nao_abre_janela(self):
        self.captura.isOpened.return_value = False
        self.assertEqual(self.executar(), 1)
        self.captura.release.assert_called_once()
        self.janela.assert_not_called()
        self.processar.assert_not_called()

    def test_janela_removida_pelo_x_encerra_normalmente(self):
        self.tecla.return_value = -1
        self.visibilidade.side_effect = app.cv2.error("Janela removida")
        self.assertEqual(self.executar(), 0)
        self.captura.release.assert_called_once()

    def test_falha_de_leitura_libera_camera(self):
        self.captura.read.return_value = (False, None)
        self.assertEqual(self.executar(), 1)
        self.captura.release.assert_called_once()
        self.fechar.assert_called_once()
        self.processar.assert_not_called()

    def test_excecao_na_analise_nega_sem_parar_video(self):
        self.futuro.set_exception(RuntimeError("Falha na analise"))
        self.tecla.side_effect = [-1, ord("q")]
        self.assertEqual(self.executar(), 0)
        tela = self.exibir.call_args.args[1]
        self.assertEqual(tuple(tela[160, 400]), app.VERMELHO)
        self.assertEqual(tuple(tela[160, 0]), (0, 0, 0))  # Nao transfere caixas ao video ao vivo.
        self.captura.release.assert_called_once()
        self.fechar.assert_called_once()

    def test_interrupcao_no_terminal_libera_camera(self):
        self.captura.read.side_effect = KeyboardInterrupt
        self.assertEqual(self.executar(), 0)
        self.captura.release.assert_called_once()
        self.fechar.assert_called_once()

    def test_resultado_fica_no_quadro_analisado_e_proximo_envio_usa_o_mais_recente(self):
        quadro_a = self.quadro.copy()
        quadro_b = np.full_like(self.quadro, 50)
        quadro_c = np.full_like(self.quadro, 100)
        area = {"x": 50, "y": 70, "w": 100, "h": 100}
        liberado = quadro_a.copy()
        app.desenhar_caixa(liberado, area, True)
        futuro_c = Future()
        self.executor.submit.side_effect = [self.futuro, futuro_c]
        self.captura.read.side_effect = [(True, quadro_a), (True, quadro_b), (True, quadro_c)]
        def finalizar_primeira_analise(*args):
            if not self.futuro.done():
                self.futuro.set_result((liberado, [], None))
        self.exibir.side_effect = finalizar_primeira_analise
        self.tecla.side_effect = [-1, -1, ord("q")]
        with patch.object(app, "monotonic", side_effect=[10.0, 10.1, 10.4]):
            self.assertEqual(self.executar(), 0)
        self.assertEqual(self.executor.submit.call_count, 2)
        np.testing.assert_array_equal(self.executor.submit.call_args_list[0].args[1], quadro_a)
        np.testing.assert_array_equal(self.executor.submit.call_args_list[1].args[1], quadro_c)
        tela_b = self.exibir.call_args_list[1].args[1]
        self.assertEqual(tuple(tela_b[140, 50]), (50, 50, 50))
        self.assertEqual(tuple(tela_b[140, 450]), app.VERDE)

    def test_video_continua_e_nao_enfileira_durante_analise_pendente(self):
        self.tecla.side_effect = [-1, -1, ord("q")]
        with patch.object(app, "monotonic", side_effect=[1.0, 2.0, 3.0]):
            self.assertEqual(self.executar(), 0)
        self.assertEqual(self.exibir.call_count, 3)
        self.executor.submit.assert_called_once()

    def test_video_atualiza_enquanto_worker_real_esta_ocupado(self):
        iniciou, liberar, terminou = Event(), Event(), Event()
        estados = []
        def analisar(*args, **kwargs):
            iniciou.set()
            liberar.wait(timeout=2)
            terminou.set()
            return self.quadro.copy(), [], None
        def exibir(*args):
            self.assertTrue(iniciou.wait(timeout=1))
            estados.append(terminou.is_set())
            if len(estados) == 3:
                liberar.set()
        self.processar.side_effect = analisar
        self.exibir.side_effect = exibir
        self.tecla.side_effect = [-1, -1, ord("q")]
        with patch.object(app, "ThreadPoolExecutor", ExecutorReal):
            self.assertEqual(self.executar(), 0)
        self.assertEqual(estados, [False, False, False])
        self.assertTrue(terminou.is_set())

    def test_largura_da_analise_preserva_video_na_resolucao_original(self):
        self.quadro = np.zeros((480, 640, 3), dtype=np.uint8)
        self.captura.read.return_value = True, self.quadro
        self.assertEqual(self.executar(), 0)
        self.assertEqual(self.executor.submit.call_args.args[1].shape, (360, 480, 3))
        self.assertEqual(self.exibir.call_args.args[1].shape, (520, 1280, 3))

    def test_modo_camera_nao_exige_fotos_de_teste(self):
        with patch.object(sys, "argv", ["main.py", "--camera"]), \
             patch.object(app, "carregar_autorizados", return_value=[{"pessoa": "Ana"}]), \
             patch.object(app, "listar_imagens") as listar, \
             patch.object(app, "executar_camera", return_value=0) as executar:
            self.assertEqual(app.main(), 0)
            listar.assert_not_called()
            executar.assert_called_once_with([{"pessoa": "Ana"}], "Facenet512", None, 0,
                                            fps_camera=30, intervalo_analise=0.25, largura_analise=480)


if __name__ == "__main__":
    unittest.main()
