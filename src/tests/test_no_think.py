#!/usr/bin/env python3
"""
Tests para verificar el modo no-think con modelos Ollama (qwen3.5:35b).

Grupos de tests
───────────────
  TestStripThinkTags    — unitarios, sin red
  TestCallApiPayload    — mockean requests.post, sin red
  TestOllamaIntegration — integración real (se omiten si Ollama/modelo no está disponible)

Uso rápido:
  python src/test_no_think.py
  python src/test_no_think.py -v
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import requests

# Añadir src/ al path para importar el paquete core
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config  # noqa: E402
from core import api      # noqa: E402
from core import ollama_utils  # noqa: E402

MODEL_NAME = "qwen3.5:35b"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_mock_response(*content_tokens):
    """Crea un mock de requests.Response con tokens SSE compatibles con call_api."""
    lines = []
    for token in content_tokens:
        chunk = {"choices": [{"delta": {"content": token}}]}
        lines.append(f"data: {json.dumps(chunk)}".encode())
    lines.append(b"data: [DONE]")
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_lines.return_value = iter(lines)
    return mock_resp


def _is_ollama_available() -> bool:
    try:
        resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def _is_model_available(model_name: str) -> bool:
    try:
        return model_name in ollama_utils.list_ollama_models()
    except Exception:
        return False


# ─── Tests unitarios: strip_think_tags ───────────────────────────────────────

class TestStripThinkTags(unittest.TestCase):
    """Tests para strip_think_tags — no requieren red."""

    def test_sin_bloques_think(self):
        self.assertEqual(api.strip_think_tags("hola mundo"), "hola mundo")

    def test_bloque_think_simple(self):
        result = api.strip_think_tags("<think>razonamiento interno</think>respuesta")
        self.assertEqual(result, "respuesta")

    def test_multiples_bloques_think(self):
        result = api.strip_think_tags("<think>a</think>medio<think>b</think>fin")
        self.assertEqual(result, "mediofin")

    def test_bloque_think_multilinea(self):
        result = api.strip_think_tags("<think>\npaso 1\npaso 2\n</think>resultado")
        self.assertEqual(result, "resultado")

    def test_string_vacio(self):
        self.assertEqual(api.strip_think_tags(""), "")

    def test_solo_think(self):
        self.assertEqual(api.strip_think_tags("<think>solo pensamiento</think>"), "")

    def test_sin_cierre_no_altera(self):
        # Regex requiere cierre; sin </think> no actúa
        text = "<think>sin cierre aquí"
        self.assertEqual(api.strip_think_tags(text), text)

    def test_texto_antes_y_despues(self):
        result = api.strip_think_tags("inicio<think>medio</think>fin")
        self.assertEqual(result, "iniciofin")


# ─── Tests de payload: call_api con mock ──────────────────────────────────────

class TestCallApiPayload(unittest.TestCase):
    """Verifica que call_api construye el payload correcto para cada combinación."""

    def _invoke(self, model, no_think_value, extra_keys=None):
        """Llama a call_api con mock y devuelve los argumentos de requests.post."""
        mock_resp = _make_mock_response("4")
        original_keys = dict(config.API_KEYS)
        if extra_keys:
            config.API_KEYS.update(extra_keys)
        try:
            with patch("requests.post", return_value=mock_resp) as mock_post:
                with patch.object(config, "NO_THINK", no_think_value):
                    api.call_api(
                        model,
                        [{"role": "user", "content": "2+2"}],
                        temperature=0.1,
                    )
            return mock_post.call_args
        finally:
            config.API_KEYS.clear()
            config.API_KEYS.update(original_keys)

    # ── Modelo Ollama ──────────────────────────────────────────────────────

    def test_ollama_no_think_true_incluye_think_false(self):
        """Ollama + NO_THINK=True → payload debe tener think=False."""
        call_args = self._invoke(MODEL_NAME, no_think_value=True)
        payload = call_args[1]["json"]
        self.assertIn("think", payload)
        self.assertIs(payload["think"], False)

    def test_ollama_no_think_false_no_incluye_think(self):
        """Ollama + NO_THINK=False → payload NO debe contener 'think'."""
        call_args = self._invoke(MODEL_NAME, no_think_value=False)
        payload = call_args[1]["json"]
        self.assertNotIn("think", payload)

    def test_ollama_url_apunta_a_local(self):
        """El modelo Ollama debe llamar a la URL local."""
        call_args = self._invoke(MODEL_NAME, no_think_value=False)
        url = call_args[0][0]
        self.assertTrue(url.startswith(config.OLLAMA_BASE_URL))

    def test_nombre_modelo_en_payload(self):
        """El nombre del modelo debe estar en el payload."""
        call_args = self._invoke(MODEL_NAME, no_think_value=True)
        payload = call_args[1]["json"]
        self.assertEqual(payload["model"], MODEL_NAME)

    def test_stream_habilitado_en_payload(self):
        """El payload siempre debe tener stream=True."""
        call_args = self._invoke(MODEL_NAME, no_think_value=True)
        payload = call_args[1]["json"]
        self.assertTrue(payload["stream"])

    # ── Modelos cloud: think NO debe aparecer ──────────────────────────────

    def test_openai_no_think_no_incluye_think(self):
        """Modelo OpenAI + NO_THINK=True → payload NO debe tener 'think'."""
        call_args = self._invoke(
            "gpt-4o",
            no_think_value=True,
            extra_keys={"OPENAI_API_KEY": "sk-fake-key"},
        )
        payload = call_args[1]["json"]
        self.assertNotIn("think", payload)

    def test_deepseek_no_think_no_incluye_think(self):
        """Modelo DeepSeek + NO_THINK=True → payload NO debe tener 'think'."""
        call_args = self._invoke(
            "deepseek-chat",
            no_think_value=True,
            extra_keys={"DEEPSEEK_API_KEY": "fake-ds-key"},
        )
        payload = call_args[1]["json"]
        self.assertNotIn("think", payload)


# ─── Tests de integración con Ollama real ─────────────────────────────────────

_OLLAMA_UP = _is_ollama_available()
_MODEL_READY = _is_model_available(MODEL_NAME) if _OLLAMA_UP else False


@unittest.skipUnless(_OLLAMA_UP, "Ollama no disponible — ejecuta: ollama serve")
@unittest.skipUnless(_MODEL_READY, f"Modelo {MODEL_NAME} no instalado — ejecuta: ollama pull {MODEL_NAME}")
class TestOllamaIntegration(unittest.TestCase):
    """Tests de integración reales contra Ollama con qwen3.5:35b."""

    def test_respuesta_sin_think_blocks_con_no_think(self):
        """Con NO_THINK=True, la respuesta no debe contener <think>...</think>."""
        with patch.object(config, "NO_THINK", True):
            messages = [{"role": "user", "content": "¿Cuánto es 2+2? Solo escribe el número."}]
            response = api.call_api(MODEL_NAME, messages, temperature=0.1)

        self.assertNotIn("<think>", response.lower())
        self.assertNotIn("</think>", response.lower())
        self.assertGreater(len(response.strip()), 0, "La respuesta no debe estar vacía")

    def test_respuesta_es_string_valido(self):
        """La respuesta debe ser una cadena no vacía con NO_THINK=True."""
        with patch.object(config, "NO_THINK", True):
            messages = [{"role": "user", "content": "Di solo 'hola'"}]
            response = api.call_api(MODEL_NAME, messages, temperature=0.0)

        self.assertIsInstance(response, str)
        self.assertGreater(len(response.strip()), 0)

    def test_payload_enviado_con_think_false(self):
        """Verifica que el payload real enviado a Ollama contiene think=False."""
        captured = {}
        original_post = requests.post

        def capture_post(url, **kwargs):
            captured.update(kwargs.get("json", {}))
            return original_post(url, **kwargs)

        with patch("requests.post", side_effect=capture_post):
            with patch.object(config, "NO_THINK", True):
                api.call_api(
                    MODEL_NAME,
                    [{"role": "user", "content": "1+1"}],
                    temperature=0.1,
                    max_tokens=10,
                )

        self.assertIn("think", captured, "El campo 'think' debe estar en el payload")
        self.assertFalse(captured["think"], "El campo 'think' debe ser False")


# ─── Punto de entrada ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
