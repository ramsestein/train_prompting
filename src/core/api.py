"""Llamadas a la API de LLM (OpenAI, DeepSeek, Ollama) con streaming."""
import json as _json
import re
import time

import requests

from . import config  # importado como módulo para ver las mutaciones de runtime (NO_THINK, API_KEYS)


def strip_think_tags(text):
    """Elimina bloques <think>...</think> de la salida del modelo."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ── helpers internos ─────────────────────────────────────────────────────────

def _call_openai_compat(url, headers, payload, model):
    """Streaming SSE (OpenAI / DeepSeek)."""
    t0 = time.time()
    resp = requests.post(url, headers=headers, json=payload, timeout=600, stream=True)
    resp.raise_for_status()

    chunks = []
    tok_prompt = 0
    tok_gen = 0
    gen_chars = 0

    for line in resp.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8") if isinstance(line, bytes) else line
        if not line_str.startswith("data: "):
            continue
        data_str = line_str[6:]
        if data_str.strip() == "[DONE]":
            break
        try:
            data = _json.loads(data_str)
        except _json.JSONDecodeError:
            continue

        if data.get("usage"):
            tok_prompt = data["usage"].get("prompt_tokens", 0)
            tok_gen = data["usage"].get("completion_tokens", 0)

        choices = data.get("choices", [])
        if choices:
            token = choices[0].get("delta", {}).get("content") or ""
            if token:
                chunks.append(token)
                gen_chars += len(token)
                if gen_chars % 200 < len(token) or gen_chars == len(token):
                    print(f"    ... generando ({gen_chars} chars)", end="\r", flush=True)

    elapsed = time.time() - t0
    raw = "".join(chunks)
    print(f"    ⏱ {model} — {elapsed:.1f}s  (prompt_tok={tok_prompt}, gen_tok={tok_gen}, chars={len(raw)})")
    return strip_think_tags(raw)


def _call_ollama_native(model, messages, payload_opts, model_label):
    """Streaming con API nativo de Ollama (/api/chat).

    Usa /api/chat en lugar de /v1/chat/completions porque solo el endpoint
    nativo respeta correctamente ``think: false`` en modelos con reasoning.
    """
    url = f"{config.OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "num_ctx": config.OLLAMA_NUM_CTX,
            "temperature": payload_opts.get("temperature", 0.3),
        },
    }
    if payload_opts.get("max_tokens") is not None:
        payload["options"]["num_predict"] = payload_opts["max_tokens"]
    if config.NO_THINK:
        payload["think"] = False

    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=600, stream=True)
    resp.raise_for_status()

    chunks = []
    chunks_think = []
    tok_prompt = 0
    tok_gen = 0
    gen_chars = 0

    for line in resp.iter_lines():
        if not line:
            continue
        try:
            data = _json.loads(line)
        except _json.JSONDecodeError:
            continue

        msg = data.get("message", {})
        token = msg.get("content") or ""
        think_token = msg.get("reasoning") or ""

        if token:
            chunks.append(token)
            gen_chars += len(token)
            if gen_chars % 200 < len(token) or gen_chars == len(token):
                print(f"    ... generando ({gen_chars} chars)", end="\r", flush=True)
        elif think_token:
            chunks_think.append(think_token)

        if data.get("done"):
            tok_prompt = data.get("prompt_eval_count", 0)
            tok_gen = data.get("eval_count", 0)
            break

    elapsed = time.time() - t0
    raw = "".join(chunks)
    raw_think = "".join(chunks_think)
    think_marker = f", think_chars={len(raw_think)}" if raw_think else ""
    print(f"    ⏱ {model_label} — {elapsed:.1f}s  (prompt_tok={tok_prompt}, gen_tok={tok_gen}, chars={len(raw)}{think_marker})")
    if not raw and raw_think:
        print("    ⚠ El modelo no produjo output (solo thinking). Usa --no-think o elimina el límite max_tokens.")
    return strip_think_tags(raw)


# ── función pública ──────────────────────────────────────────────────────────

def call_api(model, messages, temperature=0.3, max_tokens=None):
    """
    Llama a OpenAI, DeepSeek u Ollama según el nombre del modelo.
    - gpt-*, o1-*, o3-*, o4-*  → api.openai.com  (OPENAI_API_KEY)
    - deepseek-*                → api.deepseek.com (DEEPSEEK_API_KEY)
    - cualquier otro nombre     → Ollama local (API nativo /api/chat)
    """
    if model.startswith(("gpt-", "o1-", "o3-", "o4-")):
        url = "https://api.openai.com/v1/chat/completions"
        api_key = config.API_KEYS.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY no encontrada en .env")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    elif model.startswith("deepseek-"):
        url = "https://api.deepseek.com/v1/chat/completions"
        api_key = config.API_KEYS.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY no encontrada en .env")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    else:
        # Ollama: API nativo (soporta think: false correctamente)
        return _call_ollama_native(model, messages, {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }, model)

    # OpenAI / DeepSeek: SSE compatible
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    return _call_openai_compat(url, headers, payload, model)
