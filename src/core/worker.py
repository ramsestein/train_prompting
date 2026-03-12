"""Ejecución del modelo worker sobre un texto (con cadena de reviews opcional)."""
import re

from .api import call_api


def run_worker(prompt, text, entity_types, worker_model, review_prompts=None):
    """
    Ejecuta el worker con el prompt principal y opcionalmente N reviews encadenadas.
    Cada review recibe la salida de la llamada anterior y puede corregirla.
    review_prompts: lista ordenada de strings (prompt de cada review activa).
    """
    entity_types_str = "\n".join(f"- {t}" for t in sorted(entity_types))
    # Para modelos Ollama (thinking models como qwen3): no limitar max_tokens porque
    # el modelo usa el presupuesto de tokens para el bloque <think> antes de responder.
    # Si se limita, el modelo agota el límite en thinking y no produce ningún output.
    # Para OpenAI/DeepSeek se aplica el límite normal para controlar costes.
    from . import config as _cfg
    _is_ollama = not worker_model.startswith(("gpt-", "o1-", "o3-", "o4-", "deepseek-"))
    max_tokens = None if (_is_ollama and not _cfg.NO_THINK) else int(len(text.split()) * 3)

    # ── Llamada principal ────────────────────────────────────────────────
    system_msg = f"""{prompt}\n\nEtiquetas válidas:\n{entity_types_str}"""
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": text},
    ]
    output = call_api(worker_model, messages, temperature=0.1, max_tokens=max_tokens)

    # ── Reviews encadenadas ─────────────────────────────────────────────
    if review_prompts:
        _ann_pattern = re.compile(r"\[\*\*[^\]\*]+:[^\]\*]+?\*\*\]")
        for idx, review_prompt in enumerate(review_prompts, start=1):
            print(f"    [review {idx}]", end=" ", flush=True)
            review_system = f"""{review_prompt}\n\nEtiquetas válidas:\n{entity_types_str}"""
            review_messages = [
                {"role": "system", "content": review_system},
                {"role": "user", "content": output},
            ]
            review_output = call_api(
                worker_model, review_messages, temperature=0.1,
                max_tokens=max_tokens if max_tokens is not None else None,
            )
            prev_count = len(_ann_pattern.findall(output))
            new_count = len(_ann_pattern.findall(review_output))
            # Descartar si la review elimina TODAS las entidades que había
            if prev_count > 0 and new_count == 0:
                print(f"      ⚠ review_{idx} devolvió 0 entidades (había {prev_count}) — descartada, se conserva salida anterior")
            else:
                output = review_output

    return output
