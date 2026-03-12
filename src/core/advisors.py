"""Subagentes especializados en recall y precision.

Cada advisor recibe la cadena de prompts actual y los resultados del batch,
analiza los errores desde la óptica de su métrica y genera un consejo breve
que el optimizador central puede incorporar.
"""
from .api import call_api


def _build_results_block(results, focus):
    """Construye el bloque de resultados filtrando según el foco del advisor.

    - recall  → muestra solo los FN (entidades no detectadas).
    - precision → muestra solo los FP (falsos positivos).
    """
    lines = []
    for r in results:
        m = r["metrics"]
        gt_remaining = list(r["entities"])
        pred_remaining = list(r["predicted"])
        for p in list(pred_remaining):
            if p in gt_remaining:
                gt_remaining.remove(p)
                pred_remaining.remove(p)

        if focus == "recall" and not gt_remaining:
            continue  # nada que reportar para recall en esta muestra
        if focus == "precision" and not pred_remaining:
            continue

        lines.append(f"\n{'─'*40}")
        lines.append(f"Archivo: {r['name']}")
        lines.append(f"Texto (500 chars): {r['text'][:500]}")
        lines.append(f"Métricas: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
                      f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})")

        if focus == "recall":
            lines.append("Entidades NO DETECTADAS (FN):")
            for etype, etext in gt_remaining:
                lines.append(f"  {etype}: {etext}")
        else:
            lines.append("FALSOS POSITIVOS (FP):")
            for etype, etext in pred_remaining:
                lines.append(f"  {etype}: {etext}")
            if gt_remaining:
                lines.append("(Para referencia, FN en esta muestra):")
                for etype, etext in gt_remaining:
                    lines.append(f"  {etype}: {etext}")

        lines.append(f"\nRespuesta del modelo:\n{r['model_output'][:800]}")

    return "\n".join(lines)


def _build_prompts_block(prompts):
    """Serializa la cadena de prompts para enviarla al advisor."""
    lines = [f"[main]\n{prompts['main']}"]
    for key in sorted(k for k in prompts if k.startswith("review_")):
        lines.append(f"\n[{key}]\n{prompts[key]}")
    return "\n".join(lines)


def recall_advisor(prompts, results, optimizer_model):
    """Analiza los FN y genera un consejo para mejorar el recall."""
    errors_block = _build_results_block(results, "recall")
    if not errors_block.strip():
        return ""  # recall perfecto, nada que aconsejar

    total_fn = sum(r["metrics"]["fn"] for r in results)
    avg_recall = sum(r["metrics"]["recall"] for r in results) / len(results)

    fn_by_type = {}
    for r in results:
        gt_remaining = list(r["entities"])
        for p in r["predicted"]:
            if p in gt_remaining:
                gt_remaining.remove(p)
        for etype, _ in gt_remaining:
            fn_by_type[etype] = fn_by_type.get(etype, 0) + 1
    top_fn = sorted(fn_by_type.items(), key=lambda x: -x[1])[:10]
    fn_summary = ", ".join(f"{t}={c}" for t, c in top_fn)

    system_msg = f"""Eres un especialista en RECALL para sistemas de extracción de entidades.

Tu ÚNICO objetivo es proponer estrategias concretas para que el prompt del modelo detecte
MÁS entidades (reducir falsos negativos). NO te preocupes por la precisión.

CADENA DE PROMPTS ACTUAL:
{_build_prompts_block(prompts)}

ESTADÍSTICAS:
  Recall promedio: {avg_recall:.4f}
  Total FN (no detectados): {total_fn}
  Top tipos más fallados: {fn_summary}

MUESTRAS CON ENTIDADES NO DETECTADAS:
{errors_block}

INSTRUCCIONES:
1. Identifica PATRONES en las entidades que el modelo no detecta (tipos, posición en el texto,
   formato, contexto lingüístico).
2. Propón cambios concretos al prompt que ayuden a detectar estas entidades.
3. Si hay tipos de entidad que se fallan sistemáticamente, sugiere ejemplos o reglas específicas.
4. Sé específico: cita ejemplos del texto donde falló y explica qué instrucción falta.

Responde con un informe BREVE (máximo 400 palabras) con consejos accionables.
NO generes prompts, solo el análisis y las recomendaciones."""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "Analiza los fallos de recall y genera recomendaciones."},
    ]
    return call_api(optimizer_model, messages, temperature=0.4)


def precision_advisor(prompts, results, optimizer_model):
    """Analiza los FP y genera un consejo para mejorar la precisión."""
    errors_block = _build_results_block(results, "precision")
    if not errors_block.strip():
        return ""  # precision perfecta, nada que aconsejar

    total_fp = sum(r["metrics"]["fp"] for r in results)
    avg_precision = sum(r["metrics"]["precision"] for r in results) / len(results)

    fp_by_type = {}
    for r in results:
        pred_remaining = list(r["predicted"])
        for gt in r["entities"]:
            if gt in pred_remaining:
                pred_remaining.remove(gt)
        for etype, _ in pred_remaining:
            fp_by_type[etype] = fp_by_type.get(etype, 0) + 1
    top_fp = sorted(fp_by_type.items(), key=lambda x: -x[1])[:10]
    fp_summary = ", ".join(f"{t}={c}" for t, c in top_fp)

    system_msg = f"""Eres un especialista en PRECISIÓN para sistemas de extracción de entidades.

Tu ÚNICO objetivo es proponer estrategias concretas para que el prompt del modelo genere
MENOS falsos positivos (entidades marcadas que no deberían estarlo). NO te preocupes por el recall.

CADENA DE PROMPTS ACTUAL:
{_build_prompts_block(prompts)}

ESTADÍSTICAS:
  Precisión promedio: {avg_precision:.4f}
  Total FP (falsos positivos): {total_fp}
  Top tipos con más FP: {fp_summary}

MUESTRAS CON FALSOS POSITIVOS:
{errors_block}

INSTRUCCIONES:
1. Identifica PATRONES en los falsos positivos (tipos, contexto, formatos que confunden al modelo).
2. Propón reglas o restricciones concretas para eliminar esos FP.
3. Si ciertos tipos de entidad se sobredetectan, sugiere criterios de exclusión o ejemplos negativos.
4. Sé específico: cita ejemplos del texto donde marcó incorrectamente y explica por qué no es entidad.

Responde con un informe BREVE (máximo 400 palabras) con consejos accionables.
NO generes prompts, solo el análisis y las recomendaciones."""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "Analiza los falsos positivos y genera recomendaciones."},
    ]
    return call_api(optimizer_model, messages, temperature=0.4)
