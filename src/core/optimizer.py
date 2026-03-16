"""Generación y optimización del pipeline de prompts."""
import random
import re

from .api import call_api


def bootstrap_prompt(task_hint, entity_types, worker_model, optimizer_model):
    entity_types_str = "\n".join(f"- {t}" for t in sorted(entity_types))

    system_msg = f"""Eres un experto en ingeniería de prompts.

Vas a diseñar el prompt de sistema para un modelo de lenguaje ({worker_model}) cuya tarea es:
{task_hint}

El modelo recibirá un texto y debe devolver ESE MISMO TEXTO con las entidades anotadas inline
usando el formato: [**ETIQUETA: texto exacto**]

Ejemplo de salida esperada:
Nombre: [**NOMBRE_SUJETO_ASISTENCIA: Pedro**].\nFecha de nacimiento: [**FECHAS: 10/10/1963**].

Etiquetas que debe detectar:
{entity_types_str}

Genera un prompt de sistema claro, conciso y efectivo para {worker_model}.
El prompt debe:
- Explicar exactamente qué hacer y el formato de salida
- Incluir el ejemplo de formato inline
- Ser específico sobre qué tipos de entidades buscar
- Indicar que debe devolver el texto íntegro con las marcas insertadas

Responde ÚNICAMENTE con el texto del prompt, sin explicaciones previas."""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "Genera el prompt inicial."},
    ]
    print(f"  Generando prompt inicial con {optimizer_model} ...")
    return call_api(optimizer_model, messages, temperature=0.7)


def optimize_pipeline(
    prompts,
    results,
    context_doc,
    metric_name,
    optimizer_model,
    num_reviews,
    max_reviews=3,
    stagnation_cycles=0,
    review_step=5,
    strategy_hint=None,
    review_alt_hints=None,
    advisor_reports=None,
    rejected_feedback=None,
):
    """
    El optimizador recibe toda la cadena de prompts y decide cómo mejorarla.
    Puede:
      - Modificar el prompt principal
      - Modificar cualquier review existente
      - Añadir una nueva review (si no se ha llegado al máximo y stagnation >= review_step)

    prompts: dict con claves 'main', 'review_1', 'review_2', 'review_3' (las que existan)
    Devuelve dict con los mismos campos actualizados.
    """
    results_text = ""
    for r in results:
        results_text += f"\n{'─'*40}\nArchivo: {r['name']}\n"
        results_text += f"Texto original (primeros 500 chars):\n{r['text'][:500]}\n\n"
        results_text += "Etiquetas reales (ground truth):\n"
        for etype, etext in r["entities"]:
            results_text += f"  {etype}: {etext}\n"
        results_text += f"\nRespuesta del modelo (final de la cadena):\n{r['model_output']}\n\n"
        results_text += "Etiquetas predichas (parseadas):\n"
        for etype, etext in r["predicted"]:
            results_text += f"  {etype}: {etext}\n"
        m = r["metrics"]
        results_text += (
            f"\nMétricas: P={m['precision']:.3f} R={m['recall']:.3f} "
            f"F1={m['f1']:.3f} (TP={m['tp']} FP={m['fp']} FN={m['fn']})\n"
        )

    avg = sum(r["metrics"][metric_name] for r in results) / len(results)
    context_block = f"\nDOCUMENTACIÓN DE APOYO:\n{context_doc}\n" if context_doc else ""

    strategy_block = ""
    if strategy_hint:
        strategy_block = (
            f"\n💡 ENFOQUE SUGERIDO PARA ESTA ITERACIÓN:\n{strategy_hint}\n"
            f"Incorpora este enfoque en los prompts que generes cuando sea relevante.\n"
        )

    rejected_block = ""
    if rejected_feedback:
        rejected_block = (
            "\n⛔ FEEDBACK DE INTENTOS RECHAZADOS EN ESTE MISMO BATCH:\n"
            f"{rejected_feedback}\n"
            "Debes evitar repetir esos cambios tal cual y proponer una alternativa distinta.\n"
        )

    # Construir bloque con la cadena actual
    chain_block = "CADENA DE PROMPTS ACTUAL:\n"
    chain_block += f"[main]\n{prompts['main']}\n"
    for i in range(1, num_reviews + 1):
        chain_block += f"\n[review_{i}]\n{prompts.get(f'review_{i}', '(vacío)')}\n"
        if review_alt_hints and i in review_alt_hints:
            h = review_alt_hints[i]
            chain_block += (
                f"  ↻ ALTERNATIVA SUGERIDA para review_{i} (lleva ciclos estancada):\n"
                f"  '{h['name']}': {h['text']}\n"
                f"  ➡ Considera REDISEÑAR completamente este prompt con este enfoque.\n"
            )

    # Instrucción sobre si puede añadir nueva review
    can_add = num_reviews < max_reviews and stagnation_cycles >= review_step
    review_instruction = ""
    if can_add:
        review_instruction = (
            f"\n⚠ El sistema lleva {stagnation_cycles} ciclos sin mejorar. "
            f"PUEDES añadir una nueva review (review_{num_reviews + 1}) si lo consideras útil.\n"
            f"  Úsala para: revisar/corregir la salida del paso anterior, dividir subtareas, "
            f"añadir un filtro de falsos positivos, etc.\n"
        )
    else:
        review_instruction = (
            f"  Reviews activas: {num_reviews}/{max_reviews}. "
            + ("(máximo alcanzado)" if num_reviews >= max_reviews else f"Nueva review disponible tras {review_step - stagnation_cycles % review_step} ciclos más sin mejora.")
            + "\n"
        )

    # Construir el formato de respuesta esperado dinámicamente
    format_example = "---CHANGES---\n<explica brevemente qué cambias y por qué, máx 5 líneas>\n"
    format_example += "---PROMPT:main---\n<nuevo prompt principal>\n"
    for i in range(1, num_reviews + 1):
        format_example += f"---PROMPT:review_{i}---\n<nuevo prompt review {i}>\n"
    if can_add:
        format_example += f"---PROMPT:review_{num_reviews + 1}---\n<prompt de la nueva review (omite este bloque si no la añades)>\n"

    # Bloque de informes de subagentes advisors
    advisor_block = ""
    if advisor_reports:
        if advisor_reports.get("recall"):
            advisor_block += (
                "\n\U0001F50D INFORME DEL ADVISOR DE RECALL:\n"
                f"{advisor_reports['recall']}\n"
            )
        if advisor_reports.get("precision"):
            advisor_block += (
                "\n\U0001F6E1 INFORME DEL ADVISOR DE PRECISIÓN:\n"
                f"{advisor_reports['precision']}\n"
            )

    system_msg = f"""Eres un experto en ingeniería de prompts.

Tu tarea es analizar los resultados de un pipeline de etiquetado/clasificación de texto y mejorar los prompts para maximizar la métrica '{metric_name}'.
{context_block}{strategy_block}{advisor_block}{rejected_block}
{chain_block}
RESULTADOS DE LAS MUESTRAS DE PRUEBA:
{results_text}
MÉTRICA PROMEDIO ACTUAL ({metric_name}): {avg:.4f}
{review_instruction}
INSTRUCCIONES:
1. Analiza los errores en la salida FINAL de la cadena (falsos positivos y falsos negativos).
2. Identifica en qué paso del pipeline fallan (¿error del paso principal, o podría corregirse en review?).
3. Mejora los prompts de forma coordinada.
4. Si alguna review tiene un ↻ ALTERNATIVA SUGERIDA, rediseña ese prompt de review con ese enfoque.
5. Ten en cuenta los informes de los advisors especializados (si los hay) como guía prioritaria.
6. Si hay feedback de intentos rechazados, NO repitas esos cambios y corrige explícitamente las regresiones reportadas.

FORMATO DE RESPUESTA OBLIGATORIO (escribe EXACTAMENTE estos separadores):
{format_example}"""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "Genera el pipeline de prompts optimizado."},
    ]
    raw = call_api(optimizer_model, messages, temperature=0.7)

    # Parsear la respuesta
    updated = dict(prompts)  # copia
    changes = ""

    if "---CHANGES---" in raw:
        after_changes = raw.split("---CHANGES---", 1)[1]
        # Extraer texto de cambios (hasta el primer ---PROMPT:*---)
        changes_match = re.split(r"---PROMPT:[a-z_0-9]+---", after_changes, maxsplit=1)
        changes = changes_match[0].strip()
    else:
        changes = "(sin bloque ---CHANGES---)"

    # Extraer cada prompt por su clave
    for match in re.finditer(r"---PROMPT:([a-z_0-9]+)---\s*([\s\S]*?)(?=---PROMPT:|$)", raw):
        key = match.group(1).strip()
        value = match.group(2).strip()
        if len(value) >= 10:
            updated[key] = value

    return updated, changes


def optimize_prompt(current_prompt, results, context_doc, metric_name, optimizer_model):
    results_text = ""
    for r in results:
        results_text += f"\n{'─'*40}\nArchivo: {r['name']}\n"
        results_text += f"Texto original (primeros 500 chars):\n{r['text'][:500]}\n\n"
        results_text += "Etiquetas reales (ground truth):\n"
        for etype, etext in r["entities"]:
            results_text += f"  {etype}: {etext}\n"
        results_text += f"\nRespuesta del modelo:\n{r['model_output']}\n\n"
        results_text += "Etiquetas predichas (parseadas):\n"
        for etype, etext in r["predicted"]:
            results_text += f"  {etype}: {etext}\n"
        m = r["metrics"]
        results_text += (
            f"\nMétricas: P={m['precision']:.3f} R={m['recall']:.3f} "
            f"F1={m['f1']:.3f} (TP={m['tp']} FP={m['fp']} FN={m['fn']})\n"
        )

    avg = sum(r["metrics"][metric_name] for r in results) / len(results)

    context_block = ""
    if context_doc:
        context_block = f"\nDOCUMENTACIÓN DE APOYO:\n{context_doc}\n"

    system_msg = f"""Eres un experto en ingeniería de prompts.

Tu tarea es analizar los resultados de un sistema de etiquetado/clasificación de texto y generar un MEJOR prompt que maximice la métrica '{metric_name}'.
{context_block}
PROMPT ACTUAL:
{current_prompt}

RESULTADOS DE LAS MUESTRAS DE PRUEBA:
{results_text}

MÉTRICA PROMEDIO ACTUAL ({metric_name}): {avg:.4f}

INSTRUCCIONES:
1. Analiza los errores (falsos positivos y falsos negativos).
2. Identifica patrones recurrentes en los errores.
3. Genera un prompt mejorado que:
   - Sea claro y específico sobre qué buscar y qué NO.
   - Incluya ejemplos o patrones comunes cuando sea útil.
   - Aborde los errores recurrentes encontrados.
   - Maximice {metric_name}.

FORMATO DE RESPUESTA:
Primero escribe un bloque breve (máximo 5 líneas) explicando qué cambios realizas y por qué.
Luego escribe exactamente la línea:
---PROMPT---
Y a continuación escribe el nuevo prompt completo (será usado directamente como instrucción del sistema)."""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "Genera el prompt optimizado."},
    ]
    return call_api(optimizer_model, messages, temperature=0.7)


def refine_paragraph(prompt_text, optimizer_model):
    """
    Extrae un párrafo aleatorio del prompt y pide al optimizador que lo revise.
    Devuelve el prompt completo con ese párrafo sustituido por la versión revisada.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", prompt_text) if p.strip()]
    if len(paragraphs) < 2:
        return prompt_text

    idx = random.randrange(len(paragraphs))
    target = paragraphs[idx]

    system_msg = f"""Eres un experto en ingeniería de prompts.

Se te muestra un prompt completo y uno de sus párrafos. Tu única tarea es revisar y mejorar ESE PÁRRAFO hacieéndolo más claro, preciso y efectivo en el contexto del prompt completo. No cambies el resto.

PROMPT COMPLETO:
{prompt_text}

PÁRRAFO A REVISAR (párrafo #{idx + 1} de {len(paragraphs)}):
{target}

Responde ÚNCIAMENTE con el texto revisado del párrafo, sin explicaciones ni cabeceras."""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "Revisa y mejora el párrafo."},
    ]
    print(f"    [refinado párrafo #{idx + 1}/{len(paragraphs)}]", end=" ", flush=True)
    revised = call_api(optimizer_model, messages, temperature=0.5)
    if len(revised.strip()) < 5:
        return prompt_text
    paragraphs[idx] = revised.strip()
    return "\n\n".join(paragraphs)
