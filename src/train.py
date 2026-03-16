#!/usr/bin/env python3
"""
Optimización iterativa de prompts mediante evaluación automatizada.

Flujo genérico:
1. Lee el prompt actual y lo usa con un modelo worker (OpenAI / DeepSeek) para procesar N textos
2. Evalúa la salida contra anotaciones ground truth en formato BRAT (.ann)
3. Envía resultados a un modelo optimizador para generar un prompt mejorado
4. Repite n iteraciones buscando maximizar la métrica elegida

Variables de entorno (archivo .env en la raíz del proyecto):
  OPENAI_API_KEY   — para modelos gpt-* (worker por defecto: gpt-3.5-turbo)
  DEEPSEEK_API_KEY — para modelos deepseek-* (optimizador por defecto: deepseek-chat)

Modelos Ollama (locales):
  Usa --worker-model ollama  para ver los modelos instalados y elegir uno de forma interactiva.
  O pasa el nombre directamente, por ejemplo: --worker-model llama3:8b
"""

import argparse
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from core import config
from core.config import load_env, load_strategies
from core.ollama_utils import pick_ollama_model
from core.brat import load_text, save_text, extract_entity_types, parse_model_output, get_training_samples
from core.metrics import compute_match_details, compute_metrics, normalize_overlap_threshold
from core.worker import run_worker
from core.optimizer import bootstrap_prompt, optimize_pipeline, refine_paragraph
from core.advisors import recall_advisor, precision_advisor
from core.logger import log_iteration


def main():
    parser = argparse.ArgumentParser(
        description="Optimización iterativa de prompts con evaluación automática",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Ejemplos:
  # Caso MEDDOCAN por defecto
  python train.py --brat-dir corpus/train/brat --prompt prompt.txt

  # Tarea genérica con contexto extra y 3 iteraciones optimizando recall
  python train.py --brat-dir data/train --prompt mi_prompt.txt --context docs.csv -m recall -n 3

  # Worker Ollama interactivo (muestra los modelos instalados)
  python train.py --brat-dir data/ --prompt p.txt --worker-model ollama

  # Worker Ollama directo (si ya sabes el nombre)
  python train.py --brat-dir data/ --prompt p.txt --worker-model llama3:8b --optimizer-model gpt-4o
"""
    )

    # ── Rutas obligatorias ────────────────────────────────────────────────
    parser.add_argument(
        "--brat-dir", "-d", type=str, required=True,
        help="Carpeta con los pares .txt / .ann (formato BRAT) usados como benchmark",
    )
    parser.add_argument(
        "--prompt", "-p", type=str, required=True,
        help="Archivo de texto con el prompt inicial (se sobreescribirá en cada iteración)",
    )

    # ── Opcionales ───────────────────────────────────────────────────────
    parser.add_argument(
        "--context", "-c", type=str, default=None,
        help="Archivo de contexto adicional (CSV, TXT, etc.) que se pasa al optimizador",
    )
    parser.add_argument(
        "--log", type=str, default=None,
        help="Archivo de log (default: training_log.txt junto al prompt)",
    )

    # ── Modelos ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--worker-model", type=str, default="gpt-3.5-turbo",
        help=(
            "Modelo worker que ejecuta la tarea (default: gpt-3.5-turbo). "
            "Prefijos: gpt-/o1-/o3-/o4-→OpenAI; deepseek-*→DeepSeek; "
            "'ollama'→muestra los modelos Ollama instalados para elegir; "
            "cualquier otro nombre (p.ej. llama3:8b)→Ollama directo."
        ),
    )
    parser.add_argument(
        "--optimizer-model", type=str, default="deepseek-chat",
        help=(
            "Modelo que optimiza el prompt (default: deepseek-chat). "
            "Prefijos: gpt-/o1-/o3-/o4-→OpenAI; deepseek-*→DeepSeek; "
            "'ollama'→muestra los modelos Ollama instalados para elegir; "
            "cualquier otro nombre→Ollama directo."
        ),
    )
    parser.add_argument(
        "--no-think", action="store_true", default=False,
        help=(
            "Deshabilita el modo 'think' en modelos Ollama que lo soporten "
            "(ej. qwen3.5:35b). Envía think=false en el payload."
        ),
    )
    parser.add_argument(
        "--ollama-num-ctx", type=int, default=None,
        help=(
            "Tamaño del contexto para Ollama (num_ctx). "
            f"Por defecto {16384}. Aumténtalo si el prompt supera ese límite."
        ),
    )

    # ── Hiperparámetros ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "-n", "--iterations", type=int, default=100,
        help="Número de iteraciones de optimización (default: 5)",
    )
    parser.add_argument(
        "-s", "--samples", type=int, default=10,
        help="Número de textos por iteración (default: 10)",
    )
    parser.add_argument(
        "-m", "--metric", choices=["f1", "precision", "recall"], default="f1",
        help="Métrica a optimizar (default: f1)",
    )
    parser.add_argument(
        "--match-mode", choices=["strict", "relaxed"], default="strict",
        help=(
            "Modo de matching para evaluar entidades. "
            "strict=coincidencia exacta; relaxed=coincidencia por inclusión textual."
        ),
    )
    parser.add_argument(
        "--min-overlap", type=float, default=50.0, metavar="X",
        help=(
            "Umbral mínimo de solapamiento para relaxed. "
            "Acepta 0..1 o 0..100 (default: 50)."
        ),
    )
    parser.add_argument(
        "--parallel", type=int, default=1, metavar="N",
        help="Llamadas worker en paralelo (default: 1, secuencial. Usar >1 solo con modelos cloud)",
    )
    parser.add_argument(
        "--patience", type=int, default=30, metavar="N",
        help="Early stopping: detiene el entrenamiento tras N ciclos consecutivos sin mejora (default: 30)",
    )
    parser.add_argument(
        "--review-step", type=int, default=5, metavar="N",
        help="Añade una nueva review al pipeline cada N ciclos sin mejora (default: 5, max 3 reviews)",
    )
    parser.add_argument(
        "--max-reviews", type=int, default=3, metavar="N",
        help="Número máximo de reviews encadenadas (default: 3)",
    )
    parser.add_argument(
        "--accept-epsilon", type=float, default=0.002, metavar="X",
        help=(
            "Mejora mínima requerida para aceptar un candidato en el mismo batch. "
            "Se acepta solo si Δmétrica > epsilon (default: 0.002)."
        ),
    )
    parser.add_argument(
        "--candidate-attempts", type=int, default=1, metavar="N",
        help=(
            "Intentos máximos de mutación+evaluación por iteración sobre el mismo batch "
            "(default: 1)."
        ),
    )
    parser.add_argument(
        "--strategies", type=str, default=None, metavar="ARCHIVO",
        help="Archivo .md con estrategias de optimización del prompt principal (default: optimization_strategies.md)",
    )
    parser.add_argument(
        "--review-alternatives", type=str, default=None, metavar="ARCHIVO",
        help="Archivo .md con alternativas de uso para slots de review (default: review_alternatives.md en raíz)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Semilla para reproducibilidad",
    )
    args = parser.parse_args()

    if args.accept_epsilon < 0:
        sys.exit("Error: --accept-epsilon debe ser >= 0")
    if args.candidate_attempts < 1:
        sys.exit("Error: --candidate-attempts debe ser >= 1")

    try:
        overlap_threshold = normalize_overlap_threshold(args.min_overlap)
    except ValueError as e:
        sys.exit(f"Error: {e}")

    # ── Cargar API keys desde .env ───────────────────────────────────────
    config.API_KEYS.update(load_env())

    # ── Modo no-think ────────────────────────────────────────────────────
    config.NO_THINK = args.no_think
    if config.NO_THINK:
        print("  ⚙ Modo no-think activado: think=false se enviará a modelos Ollama")

    # ── num_ctx de Ollama ────────────────────────────────────────────────
    if args.ollama_num_ctx is not None:
        config.OLLAMA_NUM_CTX = args.ollama_num_ctx
    print(f"  ⚙ Ollama num_ctx={config.OLLAMA_NUM_CTX} (usa --ollama-num-ctx para cambiar)")

    # ── Selección interactiva de modelo Ollama ───────────────────────────────────
    if args.worker_model.lower() == "ollama":
        args.worker_model = pick_ollama_model()
        print(f"  Worker Ollama seleccionado: {args.worker_model}")
    elif args.optimizer_model.lower() == "ollama":
        args.optimizer_model = pick_ollama_model()
        print(f"  Optimizador Ollama seleccionado: {args.optimizer_model}")

    # ── Cargar estrategias de optimización ───────────────────────────────────
    strategies = load_strategies(args.strategies)
    review_alts = load_strategies(
        args.review_alternatives
        or str(Path(__file__).resolve().parent.parent / "review_alternatives.md")
    )

    # ── Resolver rutas ───────────────────────────────────────────────────────────────
    brat_dir = Path(args.brat_dir).resolve()
    prompt_file = Path(args.prompt).resolve()
    log_file = Path(args.log).resolve() if args.log else prompt_file.parent / "training_log.txt"
    context_doc = load_text(Path(args.context).resolve()) if args.context else None

    if not brat_dir.is_dir():
        sys.exit(f"Error: la carpeta BRAT no existe: {brat_dir}")
    if not prompt_file.is_file():
        sys.exit(f"Error: el archivo de prompt no existe: {prompt_file}")

    if args.seed is not None:
        random.seed(args.seed)

    # ── Descubrir tipos de entidad del corpus ──────────────────────────────────────────
    entity_types = extract_entity_types(brat_dir)
    if not entity_types:
        sys.exit("Error: no se encontraron entidades en los .ann de la carpeta BRAT")

    print("╔═════════════════════════════════════════════════╗")
    print("║  Optimización iterativa de prompt                ║")
    print("╚═════════════════════════════════════════════════╝")
    print(f"  BRAT dir      : {brat_dir}")
    print(f"  Prompt        : {prompt_file}")
    print(f"  Contexto      : {args.context or '(ninguno)'}")
    print(f"  Log           : {log_file}")
    print(f"  Worker        : {args.worker_model}")
    print(f"  Optimizador   : {args.optimizer_model}")
    print(f"  Iteraciones   : {args.iterations}")
    print(f"  Muestras      : {args.samples}")
    print(f"  Métrica       : {args.metric}")
    print(f"  Match mode    : {args.match_mode}")
    if args.match_mode == "relaxed":
        print(f"  Min overlap   : {overlap_threshold:.0%} ({overlap_threshold:.2f})")
    print(f"  Entidades     : {len(entity_types)} tipos detectados")
    print(f"  Parallelismo  : {args.parallel} workers")
    print(f"  Patience      : {args.patience} ciclos sin mejora")
    print(f"  Review step   : cada {args.review_step} ciclos sin mejora (máx {args.max_reviews} reviews)")
    print(f"  Accept ε      : {args.accept_epsilon:.4f} (Δ{args.metric} > ε)")
    print(f"  Cand attempts : {args.candidate_attempts} por iteración")
    strat_src = args.strategies or "optimization_strategies.md"
    print(f"  Estrategias   : {len(strategies)} cargadas ({strat_src})")
    alt_src = args.review_alternatives or "review_alternatives.md"
    print(f"  Rev. alternativas: {len(review_alts)} cargadas ({alt_src})")
    print()

    # ── Bootstrap: generar prompt inicial con el optimizador ──────────────────────────
    task_hint = load_text(prompt_file)
    print("\n─── Bootstrap ───────────────────────────────────────────")
    print(f"  Tarea base: {task_hint[:200]}")
    try:
        initial_prompt = bootstrap_prompt(
            task_hint, entity_types, args.worker_model, args.optimizer_model
        )
        if len(initial_prompt) >= 10:
            save_text(prompt_file, initial_prompt)
            print(f"  ✓ Prompt inicial generado ({len(initial_prompt)} chars)")
            print(f"  Preview: {initial_prompt[:300]}{'...' if len(initial_prompt) > 300 else ''}")
        else:
            print(f"  ⚠ Bootstrap devolvió prompt vacío, usando hint original.")
    except Exception as e:
        print(f"  ⚠ Error en bootstrap: {e}. Usando hint original.")
    print("─────────────────────────────────────────────────────\n")

    best_score = 0.0
    best_prompt_file = prompt_file.parent / (prompt_file.stem + "_best" + prompt_file.suffix)
    no_improve_streak = 0
    prev_iter_score = 0.0
    last_improvement_delta = 0.0
    strat_idx = 0

    # Pipeline activo (best): 'main' siempre presente, 'review_N' se añaden dinámicamente
    best_pipeline = {"main": load_text(prompt_file)}
    num_reviews = 0
    review_stagnation = {}   # review_idx -> ciclos sin mejora desde que se añadió
    review_alt_counters = {} # review_idx -> índice rotatorio de alternativas
    max_zero_retries = 3

    def is_transient_zero_result(metrics):
        return (
            metrics["f1"] == 0
            and metrics["precision"] == 0
            and metrics["recall"] == 0
            and metrics["tp"] == 0
            and metrics["fp"] == 0
            and metrics["fn"] > 0
        )

    def evaluate_pipeline(eval_pipeline, samples, phase_label):
        eval_prompt = eval_pipeline["main"]
        eval_num_reviews = sum(1 for k in eval_pipeline if k.startswith("review_"))
        eval_review_list = [eval_pipeline[f"review_{i}"] for i in range(1, eval_num_reviews + 1)]
        review_label = f" + {eval_num_reviews} review(s)" if eval_num_reviews > 0 else ""

        print(f"  {phase_label}: prompt={len(eval_prompt)} chars{review_label}")
        worker_start = time.time()
        mode = "paralelo" if args.parallel > 1 else "secuencial"
        print(f"  {phase_label}: ejecutando {args.worker_model} ({mode}) sobre {len(samples)} muestras...")

        def process_sample(idx, sample):
            total_elapsed = 0.0
            for attempt in range(1, max_zero_retries + 2):
                t0 = time.time()
                output = run_worker(
                    eval_prompt, sample["text"], entity_types, args.worker_model,
                    review_prompts=eval_review_list,
                )
                predicted = parse_model_output(output, entity_types)
                metrics = compute_metrics(
                    predicted,
                    sample["entities"],
                    match_mode=args.match_mode,
                    min_overlap=overlap_threshold,
                )
                elapsed = time.time() - t0
                total_elapsed += elapsed

                if is_transient_zero_result(metrics) and attempt <= max_zero_retries:
                    print(
                        f"         ⚠ [{phase_label}] {sample['name']}: salida anómala en cero "
                        f"(FN={metrics['fn']}); reintento {attempt}/{max_zero_retries}",
                        flush=True,
                    )
                    continue

                return idx, {
                    **sample,
                    "model_output": output,
                    "predicted": predicted,
                    "metrics": metrics,
                    "time": total_elapsed,
                    "zero_retries": attempt - 1,
                }

        def print_result(completed, total, name, result):
            m = result["metrics"]
            retry_note = f"  (reintentos={result['zero_retries']})" if result.get("zero_retries", 0) > 0 else ""
            print(
                f"  [{phase_label} {completed}/{total}] {name}  "
                f"F1={m['f1']:.3f}  P={m['precision']:.3f}  "
                f"R={m['recall']:.3f}  "
                f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})  "
                f"[{result['time']:.1f}s]{retry_note}",
                flush=True,
            )

        results = []
        if args.parallel <= 1:
            # Secuencial: ves cada muestra conforme termina
            for i, sample in enumerate(samples):
                print(f"  [{phase_label} {i+1}/{len(samples)}] {sample['name']} ...", flush=True)
                try:
                    _, result = process_sample(i, sample)
                    results.append(result)
                    m = result["metrics"]
                    print(
                        f"         F1={m['f1']:.3f}  P={m['precision']:.3f}  "
                        f"R={m['recall']:.3f}  "
                        f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})  "
                        f"[{result['time']:.1f}s]"
                        f"{'  (reintentos=' + str(result.get('zero_retries', 0)) + ')' if result.get('zero_retries', 0) > 0 else ''}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"         ERROR: {e}", flush=True)
        else:
            # Paralelo: para modelos cloud
            completed = 0
            slot_results = [None] * len(samples)
            with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                futures = {
                    pool.submit(process_sample, i, s): i for i, s in enumerate(samples)
                }
                for future in as_completed(futures):
                    i = futures[future]
                    name = samples[i]["name"]
                    completed += 1
                    try:
                        idx, result = future.result()
                        slot_results[idx] = result
                        print_result(completed, len(samples), name, result)
                    except Exception as e:
                        print(f"  [{phase_label} {completed}/{len(samples)}] {name}  ERROR: {e}", flush=True)
            results = [r for r in slot_results if r is not None]

        worker_elapsed = time.time() - worker_start
        print(f"  {phase_label}: batch completado en {worker_elapsed:.1f}s")
        return results, worker_elapsed

    def summarize_results(results):
        avg_metrics = {
            "precision": sum(r["metrics"]["precision"] for r in results) / len(results),
            "recall": sum(r["metrics"]["recall"] for r in results) / len(results),
            "f1": sum(r["metrics"]["f1"] for r in results) / len(results),
        }
        total_tp = sum(r["metrics"]["tp"] for r in results)
        total_fp = sum(r["metrics"]["fp"] for r in results)
        total_fn = sum(r["metrics"]["fn"] for r in results)

        fn_by_type = {}
        fp_by_type = {}
        for r in results:
            match_details = compute_match_details(
                r["predicted"],
                r["entities"],
                match_mode=args.match_mode,
                min_overlap=overlap_threshold,
            )
            for etype, _ in match_details["unmatched_gt"]:
                fn_by_type[etype] = fn_by_type.get(etype, 0) + 1
            for etype, _ in match_details["unmatched_pred"]:
                fp_by_type[etype] = fp_by_type.get(etype, 0) + 1

        return {
            "avg_metrics": avg_metrics,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "fn_by_type": fn_by_type,
            "fp_by_type": fp_by_type,
            "score": avg_metrics[args.metric],
        }

    def print_summary(label, summary, suffix=""):
        avg = summary["avg_metrics"]
        print(
            f"\n  ► {label}: P={avg['precision']:.4f}  "
            f"R={avg['recall']:.4f}  F1={avg['f1']:.4f}{suffix}"
        )
        print(f"    Totales: TP={summary['total_tp']}  FP={summary['total_fp']}  FN={summary['total_fn']}")
        if summary["fn_by_type"]:
            top_fn = sorted(summary["fn_by_type"].items(), key=lambda x: -x[1])[:5]
            print(f"    Top FN (no detectados): {', '.join(f'{t}={c}' for t, c in top_fn)}")
        if summary["fp_by_type"]:
            top_fp = sorted(summary["fp_by_type"].items(), key=lambda x: -x[1])[:5]
            print(f"    Top FP (falsos positivos): {', '.join(f'{t}={c}' for t, c in top_fp)}")

    def build_rejection_feedback(attempt_idx, changes_text, parent_summary, candidate_summary, delta_score):
        metric_delta_p = candidate_summary["avg_metrics"]["precision"] - parent_summary["avg_metrics"]["precision"]
        metric_delta_r = candidate_summary["avg_metrics"]["recall"] - parent_summary["avg_metrics"]["recall"]
        metric_delta_f1 = candidate_summary["avg_metrics"]["f1"] - parent_summary["avg_metrics"]["f1"]

        def top_worsening(candidate_counts, parent_counts):
            deltas = []
            for etype in (set(candidate_counts.keys()) | set(parent_counts.keys())):
                diff = candidate_counts.get(etype, 0) - parent_counts.get(etype, 0)
                if diff > 0:
                    deltas.append((etype, diff))
            deltas.sort(key=lambda x: -x[1])
            return deltas[:5]

        worse_fn = top_worsening(candidate_summary["fn_by_type"], parent_summary["fn_by_type"])
        worse_fp = top_worsening(candidate_summary["fp_by_type"], parent_summary["fp_by_type"])
        worse_fn_text = ", ".join(f"{t}=+{d}" for t, d in worse_fn) if worse_fn else "ninguna"
        worse_fp_text = ", ".join(f"{t}=+{d}" for t, d in worse_fp) if worse_fp else "ninguna"

        feedback_lines = [
            f"Intento rechazado #{attempt_idx}.",
            f"Cambios propuestos: {changes_text}",
            (
                f"Delta en batch (métrica={args.metric}): {delta_score:+.4f} "
                f"(epsilon={args.accept_epsilon:.4f}, requerido > epsilon)"
            ),
            f"Delta P={metric_delta_p:+.4f}, Delta R={metric_delta_r:+.4f}, Delta F1={metric_delta_f1:+.4f}",
            (
                f"Totales parent TP/FP/FN={parent_summary['total_tp']}/{parent_summary['total_fp']}/{parent_summary['total_fn']} | "
                f"candidate={candidate_summary['total_tp']}/{candidate_summary['total_fp']}/{candidate_summary['total_fn']}"
            ),
            f"Regresiones de FN por tipo: {worse_fn_text}",
            f"Regresiones de FP por tipo: {worse_fp_text}",
            "No repitas estos cambios tal cual; propone una alternativa distinta.",
        ]
        return "\n".join(feedback_lines)

    for iteration in range(1, args.iterations + 1):
        iter_start = time.time()
        print(f"\n┌── Iteración {iteration}/{args.iterations} ──┐")

        parent_pipeline = dict(best_pipeline)
        num_reviews = sum(1 for k in parent_pipeline if k.startswith("review_"))
        review_label = f" + {num_reviews} review(s)" if num_reviews > 0 else ""
        print(f"  Prompt activo (best): {len(parent_pipeline['main'])} chars{review_label}")

        # 1. Seleccionar muestras
        samples = get_training_samples(brat_dir, args.samples)
        print(f"  Muestras seleccionadas: {len(samples)}")

        # 2. Evaluar baseline (parent) en este batch
        parent_results, parent_worker_elapsed = evaluate_pipeline(parent_pipeline, samples, "BASELINE")
        if not parent_results:
            print("  ⚠ Sin resultados, saltando iteración.")
            continue

        parent_summary = summarize_results(parent_results)
        parent_score = parent_summary["score"]
        best_score = max(best_score, parent_score)

        print_summary("Baseline", parent_summary)

        # 3. Señales de estancamiento para guiar la optimización
        current_delta = parent_score - prev_iter_score
        low_growth = (
            iteration > 1
            and 0 < current_delta < last_improvement_delta * 0.25
            and last_improvement_delta > 0
        )
        inject_strategy = strategies and (
            (no_improve_streak > 0 and no_improve_streak % 2 == 0) or low_growth
        )
        strategy_hint = None
        if inject_strategy:
            strategy_hint = strategies[strat_idx % len(strategies)]
            strat_idx += 1
            reason = "bajo crecimiento" if low_growth else f"racha {no_improve_streak}"
            print(f"  💡 Estrategia [{strat_idx}] ({reason}): «{strategy_hint['name']}»")

        if current_delta > 0:
            last_improvement_delta = current_delta
        prev_iter_score = parent_score

        # 4. Construir hints de alternativas para reviews estancadas
        review_alt_hints = {}
        if review_alts:
            for i in range(1, num_reviews + 1):
                if review_stagnation.get(i, 0) >= 1:
                    alt_idx = review_alt_counters.get(i, 0)
                    review_alt_hints[i] = review_alts[alt_idx % len(review_alts)]
                    print(f"  ↻ Review_{i} alternativa [{alt_idx + 1}/{len(review_alts)}]: «{review_alt_hints[i]['name']}»")

        reviews_label = f" ({num_reviews} reviews activas)" if num_reviews > 0 else ""
        print(f"  Optimizando pipeline con {args.optimizer_model}{reviews_label} ...")

        # ── Subagentes advisors ──────────────────────────────────────
        advisor_reports = {}
        if args.metric == "f1":
            # Para F1: solo el advisor de la métrica más baja
            _use_recall = parent_summary["avg_metrics"]["recall"] <= parent_summary["avg_metrics"]["precision"]
            _adv_name = "recall" if _use_recall else "precision"
            print(f"  📊 F1 mode → métrica más baja: {_adv_name} "
                  f"(R={parent_summary['avg_metrics']['recall']:.4f} P={parent_summary['avg_metrics']['precision']:.4f})")
            if _use_recall:
                print(f"  🔍 Advisor de recall analizando FN ...")
                try:
                    adv = recall_advisor(parent_pipeline, parent_results, args.optimizer_model)
                    if adv.strip():
                        advisor_reports["recall"] = adv
                        print(f"     Informe de recall: {len(adv)} chars")
                except Exception as e:
                    print(f"     ⚠ Error en advisor recall: {e}")
            else:
                print(f"  🛡 Advisor de precisión analizando FP ...")
                try:
                    adv = precision_advisor(parent_pipeline, parent_results, args.optimizer_model)
                    if adv.strip():
                        advisor_reports["precision"] = adv
                        print(f"     Informe de precisión: {len(adv)} chars")
                except Exception as e:
                    print(f"     ⚠ Error en advisor precisión: {e}")
        elif args.metric == "recall":
            print(f"  🔍 Advisor de recall analizando FN ...")
            try:
                adv = recall_advisor(parent_pipeline, parent_results, args.optimizer_model)
                if adv.strip():
                    advisor_reports["recall"] = adv
                    print(f"     Informe de recall: {len(adv)} chars")
            except Exception as e:
                print(f"     ⚠ Error en advisor recall: {e}")
        elif args.metric == "precision":
            print(f"  🛡 Advisor de precisión analizando FP ...")
            try:
                adv = precision_advisor(parent_pipeline, parent_results, args.optimizer_model)
                if adv.strip():
                    advisor_reports["precision"] = adv
                    print(f"     Informe de precisión: {len(adv)} chars")
            except Exception as e:
                print(f"     ⚠ Error en advisor precisión: {e}")

        # 6. Generar/evaluar candidato(s) en el mismo batch
        optimizer_elapsed_total = 0.0
        candidate_worker_elapsed_total = 0.0
        rejected_feedback_entries = []
        accepted_pipeline = None
        accepted_changes = ""
        accepted_summary = None
        accepted_attempt = None
        accepted_delta = 0.0

        for cand_attempt in range(1, args.candidate_attempts + 1):
            try:
                opt_call_start = time.time()
                updated_pipeline, changes = optimize_pipeline(
                    parent_pipeline,
                    parent_results,
                    context_doc,
                    args.metric,
                    args.optimizer_model,
                    num_reviews=num_reviews,
                    max_reviews=args.max_reviews,
                    stagnation_cycles=no_improve_streak,
                    review_step=args.review_step,
                    strategy_hint=strategy_hint["text"] if strategy_hint else None,
                    review_alt_hints=review_alt_hints if review_alt_hints else None,
                    advisor_reports=advisor_reports if advisor_reports else None,
                    rejected_feedback="\n\n".join(rejected_feedback_entries[-2:]) if rejected_feedback_entries else None,
                )
                optimizer_elapsed_total += time.time() - opt_call_start

                new_main = updated_pipeline.get("main", "").strip()
                if len(new_main) < 10:
                    note = f"Intento {cand_attempt}: prompt principal vacío tras optimización."
                    print(f"  ⚠ {note}")
                    rejected_feedback_entries.append(note)
                    continue

                # Múltiplo de 3 fallos: refinar un párrafo aleatorio del prompt principal candidato
                if no_improve_streak > 0 and no_improve_streak % 3 == 0:
                    print(f"  🔍 Múltiplo de 3 ciclos sin mejora — refinando párrafo aleatorio (intento {cand_attempt})...")
                    ref_start = time.time()
                    refined_main = refine_paragraph(updated_pipeline["main"], args.optimizer_model)
                    optimizer_elapsed_total += time.time() - ref_start
                    if refined_main != updated_pipeline["main"]:
                        updated_pipeline["main"] = refined_main
                        print(f"  ✓ Párrafo refinado ({len(refined_main)} chars total)")

                new_review_key = f"review_{num_reviews + 1}"
                if new_review_key in updated_pipeline and num_reviews < args.max_reviews:
                    _cand_review_idx = num_reviews + 1
                    print(f"  ✚ Candidato intento {cand_attempt} propone review_{_cand_review_idx} ({len(updated_pipeline[new_review_key])} chars)")

                phase_label = f"CANDIDATO {cand_attempt}/{args.candidate_attempts}"
                candidate_results, candidate_worker_elapsed = evaluate_pipeline(updated_pipeline, samples, phase_label)
                candidate_worker_elapsed_total += candidate_worker_elapsed
                if not candidate_results:
                    note = f"Intento {cand_attempt}: sin resultados de evaluación del candidato."
                    print(f"  ⚠ {note}")
                    rejected_feedback_entries.append(note)
                    continue

                candidate_summary = summarize_results(candidate_results)
                candidate_score = candidate_summary["score"]
                best_score = max(best_score, candidate_score)
                delta_score = candidate_score - parent_score

                print_summary(
                    f"Candidato intento {cand_attempt}",
                    candidate_summary,
                    suffix=f"  (Δ{args.metric}={delta_score:+.4f}, ε={args.accept_epsilon:.4f})",
                )

                if delta_score > args.accept_epsilon:
                    accepted_pipeline = dict(updated_pipeline)
                    accepted_changes = changes
                    accepted_summary = candidate_summary
                    accepted_attempt = cand_attempt
                    accepted_delta = delta_score
                    print(
                        f"  ✅ Candidato aceptado en intento {cand_attempt}/{args.candidate_attempts}: "
                        f"Δ{args.metric}={delta_score:+.4f} > ε={args.accept_epsilon:.4f}"
                    )
                    break

                print(
                    f"  ✗ Intento {cand_attempt} rechazado: "
                    f"Δ{args.metric}={delta_score:+.4f} <= ε={args.accept_epsilon:.4f}"
                )
                rejected_feedback_entries.append(
                    build_rejection_feedback(cand_attempt, changes, parent_summary, candidate_summary, delta_score)
                )

            except Exception as e:
                note = f"Intento {cand_attempt}: error al optimizar/evaluar candidato: {e}"
                print(f"  ✗ {note}")
                rejected_feedback_entries.append(note)

        # 7. Decidir aceptación y actualizar estado
        if accepted_pipeline is not None:
            best_pipeline = dict(accepted_pipeline)
            save_text(best_prompt_file, best_pipeline["main"])
            no_improve_streak = 0

            new_best_reviews = sum(1 for k in best_pipeline if k.startswith("review_"))
            review_stagnation = {i: 0 for i in range(1, new_best_reviews + 1)}
            review_alt_counters = {
                i: review_alt_counters.get(i, 0)
                for i in range(1, new_best_reviews + 1)
            }
            for i in range(1, new_best_reviews + 1):
                review_alt_counters.setdefault(i, 0)

            for i in review_alt_hints:
                if i in review_alt_counters:
                    review_alt_counters[i] = review_alt_counters.get(i, 0) + 1

            active_pipeline = best_pipeline
            active_summary = accepted_summary
            changes_to_log = (
                f"ACEPTADO intento {accepted_attempt}/{args.candidate_attempts} "
                f"(Δ{args.metric}={accepted_delta:+.4f} > ε={args.accept_epsilon:.4f})\n"
                f"Cambios:\n{accepted_changes}"
            )
        else:
            no_improve_streak += 1
            for k in review_stagnation:
                review_stagnation[k] += 1

            active_pipeline = parent_pipeline
            active_summary = parent_summary
            changes_to_log = (
                f"RECHAZADO: ningún candidato superó ε={args.accept_epsilon:.4f} "
                f"tras {args.candidate_attempts} intento(s). Se conserva el pipeline activo."
            )
            if rejected_feedback_entries:
                changes_to_log += (
                    "\n\nÚltimo feedback de rechazo:\n"
                    + rejected_feedback_entries[-1][:2000]
                )
            print(
                f"  ↺ Sin cambios: ningún candidato aceptado "
                f"(racha={no_improve_streak}/{args.patience})"
            )

        # 8. Persistir SIEMPRE el pipeline activo (nunca dejar candidato rechazado en disco)
        save_text(prompt_file, active_pipeline["main"])
        for rkey in sorted((k for k in active_pipeline if k.startswith("review_")), key=lambda x: int(x.split("_")[1])):
            ridx = int(rkey.split("_")[1])
            rfile = prompt_file.parent / (prompt_file.stem + f"_review{ridx}" + prompt_file.suffix)
            save_text(rfile, active_pipeline[rkey])

        log_iteration(log_file, iteration, active_summary["avg_metrics"], changes_to_log, active_pipeline)

        _active_rev_keys = sorted((k for k in active_pipeline if k.startswith("review_")), key=lambda x: int(x.split("_")[1]))
        _rev_sizes = "".join(f", {k}={len(active_pipeline[k])} chars" for k in _active_rev_keys)
        _state = "ACEPTADO" if accepted_pipeline is not None else "SIN CAMBIOS"
        print(f"  ✓ Estado iteración: {_state} — main={len(active_pipeline['main'])} chars{_rev_sizes}")

        iter_elapsed = time.time() - iter_start
        worker_total_elapsed = parent_worker_elapsed + candidate_worker_elapsed_total
        print(f"  ✓ Log guardado en {log_file.name}")
        print(
            f"  ⏱ Tiempos: baseline-worker={parent_worker_elapsed:.1f}s  "
            f"candidate-worker={candidate_worker_elapsed_total:.1f}s  "
            f"worker-total={worker_total_elapsed:.1f}s  "
            f"optimizer={optimizer_elapsed_total:.1f}s  total={iter_elapsed:.1f}s"
        )
        if accepted_changes and "(sin bloque" not in accepted_changes:
            print(f"  Cambios aceptados: {accepted_changes[:300]}{'...' if len(accepted_changes) > 300 else ''}")

        # Early stopping
        if no_improve_streak >= args.patience:
            print(f"\n  ⏹ Early stopping: {no_improve_streak} ciclos consecutivos sin mejorar {args.metric}.")
            print(f"  Restaurando mejor pipeline (score={best_score:.4f}) desde {best_prompt_file.name}")
            save_text(prompt_file, best_pipeline["main"])
            break

    # Resumen final
    # Garantizar que prompt_file termina con el mejor pipeline visto
    save_text(prompt_file, best_pipeline["main"])
    best_num_reviews = sum(1 for k in best_pipeline if k.startswith("review_"))
    for i in range(1, best_num_reviews + 1):
        rkey = f"review_{i}"
        rfile = prompt_file.parent / (prompt_file.stem + f"_review{i}" + prompt_file.suffix)
        save_text(rfile, best_pipeline[rkey])

    print("\n╔═════════════════════════════════════════════════╗")
    print("║  Entrenamiento completado                        ║")
    print("╚═════════════════════════════════════════════════╝")
    print(f"  Mejor {args.metric}: {best_score:.4f}")
    print(f"  Reviews en mejor pipeline: {best_num_reviews}")
    print(f"  Mejor prompt principal: {best_prompt_file}")
    preview = best_pipeline["main"][:300] + "..." if len(best_pipeline["main"]) > 300 else best_pipeline["main"]
    print(f"\nPrompt principal ({len(best_pipeline['main'])} chars):\n{preview}")


if __name__ == "__main__":
    main()
