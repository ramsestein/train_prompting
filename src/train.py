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
from core.metrics import compute_metrics
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
    print(f"  Entidades     : {len(entity_types)} tipos detectados")
    print(f"  Parallelismo  : {args.parallel} workers")
    print(f"  Patience      : {args.patience} ciclos sin mejora")
    print(f"  Review step   : cada {args.review_step} ciclos sin mejora (máx {args.max_reviews} reviews)")
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

    # Pipeline de prompts: 'main' siempre presente, 'review_N' se añaden dinámicamente
    pipeline = {"main": load_text(prompt_file)}
    best_pipeline = dict(pipeline)
    num_reviews = 0
    review_stagnation = {}   # review_idx -> ciclos sin mejora desde que se añadió
    review_alt_counters = {} # review_idx -> índice rotatorio de alternativas

    for iteration in range(1, args.iterations + 1):
        iter_start = time.time()
        print(f"\n┌── Iteración {iteration}/{args.iterations} ──┐")

        # Sincronizar pipeline['main'] con el archivo (por si se editó manualmente)
        pipeline["main"] = load_text(prompt_file)
        current_prompt = pipeline["main"]
        # Evolutivo: num_reviews refleja el best; cand_num_reviews refleja el candidato actual
        num_reviews = sum(1 for k in best_pipeline if k.startswith("review_"))
        cand_num_reviews = sum(1 for k in pipeline if k.startswith("review_"))
        review_list = [pipeline[f"review_{i}"] for i in range(1, cand_num_reviews + 1)]
        cand_extra = f" [best={num_reviews}]" if cand_num_reviews != num_reviews else ""
        review_label = f" + {cand_num_reviews} review(s){cand_extra}" if cand_num_reviews > 0 else ""
        print(f"  Prompt actual: {len(current_prompt)} chars{review_label}")

        # 1. Seleccionar muestras
        samples = get_training_samples(brat_dir, args.samples)
        print(f"  Muestras seleccionadas: {len(samples)}")

        # 2. Ejecutar modelo worker
        worker_start = time.time()
        mode = "paralelo" if args.parallel > 1 else "secuencial"
        print(f"  Ejecutando {args.worker_model} ({mode}) sobre {len(samples)} muestras...")

        def process_sample(idx, sample):
            t0 = time.time()
            output = run_worker(
                current_prompt, sample["text"], entity_types, args.worker_model,
                review_prompts=review_list,
            )
            predicted = parse_model_output(output, entity_types)
            metrics = compute_metrics(predicted, sample["entities"])
            elapsed = time.time() - t0
            return idx, {**sample, "model_output": output, "predicted": predicted, "metrics": metrics, "time": elapsed}

        def print_result(completed, total, name, result):
            m = result["metrics"]
            print(
                f"  [{completed}/{total}] {name}  "
                f"F1={m['f1']:.3f}  P={m['precision']:.3f}  "
                f"R={m['recall']:.3f}  "
                f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})  "
                f"[{result['time']:.1f}s]",
                flush=True,
            )

        results = []
        if args.parallel <= 1:
            # Secuencial: ves cada muestra conforme termina
            for i, sample in enumerate(samples):
                print(f"  [{i+1}/{len(samples)}] {sample['name']} ...", flush=True)
                try:
                    _, result = process_sample(i, sample)
                    results.append(result)
                    m = result["metrics"]
                    print(
                        f"         F1={m['f1']:.3f}  P={m['precision']:.3f}  "
                        f"R={m['recall']:.3f}  "
                        f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})  "
                        f"[{result['time']:.1f}s]",
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
                        print(f"  [{completed}/{len(samples)}] {name}  ERROR: {e}", flush=True)
            results = [r for r in slot_results if r is not None]

        worker_elapsed = time.time() - worker_start
        print(f"  Worker batch completado en {worker_elapsed:.1f}s")

        if not results:
            print("  ⚠ Sin resultados, saltando iteración.")
            continue

        # 3. Métricas promedio
        avg_metrics = {
            "precision": sum(r["metrics"]["precision"] for r in results) / len(results),
            "recall": sum(r["metrics"]["recall"] for r in results) / len(results),
            "f1": sum(r["metrics"]["f1"] for r in results) / len(results),
        }
        total_tp = sum(r["metrics"]["tp"] for r in results)
        total_fp = sum(r["metrics"]["fp"] for r in results)
        total_fn = sum(r["metrics"]["fn"] for r in results)

        # Desglose de errores por tipo de entidad
        fn_by_type = {}
        fp_by_type = {}
        for r in results:
            gt_remaining = list(r["entities"])
            pred_remaining = list(r["predicted"])
            for p in list(pred_remaining):
                if p in gt_remaining:
                    gt_remaining.remove(p)
                    pred_remaining.remove(p)
            for etype, _ in gt_remaining:
                fn_by_type[etype] = fn_by_type.get(etype, 0) + 1
            for etype, _ in pred_remaining:
                fp_by_type[etype] = fp_by_type.get(etype, 0) + 1

        current_iter_score = avg_metrics[args.metric]
        improvement = ""
        if current_iter_score > best_score:
            improvement = f"  ↑ MEJORA ({current_iter_score - best_score:+.4f})"
            best_score = current_iter_score
            best_pipeline = dict(pipeline)
            save_text(best_prompt_file, best_pipeline["main"])
            no_improve_streak = 0
            # Inicializar stagnation/counter para reviews recién promovidas al best
            new_best_reviews = sum(1 for k in best_pipeline if k.startswith("review_"))
            for i in range(1, new_best_reviews + 1):
                review_stagnation[i] = 0
                review_alt_counters.setdefault(i, 0)
        else:
            no_improve_streak += 1
            for k in review_stagnation:
                review_stagnation[k] += 1
            if iteration > 1:
                improvement = f"  ↓ sin mejora (best={best_score:.4f}, racha={no_improve_streak}/{args.patience})"

        print(f"\n  ► Promedio: P={avg_metrics['precision']:.4f}  "
              f"R={avg_metrics['recall']:.4f}  F1={avg_metrics['f1']:.4f}"
              f"{improvement}")
        print(f"    Totales: TP={total_tp}  FP={total_fp}  FN={total_fn}")
        if fn_by_type:
            top_fn = sorted(fn_by_type.items(), key=lambda x: -x[1])[:5]
            print(f"    Top FN (no detectados): {', '.join(f'{t}={c}' for t, c in top_fn)}")
        if fp_by_type:
            top_fp = sorted(fp_by_type.items(), key=lambda x: -x[1])[:5]
            print(f"    Top FP (falsos positivos): {', '.join(f'{t}={c}' for t, c in top_fp)}")

        # 4. Optimizar pipeline de prompts
        opt_start = time.time()

        # Decidir si inyectar una estrategia de optimización
        current_delta = current_iter_score - prev_iter_score
        low_growth = (
            iteration > 1
            and 0 < current_delta < last_improvement_delta * 0.5
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
        prev_iter_score = current_iter_score

        # Construir hints de alternativas para reviews estancadas
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
            _use_recall = avg_metrics["recall"] <= avg_metrics["precision"]
            _adv_name = "recall" if _use_recall else "precision"
            print(f"  📊 F1 mode → métrica más baja: {_adv_name} "
                  f"(R={avg_metrics['recall']:.4f} P={avg_metrics['precision']:.4f})")
            if _use_recall:
                print(f"  🔍 Advisor de recall analizando FN ...")
                try:
                    adv = recall_advisor(best_pipeline, results, args.optimizer_model)
                    if adv.strip():
                        advisor_reports["recall"] = adv
                        print(f"     Informe de recall: {len(adv)} chars")
                except Exception as e:
                    print(f"     ⚠ Error en advisor recall: {e}")
            else:
                print(f"  🛡 Advisor de precisión analizando FP ...")
                try:
                    adv = precision_advisor(best_pipeline, results, args.optimizer_model)
                    if adv.strip():
                        advisor_reports["precision"] = adv
                        print(f"     Informe de precisión: {len(adv)} chars")
                except Exception as e:
                    print(f"     ⚠ Error en advisor precisión: {e}")
        elif args.metric == "recall":
            print(f"  🔍 Advisor de recall analizando FN ...")
            try:
                adv = recall_advisor(best_pipeline, results, args.optimizer_model)
                if adv.strip():
                    advisor_reports["recall"] = adv
                    print(f"     Informe de recall: {len(adv)} chars")
            except Exception as e:
                print(f"     ⚠ Error en advisor recall: {e}")
        elif args.metric == "precision":
            print(f"  🛡 Advisor de precisión analizando FP ...")
            try:
                adv = precision_advisor(best_pipeline, results, args.optimizer_model)
                if adv.strip():
                    advisor_reports["precision"] = adv
                    print(f"     Informe de precisión: {len(adv)} chars")
            except Exception as e:
                print(f"     ⚠ Error en advisor precisión: {e}")

        try:
            # Evolutivo: siempre mutar desde el best, no desde el candidato actual
            updated_pipeline, changes = optimize_pipeline(
                best_pipeline,
                results,
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
            )
            opt_elapsed = time.time() - opt_start

            new_main = updated_pipeline.get("main", "").strip()
            iter_elapsed = time.time() - iter_start

            if len(new_main) < 10:
                print(f"  ⚠ Prompt principal vacío tras optimización, conservando el anterior.")
                log_iteration(log_file, iteration, avg_metrics, f"DESCARTADO: {changes}", pipeline)
            else:
                # Detectar si el optimizador añadió una nueva review
                new_review_key = f"review_{num_reviews + 1}"
                if new_review_key in updated_pipeline and num_reviews < args.max_reviews:
                    # La review está en el candidato; se confirma en best solo si mejora la métrica
                    _cand_review_idx = num_reviews + 1
                    review_file = prompt_file.parent / (prompt_file.stem + f"_review{_cand_review_idx}" + prompt_file.suffix)
                    save_text(review_file, updated_pipeline[new_review_key])
                    print(f"  ✚ Candidato propone review_{_cand_review_idx} ({len(updated_pipeline[new_review_key])} chars) → {review_file.name} [se confirma si mejora]")

                pipeline = updated_pipeline

                # Múltiplo de 3 fallos: refinar un párrafo aleatorio del prompt principal
                if no_improve_streak > 0 and no_improve_streak % 3 == 0:
                    print(f"  🔍 Múltiplo de 3 ciclos sin mejora — refinando párrafo aleatorio...")
                    refined_main = refine_paragraph(pipeline["main"], args.optimizer_model)
                    if refined_main != pipeline["main"]:
                        pipeline["main"] = refined_main
                        print(f"  ✓ Párrafo refinado ({len(refined_main)} chars total)")

                save_text(prompt_file, pipeline["main"])
                for rkey in sorted(k for k in pipeline if k.startswith("review_")):
                    ridx = int(rkey.split("_")[1])
                    rfile = prompt_file.parent / (prompt_file.stem + f"_review{ridx}" + prompt_file.suffix)
                    save_text(rfile, pipeline[rkey])

                log_iteration(log_file, iteration, avg_metrics, changes, pipeline)
                _cand_rev_keys = sorted(k for k in pipeline if k.startswith("review_"))
                _rev_sizes = "".join(f", {k}={len(pipeline[k])} chars" for k in _cand_rev_keys)
                print(f"  ✓ Candidato: main={len(pipeline['main'])} chars{_rev_sizes}")

                # Avanzar contadores de alternativas para las reviews que recibieron hint
                for i in review_alt_hints:
                    review_alt_counters[i] = review_alt_counters.get(i, 0) + 1

            print(f"  ✓ Log guardado en {log_file.name}")
            print(f"  ⏱ Tiempos: worker={worker_elapsed:.1f}s  optimizer={opt_elapsed:.1f}s  total={iter_elapsed:.1f}s")
            if changes and "(sin bloque" not in changes:
                print(f"  Cambios: {changes[:300]}{'...' if len(changes) > 300 else ''}")

        except Exception as e:
            print(f"  ✗ Error al optimizar: {e}")
            log_iteration(log_file, iteration, avg_metrics, f"Error: {e}", pipeline)

        # Early stopping
        if no_improve_streak >= args.patience:
            print(f"\n  ⏹ Early stopping: {no_improve_streak} ciclos consecutivos sin mejorar {args.metric}.")
            print(f"  Restaurando mejor pipeline (score={best_score:.4f}) desde {best_prompt_file.name}")
            pipeline = best_pipeline
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
