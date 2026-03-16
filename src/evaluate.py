#!/usr/bin/env python3
"""
Evaluación de un prompt sobre un corpus BRAT completo.

Usa un prompt fijo y un modelo (API u Ollama) para procesar todos (o N)
documentos de una carpeta BRAT y reportar métricas detalladas.

Ejemplos:
  # Evaluar con todos los documentos de dev/brat
  python evaluate.py -p prompt_best_1.txt -d ../SPACCC_MEDDOCAN/corpus/dev/brat --model llama3:8b

  # Solo 20 documentos, modelo cloud
  python evaluate.py -p prompt.txt -d ../SPACCC_MEDDOCAN/corpus/dev/brat --model gpt-4o -n 20

  # Modelo Ollama interactivo + no-think + paralelo
  python evaluate.py -p prompt.txt -d ../SPACCC_MEDDOCAN/corpus/dev/brat --model ollama --no-think --parallel 4
"""

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from core import config
from core.config import load_env
from core.ollama_utils import pick_ollama_model
from core.brat import load_text, extract_entity_types, parse_ann, parse_model_output
from core.metrics import compute_metrics, normalize_overlap_threshold
from core.worker import run_worker


def get_all_samples(brat_dir):
    """Devuelve TODOS los pares .txt/.ann de un directorio BRAT, ordenados por nombre."""
    txt_files = sorted(glob.glob(str(Path(brat_dir) / "*.txt")))
    samples = []
    for txt_path in txt_files:
        ann_path = txt_path[:-4] + ".ann"
        if os.path.exists(ann_path):
            ann_raw = load_text(ann_path)
            samples.append({
                "name": os.path.basename(txt_path),
                "text": load_text(txt_path),
                "ann_raw": ann_raw,
                "entities": parse_ann(ann_raw),
            })
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Evaluación de un prompt sobre un corpus BRAT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Ejemplos:
  python evaluate.py -p prompt_best_1.txt -d ../SPACCC_MEDDOCAN/corpus/dev/brat --model llama3:8b
  python evaluate.py -p prompt.txt -d ../SPACCC_MEDDOCAN/corpus/dev/brat --model gpt-4o -n 20
  python evaluate.py -p prompt.txt -d ../SPACCC_MEDDOCAN/corpus/dev/brat --model ollama --no-think
"""
    )

    parser.add_argument(
        "--prompt", "-p", type=str, required=True,
        help="Archivo de texto con el prompt a evaluar",
    )
    parser.add_argument(
        "--brat-dir", "-d", type=str, required=True,
        help="Carpeta con los pares .txt / .ann (formato BRAT) para evaluación",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help=(
            "Modelo a usar. Prefijos: gpt-/o1-/o3-/o4-→OpenAI; deepseek-*→DeepSeek; "
            "'ollama'→selección interactiva; cualquier otro nombre→Ollama directo."
        ),
    )
    parser.add_argument(
        "-n", "--num-docs", type=int, default=None,
        help="Número de documentos a evaluar. Si no se indica, se evalúan todos.",
    )
    parser.add_argument(
        "--parallel", type=int, default=1, metavar="N",
        help="Llamadas en paralelo (default: 1, secuencial. Usar >1 solo con modelos cloud)",
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
        "--no-think", action="store_true", default=False,
        help="Deshabilita el modo 'think' en modelos Ollama que lo soporten",
    )
    parser.add_argument(
        "--ollama-num-ctx", type=int, default=None,
        help=f"Tamaño del contexto para Ollama (num_ctx). Por defecto {16384}.",
    )
    parser.add_argument(
        "--review-prompts", type=str, nargs="*", default=None, metavar="ARCHIVO",
        help="Archivos de texto con prompts de review encadenados (opcional)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Archivo donde guardar el informe detallado (default: imprime en consola)",
    )

    args = parser.parse_args()

    try:
        overlap_threshold = normalize_overlap_threshold(args.min_overlap)
    except ValueError as e:
        sys.exit(f"Error: {e}")

    # ── Configuración ────────────────────────────────────────────────────
    config.API_KEYS.update(load_env())
    config.NO_THINK = args.no_think
    if args.ollama_num_ctx is not None:
        config.OLLAMA_NUM_CTX = args.ollama_num_ctx

    # ── Selección interactiva de Ollama ──────────────────────────────────
    model = args.model
    if model.lower() == "ollama":
        model = pick_ollama_model()
        print(f"  Modelo Ollama seleccionado: {model}")

    # ── Resolver rutas ───────────────────────────────────────────────────
    brat_dir = Path(args.brat_dir).resolve()
    prompt_file = Path(args.prompt).resolve()

    if not brat_dir.is_dir():
        sys.exit(f"Error: la carpeta BRAT no existe: {brat_dir}")
    if not prompt_file.is_file():
        sys.exit(f"Error: el archivo de prompt no existe: {prompt_file}")

    prompt_text = load_text(prompt_file)
    entity_types = extract_entity_types(brat_dir)
    if not entity_types:
        sys.exit("Error: no se encontraron entidades en los .ann de la carpeta BRAT")

    # ── Cargar review prompts opcionales ─────────────────────────────────
    review_list = None
    if args.review_prompts:
        review_list = []
        for rp_path in args.review_prompts:
            rp = Path(rp_path).resolve()
            if not rp.is_file():
                sys.exit(f"Error: archivo de review no encontrado: {rp}")
            review_list.append(load_text(rp))

    # ── Cargar muestras ──────────────────────────────────────────────────
    all_samples = get_all_samples(brat_dir)
    if not all_samples:
        sys.exit(f"Error: no se encontraron pares .txt/.ann en {brat_dir}")

    if args.num_docs is not None:
        samples = all_samples[:args.num_docs]
    else:
        samples = all_samples

    # ── Banner ───────────────────────────────────────────────────────────
    print("╔═════════════════════════════════════════════════╗")
    print("║        Evaluación de prompt sobre corpus        ║")
    print("╚═════════════════════════════════════════════════╝")
    print(f"  Prompt        : {prompt_file}")
    print(f"  BRAT dir      : {brat_dir}")
    print(f"  Modelo        : {model}")
    print(f"  Documentos    : {len(samples)} de {len(all_samples)} disponibles")
    print(f"  Entidades     : {len(entity_types)} tipos detectados — {sorted(entity_types)}")
    print(f"  Match mode    : {args.match_mode}")
    if args.match_mode == "relaxed":
        print(f"  Min overlap   : {overlap_threshold:.0%} ({overlap_threshold:.2f})")
    print(f"  Paralelismo   : {args.parallel} workers")
    reviews_label = f"{len(review_list)} reviews" if review_list else "sin reviews"
    print(f"  Reviews       : {reviews_label}")
    if config.NO_THINK:
        print("  No-think      : activado")
    print()

    # ── Ejecución ────────────────────────────────────────────────────────
    eval_start = time.time()

    def process_sample(idx, sample):
        t0 = time.time()
        output = run_worker(
            prompt_text, sample["text"], entity_types, model,
            review_prompts=review_list,
        )
        predicted = parse_model_output(output, entity_types)
        metrics = compute_metrics(
            predicted,
            sample["entities"],
            match_mode=args.match_mode,
            min_overlap=overlap_threshold,
        )
        elapsed = time.time() - t0
        return idx, {
            **sample,
            "model_output": output,
            "predicted": predicted,
            "metrics": metrics,
            "time": elapsed,
        }

    results = [None] * len(samples)
    errors = []

    if args.parallel <= 1:
        for i, sample in enumerate(samples):
            print(f"  [{i+1}/{len(samples)}] {sample['name']} ...", flush=True)
            try:
                _, result = process_sample(i, sample)
                results[i] = result
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
                errors.append((sample["name"], str(e)))
    else:
        completed = 0
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
                    results[idx] = result
                    m = result["metrics"]
                    print(
                        f"  [{completed}/{len(samples)}] {name}  "
                        f"F1={m['f1']:.3f}  P={m['precision']:.3f}  "
                        f"R={m['recall']:.3f}  "
                        f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})  "
                        f"[{result['time']:.1f}s]",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  [{completed}/{len(samples)}] {name}  ERROR: {e}", flush=True)
                    errors.append((name, str(e)))

    eval_elapsed = time.time() - eval_start

    # Filtrar resultados válidos
    valid_results = [r for r in results if r is not None]

    if not valid_results:
        print("\n  ⚠ No se obtuvieron resultados válidos.")
        sys.exit(1)

    # ── Métricas globales ────────────────────────────────────────────────
    total_tp = sum(r["metrics"]["tp"] for r in valid_results)
    total_fp = sum(r["metrics"]["fp"] for r in valid_results)
    total_fn = sum(r["metrics"]["fn"] for r in valid_results)

    # Micro-average (sobre entidades individuales)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0 else 0.0
    )

    # Macro-average (promedio de métricas por documento)
    macro_precision = sum(r["metrics"]["precision"] for r in valid_results) / len(valid_results)
    macro_recall = sum(r["metrics"]["recall"] for r in valid_results) / len(valid_results)
    macro_f1 = sum(r["metrics"]["f1"] for r in valid_results) / len(valid_results)

    # ── Métricas por tipo de entidad ─────────────────────────────────────
    type_stats = {}
    for etype in sorted(entity_types):
        pred_of_type = []
        gt_of_type = []
        for r in valid_results:
            gt_of_type.extend((t, txt) for t, txt in r["entities"] if t == etype)
            pred_of_type.extend((t, txt) for t, txt in r["predicted"] if t == etype)

        type_stats[etype] = compute_metrics(
            pred_of_type,
            gt_of_type,
            match_mode=args.match_mode,
            min_overlap=overlap_threshold,
        )

    # ── Informe ──────────────────────────────────────────────────────────
    report_lines = []
    report_lines.append("")
    report_lines.append("=" * 65)
    report_lines.append("  RESULTADOS DE EVALUACIÓN")
    report_lines.append("=" * 65)
    report_lines.append(f"  Modelo        : {model}")
    report_lines.append(f"  Prompt        : {prompt_file.name}")
    report_lines.append(f"  Documentos    : {len(valid_results)} evaluados / {len(errors)} errores")
    report_lines.append(f"  Match mode    : {args.match_mode}")
    if args.match_mode == "relaxed":
        report_lines.append(f"  Min overlap   : {overlap_threshold:.0%} ({overlap_threshold:.2f})")
    report_lines.append(f"  Tiempo total  : {eval_elapsed:.1f}s")
    report_lines.append("")
    report_lines.append("  ── Micro-average (entidades globales) ──")
    report_lines.append(f"    Precision : {micro_precision:.4f}")
    report_lines.append(f"    Recall    : {micro_recall:.4f}")
    report_lines.append(f"    F1        : {micro_f1:.4f}")
    report_lines.append(f"    TP={total_tp}  FP={total_fp}  FN={total_fn}")
    report_lines.append("")
    report_lines.append("  ── Macro-average (promedio por documento) ──")
    report_lines.append(f"    Precision : {macro_precision:.4f}")
    report_lines.append(f"    Recall    : {macro_recall:.4f}")
    report_lines.append(f"    F1        : {macro_f1:.4f}")
    report_lines.append("")
    report_lines.append("  ── Métricas por tipo de entidad ──")
    report_lines.append(f"    {'Tipo':<25} {'Prec':>7} {'Rec':>7} {'F1':>7}  {'TP':>5} {'FP':>5} {'FN':>5}")
    report_lines.append(f"    {'-'*25} {'-'*7} {'-'*7} {'-'*7}  {'-'*5} {'-'*5} {'-'*5}")
    for etype in sorted(type_stats.keys()):
        s = type_stats[etype]
        report_lines.append(
            f"    {etype:<25} {s['precision']:>7.4f} {s['recall']:>7.4f} {s['f1']:>7.4f}"
            f"  {s['tp']:>5} {s['fp']:>5} {s['fn']:>5}"
        )
    report_lines.append("")

    # Top 5 peores documentos
    sorted_by_f1 = sorted(valid_results, key=lambda r: r["metrics"]["f1"])
    report_lines.append("  ── 5 peores documentos (por F1) ──")
    for r in sorted_by_f1[:5]:
        m = r["metrics"]
        report_lines.append(
            f"    {r['name']:<45} F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
            f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})"
        )
    report_lines.append("")

    # Top 5 mejores documentos
    sorted_by_f1_desc = sorted(valid_results, key=lambda r: r["metrics"]["f1"], reverse=True)
    report_lines.append("  ── 5 mejores documentos (por F1) ──")
    for r in sorted_by_f1_desc[:5]:
        m = r["metrics"]
        report_lines.append(
            f"    {r['name']:<45} F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
            f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})"
        )
    report_lines.append("")

    if errors:
        report_lines.append(f"  ── Errores ({len(errors)}) ──")
        for name, err in errors:
            report_lines.append(f"    {name}: {err}")
        report_lines.append("")

    report_lines.append("=" * 65)

    report = "\n".join(report_lines)
    print(report)

    # ── Guardar informe si se pide ───────────────────────────────────────
    if args.output:
        output_path = Path(args.output).resolve()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n  Informe guardado en: {output_path}")


if __name__ == "__main__":
    main()
