"""Escritura del log de iteraciones en disco."""
from datetime import datetime


def log_iteration(log_path, iteration, avg_metrics, changes, prompts_dict):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prompts_block = ""
    for key in ["main"] + [f"review_{i}" for i in range(1, 4)]:
        if key in prompts_dict:
            prompts_block += f"[{key}]\n{prompts_dict[key]}\n\n"
    entry = (
        f"\n{'='*60}\n"
        f"Iteración {iteration} — {timestamp}\n"
        f"Métricas promedio: P={avg_metrics['precision']:.4f} "
        f"R={avg_metrics['recall']:.4f} F1={avg_metrics['f1']:.4f}\n"
        f"Reviews activas: {sum(1 for k in prompts_dict if k.startswith('review_'))}\n"
        f"Cambios realizados:\n{changes}\n"
        f"Pipeline de prompts:\n{prompts_block}"
        f"{'='*60}\n"
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)
