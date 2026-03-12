"""Cálculo de métricas de evaluación (precision, recall, F1)."""


def compute_metrics(predicted, ground_truth):
    """
    Calcula precision, recall, F1 comparando listas de (tipo, texto).
    Usa matching exacto por (tipo, texto). Maneja duplicados correctamente.
    """
    pred_remaining = list(predicted)
    tp = 0

    for gt in ground_truth:
        if gt in pred_remaining:
            pred_remaining.remove(gt)
            tp += 1

    fp = len(pred_remaining)
    fn = len(ground_truth) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
