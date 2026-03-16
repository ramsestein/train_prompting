"""Cálculo de métricas de evaluación (precision, recall, F1)."""

VALID_MATCH_MODES = {"strict", "relaxed"}


def normalize_match_mode(match_mode):
    """Valida y normaliza el modo de matching."""
    mode = (match_mode or "strict").strip().lower()
    if mode not in VALID_MATCH_MODES:
        modes = ", ".join(sorted(VALID_MATCH_MODES))
        raise ValueError(f"match_mode inválido: '{match_mode}'. Debe ser uno de: {modes}")
    return mode


def normalize_overlap_threshold(min_overlap):
    """
    Normaliza el umbral de solapamiento.

    Acepta valor fraccional (0.5) o porcentaje (50).
    Devuelve siempre un float en rango [0, 1].
    """
    try:
        threshold = float(min_overlap)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"min_overlap inválido: '{min_overlap}'") from exc

    if threshold < 0:
        raise ValueError("min_overlap no puede ser negativo")

    if threshold > 1:
        threshold = threshold / 100.0

    if threshold > 1:
        raise ValueError("min_overlap fuera de rango. Usa 0..1 o 0..100")

    return threshold


def overlap_ratio_vs_ground_truth(pred_text, gt_text):
    """
    Calcula solapamiento (0..1) respecto al largo de la entidad real (ground truth).

    Se considera solo relación de inclusión textual:
    - pred ⊆ gt  -> ratio = len(pred) / len(gt)
    - gt ⊆ pred  -> ratio = 1.0
    - sin inclusión -> 0.0
    """
    pred = pred_text.strip()
    gt = gt_text.strip()

    if not pred or not gt:
        return 0.0
    if pred == gt:
        return 1.0
    if pred in gt:
        return len(pred) / len(gt)
    if gt in pred:
        return 1.0
    return 0.0


def _entities_match(pred_entity, gt_entity, mode, threshold):
    pred_type, pred_text = pred_entity
    gt_type, gt_text = gt_entity

    if pred_type != gt_type:
        return False

    if mode == "strict":
        return pred_text == gt_text

    overlap_ratio = overlap_ratio_vs_ground_truth(pred_text, gt_text)
    return overlap_ratio >= threshold


def compute_match_details(predicted, ground_truth, match_mode="strict", min_overlap=0.5):
    """
    Devuelve el detalle de matching entre predicciones y ground truth.

    Aplica matching bipartito máximo 1:1 para manejar duplicados correctamente.
    """
    mode = normalize_match_mode(match_mode)
    threshold = normalize_overlap_threshold(min_overlap)

    pred_list = list(predicted)
    gt_list = list(ground_truth)

    edges = []
    for gt_entity in gt_list:
        candidates = []
        for pred_idx, pred_entity in enumerate(pred_list):
            if _entities_match(pred_entity, gt_entity, mode, threshold):
                candidates.append(pred_idx)
        edges.append(candidates)

    match_pred_to_gt = {}

    def _dfs(gt_idx, visited):
        for pred_idx in edges[gt_idx]:
            if pred_idx in visited:
                continue
            visited.add(pred_idx)

            current_gt = match_pred_to_gt.get(pred_idx)
            if current_gt is None or _dfs(current_gt, visited):
                match_pred_to_gt[pred_idx] = gt_idx
                return True
        return False

    # Procesar primero GT con menos candidatos favorece estabilidad del matching.
    for gt_idx in sorted(range(len(gt_list)), key=lambda i: len(edges[i])):
        _dfs(gt_idx, set())

    matched_pairs = sorted(
        ((gt_idx, pred_idx) for pred_idx, gt_idx in match_pred_to_gt.items()),
        key=lambda pair: (pair[0], pair[1]),
    )

    matched_gt_idxs = {gt_idx for gt_idx, _ in matched_pairs}
    matched_pred_idxs = {pred_idx for _, pred_idx in matched_pairs}

    unmatched_gt_idxs = [i for i in range(len(gt_list)) if i not in matched_gt_idxs]
    unmatched_pred_idxs = [i for i in range(len(pred_list)) if i not in matched_pred_idxs]

    return {
        "mode": mode,
        "min_overlap": threshold,
        "tp": len(matched_pairs),
        "fp": len(unmatched_pred_idxs),
        "fn": len(unmatched_gt_idxs),
        "matched_pairs": matched_pairs,
        "unmatched_gt": [gt_list[i] for i in unmatched_gt_idxs],
        "unmatched_pred": [pred_list[i] for i in unmatched_pred_idxs],
    }


def compute_metrics(predicted, ground_truth, match_mode="strict", min_overlap=0.5):
    """
    Calcula precision, recall, F1 comparando listas de (tipo, texto).

    Modos:
      - strict: acierto solo si (tipo, texto) coincide exactamente.
      - relaxed: acierto si hay inclusión textual y el solapamiento respecto
                 a la entidad real supera min_overlap.
    """
    details = compute_match_details(
        predicted,
        ground_truth,
        match_mode=match_mode,
        min_overlap=min_overlap,
    )

    tp = details["tp"]
    fp = details["fp"]
    fn = details["fn"]

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
