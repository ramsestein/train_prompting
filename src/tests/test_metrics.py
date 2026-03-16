#!/usr/bin/env python3
"""Tests para métricas strict/relaxed y umbral de solapamiento."""

import os
import sys
import unittest

# Añadir src/ al path para importar el paquete core
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.metrics import compute_match_details, compute_metrics, normalize_overlap_threshold  # noqa: E402


class TestMetrics(unittest.TestCase):
    def test_strict_match_exacto(self):
        predicted = [("ENFERMEDAD", "insuficiencia renal")]
        ground_truth = [("ENFERMEDAD", "insuficiencia renal")]

        m = compute_metrics(predicted, ground_truth, match_mode="strict")

        self.assertEqual(m["tp"], 1)
        self.assertEqual(m["fp"], 0)
        self.assertEqual(m["fn"], 0)

    def test_strict_no_acepta_parcial(self):
        predicted = [("ENFERMEDAD", "insuficiencia renal")]
        ground_truth = [("ENFERMEDAD", "insuficiencia renal aguda")]

        m = compute_metrics(predicted, ground_truth, match_mode="strict")

        self.assertEqual(m["tp"], 0)
        self.assertEqual(m["fp"], 1)
        self.assertEqual(m["fn"], 1)

    def test_relaxed_acepta_inclusion_con_umbral(self):
        predicted = [("ENFERMEDAD", "insuficiencia renal")]
        ground_truth = [("ENFERMEDAD", "insuficiencia renal aguda")]

        m = compute_metrics(predicted, ground_truth, match_mode="relaxed", min_overlap=0.5)

        self.assertEqual(m["tp"], 1)
        self.assertEqual(m["fp"], 0)
        self.assertEqual(m["fn"], 0)

    def test_relaxed_rechaza_inclusion_bajo_umbral(self):
        predicted = [("ENFERMEDAD", "renal")]
        ground_truth = [("ENFERMEDAD", "insuficiencia renal aguda")]

        m = compute_metrics(predicted, ground_truth, match_mode="relaxed", min_overlap=0.5)

        self.assertEqual(m["tp"], 0)
        self.assertEqual(m["fp"], 1)
        self.assertEqual(m["fn"], 1)

    def test_relaxed_acepta_si_real_esta_incluida_en_predicha(self):
        predicted = [("ENFERMEDAD", "insuficiencia renal aguda severa")]
        ground_truth = [("ENFERMEDAD", "insuficiencia renal aguda")]

        m = compute_metrics(predicted, ground_truth, match_mode="relaxed", min_overlap=0.9)

        self.assertEqual(m["tp"], 1)
        self.assertEqual(m["fp"], 0)
        self.assertEqual(m["fn"], 0)

    def test_relaxed_exige_mismo_tipo(self):
        predicted = [("OTRO", "insuficiencia renal")]
        ground_truth = [("ENFERMEDAD", "insuficiencia renal")]

        m = compute_metrics(predicted, ground_truth, match_mode="relaxed", min_overlap=0.1)

        self.assertEqual(m["tp"], 0)
        self.assertEqual(m["fp"], 1)
        self.assertEqual(m["fn"], 1)

    def test_matching_es_uno_a_uno_con_duplicados(self):
        predicted = [
            ("ENFERMEDAD", "insuficiencia renal"),
            ("ENFERMEDAD", "insuficiencia renal"),
        ]
        ground_truth = [("ENFERMEDAD", "insuficiencia renal")]

        d = compute_match_details(predicted, ground_truth, match_mode="strict")

        self.assertEqual(d["tp"], 1)
        self.assertEqual(d["fp"], 1)
        self.assertEqual(d["fn"], 0)
        self.assertEqual(len(d["matched_pairs"]), 1)

    def test_normalize_overlap_threshold_acepta_porcentaje(self):
        self.assertAlmostEqual(normalize_overlap_threshold(50), 0.5)
        self.assertAlmostEqual(normalize_overlap_threshold(0.5), 0.5)

    def test_normalize_overlap_threshold_fuera_rango(self):
        with self.assertRaises(ValueError):
            normalize_overlap_threshold(150)


if __name__ == "__main__":
    unittest.main(verbosity=2)
