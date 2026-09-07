"""Fast smoke tests that do not rerun the full Monte Carlo campaigns."""

from __future__ import annotations

import importlib
import unittest

import numpy as np

from detector import empirical_threshold, false_alarm_rate
from graph_model import build_graph, graph_energy_batch, laplacian_matrix


MODULES = (
    "config",
    "detector",
    "graph_model",
    "experiments.run_experiment",
    "experiments.experiment_topology",
    "analysis.analyze_gesnr",
    "experiments.experiment4_validation",
    "experiments.experiment5_local_gesnr",
    "experiments.experiment6_strict_validation",
    "experiments.experiment6B_numerical_verification",
    "experiments.experiment7_spectral_detection",
    "experiments.experiment8_general_covariance",
    "experiments.experiment9_signal_geometry",
    "experiments.experiment10_final_benchmark",
)


class RepositorySmokeTests(unittest.TestCase):
    def test_all_modules_import(self) -> None:
        for module in MODULES:
            with self.subTest(module=module):
                importlib.import_module(module)

    def test_core_laplacian_detector_pipeline(self) -> None:
        graph = build_graph(12, "path")
        laplacian = laplacian_matrix(graph)
        rng = np.random.default_rng(42)
        samples = rng.normal(size=(200, 12))
        energies = graph_energy_batch(samples, laplacian)
        threshold = empirical_threshold(energies, 0.05)
        pfa = false_alarm_rate(energies, threshold)
        self.assertEqual(energies.shape, (200,))
        self.assertTrue(np.isfinite(threshold))
        self.assertGreaterEqual(pfa, 0.0)
        self.assertLessEqual(pfa, 1.0)


if __name__ == "__main__":
    unittest.main()
