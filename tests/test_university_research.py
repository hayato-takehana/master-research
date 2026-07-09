import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from active.tf_delete_v1_predata.university_research import university_research


def make_numeric_context():
    return SimpleNamespace(
        X_train=np.array([[-2.0], [-1.0], [1.0], [2.0]]),
        train_labels=np.array([0, 0, 1, 1]),
        X_test=np.array([[-1.5], [1.5]]),
        test_labels=np.array([0, 1]),
        feature_count=1,
    )


def make_outer_fold_contexts():
    return [
        {
            "outer_fold": fold,
            "outer_context": make_numeric_context(),
            "inner_contexts": [
                make_numeric_context(),
                make_numeric_context(),
            ],
        }
        for fold in (1, 2)
    ]


class UniversityResearchTests(unittest.TestCase):
    def test_hyperparameter_ranges_match_term_difference_experiment(self):
        expected_values = [
            0.00001,
            0.0001,
            0.001,
            0.01,
            0.1,
            1,
            10,
            100,
            1000,
            10000,
            100000,
        ]
        self.assertEqual(university_research.LINEAR_C_VALUES, expected_values)
        self.assertEqual(university_research.RBF_C_VALUES, expected_values)
        self.assertEqual(university_research.RBF_GAMMA_VALUES, expected_values)
        self.assertEqual(university_research.MARGIN_THRESHOLD, 0.9990)

    def test_margin_condition_drops_small_c_and_accepts_large_c(self):
        context = make_numeric_context()

        invalid_result = university_research.fit_and_evaluate_context(
            context,
            kernel="linear",
            c=0.00001,
        )
        valid_result = university_research.fit_and_evaluate_context(
            context,
            kernel="linear",
            c=100,
        )

        self.assertIsNone(invalid_result)
        self.assertIsNotNone(valid_result)
        self.assertEqual(valid_result["accuracy"], 1.0)

    def test_linear_and_rbf_nested_cv_run_separately(self):
        contexts = make_outer_fold_contexts()

        linear_result = university_research.run_margin_constrained_nested_cv(
            contexts,
            kernel="linear",
            c_values=[100],
        )
        rbf_result = university_research.run_margin_constrained_nested_cv(
            contexts,
            kernel="rbf",
            c_values=[100],
            gamma_values=[1],
        )

        self.assertEqual(linear_result["kernel"], "linear")
        self.assertEqual(linear_result["best_cs"], [100, 100])
        self.assertNotIn("best_gammas", linear_result)
        self.assertEqual(rbf_result["kernel"], "rbf")
        self.assertEqual(rbf_result["best_cs"], [100, 100])
        self.assertEqual(rbf_result["best_gammas"], [1, 1])
        self.assertEqual(linear_result["accuracy"], 1.0)
        self.assertEqual(rbf_result["accuracy"], 1.0)

    def test_linear_and_rbf_outputs_are_saved_separately(self):
        contexts = make_outer_fold_contexts()
        linear_result = university_research.run_margin_constrained_nested_cv(
            contexts,
            kernel="linear",
            c_values=[100],
        )
        rbf_result = university_research.run_margin_constrained_nested_cv(
            contexts,
            kernel="rbf",
            c_values=[100],
            gamma_values=[1],
        )

        original_save_dir = university_research.SAVE_DIR
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                university_research.SAVE_DIR = Path(temp_dir)
                linear_paths = university_research.save_result(linear_result)
                rbf_paths = university_research.save_result(rbf_result)

                self.assertEqual(linear_paths["metrics"].parent.name, "linear")
                self.assertEqual(rbf_paths["metrics"].parent.name, "rbf")
                for output_path in (*linear_paths.values(), *rbf_paths.values()):
                    self.assertTrue(output_path.exists())
        finally:
            university_research.SAVE_DIR = original_save_dir


if __name__ == "__main__":
    unittest.main()
