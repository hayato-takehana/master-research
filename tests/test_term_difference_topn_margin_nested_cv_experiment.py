import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("DIFFERENCES_BETWEEN_LABELS_OUTPUT_ROOT", str(PROJECT_ROOT))

from active.tf_delete_v1_predata.research import (  # noqa: E402
    term_difference_topn_margin_nested_cv_experiment as experiment,
)


class TermFrequencyIqrTests(unittest.TestCase):
    def test_experiment_conditions_include_twelve_feature_conditions(self):
        self.assertEqual(
            experiment.FEATURE_CONDITIONS,
            (
                ("term_frequency", "column_min_max"),
                ("term_frequency", "row_l2"),
                ("log_term_frequency", "column_min_max"),
                ("log_term_frequency", "row_l2"),
                ("tf_s", "full_vocab_row_l2"),
                ("log_tf_s", "full_vocab_row_l2"),
                ("term_frequency", "column_min_max_df_gt_2pct"),
                ("term_frequency", "row_l2_df_gt_2pct"),
                ("log_term_frequency", "column_min_max_df_gt_2pct"),
                ("log_term_frequency", "row_l2_df_gt_2pct"),
                ("tf_s", "full_vocab_row_l2_df_gt_2pct"),
                ("log_tf_s", "full_vocab_row_l2_df_gt_2pct"),
            ),
        )
        self.assertEqual(len(experiment.EXPERIMENT_CONDITIONS), 24)

    def test_document_frequency_filter_drops_two_percent_or_less(self):
        train_counts = np.zeros((100, 3), dtype=float)
        train_counts[:2, 0] = 1.0
        train_counts[:3, 1] = 1.0
        train_counts[:, 2] = 1.0

        (
            filtered_train,
            filtered_test,
            filtered_names,
            filtered_ratios,
        ) = experiment.filter_count_matrices_by_document_frequency(
            csr_matrix(train_counts),
            csr_matrix(np.ones((2, 3), dtype=float)),
            np.array(["two_percent", "three_percent", "all"]),
        )

        self.assertEqual(
            filtered_names.tolist(),
            ["three_percent", "all"],
        )
        np.testing.assert_allclose(filtered_ratios, [0.03, 1.0])
        self.assertEqual(filtered_train.shape, (100, 2))
        self.assertEqual(filtered_test.shape, (2, 2))

    def test_relative_tf_uses_original_document_length_after_filtering(self):
        difference_df = experiment.build_term_frequency_difference_table(
            csr_matrix(np.array([[1.0], [0.0]])),
            np.array([1, 0]),
            np.array(["retained_term"]),
            document_word_counts=np.array([10.0, 10.0]),
        )

        self.assertAlmostEqual(
            difference_df.loc[0, "label1_value"],
            0.1,
        )
        self.assertAlmostEqual(
            difference_df.loc[0, "absolute_difference"],
            0.1,
        )

    def test_selects_only_term_frequency_features(self):
        difference_df = pd.DataFrame(
            {
                "term": ["first", "second"],
                "absolute_difference": [0.8, 0.4],
            }
        )

        selected_df = experiment.select_term_frequency_terms(
            difference_df,
            feature_count=1,
        )

        self.assertEqual(selected_df["term"].tolist(), ["first"])
        self.assertEqual(
            selected_df["selection_mode"].tolist(),
            ["term_frequency"],
        )
        self.assertEqual(
            selected_df["selection_source"].tolist(),
            ["term_frequency"],
        )

    def test_log_normalized_frequency_uses_requested_formula(self):
        counts = csr_matrix(
            np.array(
                [
                    [0.0, 1.0],
                    [1.0, 4.0],
                    [2.0, 0.0],
                ]
            )
        )
        document_word_counts = np.array([10.0, 10.0, 10.0])

        actual = experiment._log_normalized_frequency_matrix(
            counts,
            document_word_counts,
        ).toarray()
        denominator = 1.0 + np.log(10.0)
        expected = np.array(
            [
                [0.0, 1.0 / denominator],
                [1.0 / denominator, (1.0 + np.log(4.0)) / denominator],
                [(1.0 + np.log(2.0)) / denominator, 0.0],
            ]
        )

        np.testing.assert_allclose(actual, expected)

    def test_log_tf_difference_is_absolute_label_mean_difference(self):
        counts = csr_matrix(
            np.array(
                [
                    [1.0],
                    [4.0],
                    [0.0],
                    [2.0],
                ]
            )
        )
        labels = np.array([1, 1, 0, 0])
        document_word_counts = np.full(4, 10.0)

        difference_df = experiment.build_log_term_frequency_difference_table(
            counts,
            labels,
            np.array(["token"]),
            document_word_counts,
        )

        denominator = 1.0 + np.log(10.0)
        expected_label1 = (
            1.0 + (1.0 + np.log(4.0))
        ) / (2.0 * denominator)
        expected_label0 = (
            1.0 + np.log(2.0)
        ) / (2.0 * denominator)
        self.assertAlmostEqual(
            difference_df.loc[0, "label1_value"],
            expected_label1,
        )
        self.assertAlmostEqual(
            difference_df.loc[0, "label0_value"],
            expected_label0,
        )
        self.assertAlmostEqual(
            difference_df.loc[0, "absolute_difference"],
            abs(expected_label1 - expected_label0),
        )

    def test_log_tf_features_are_min_max_scaled_by_train_column(self):
        train_counts = csr_matrix(np.array([[0.0], [1.0], [4.0]]))
        test_counts = csr_matrix(np.array([[2.0]]))
        selected_terms_df = pd.DataFrame(
            [
                {
                    "selection_mode": "log_term_frequency",
                    "selection_rank": 1,
                    "term": "token",
                    "selection_source": "log_term_frequency",
                }
            ]
        )

        feature_matrices = experiment.build_selected_feature_matrices(
            train_counts,
            test_counts,
            np.array(["token"]),
            selected_terms_df,
            np.full(3, 10.0),
            np.array([10.0]),
        )

        np.testing.assert_allclose(
            feature_matrices["column_min_max"]["train"].ravel(),
            [0.0, 1.0 / (1.0 + np.log(4.0)), 1.0],
        )
        np.testing.assert_allclose(
            feature_matrices["column_min_max"]["test"].ravel(),
            [(1.0 + np.log(2.0)) / (1.0 + np.log(4.0))],
        )
        self.assertEqual(
            feature_matrices["feature_metadata"]["column_min_max"][
                "feature_transform"
            ].tolist(),
            ["log_normalized_frequency_column_min_max"],
        )

    def test_relative_tf_row_l2_uses_raw_frequency_numerator(self):
        train_counts = csr_matrix(
            np.array(
                [
                    [3.0, 4.0],
                    [0.0, 0.0],
                    [1.0, 0.0],
                ]
            )
        )
        selected_terms_df = pd.DataFrame(
            [
                {
                    "selection_mode": "term_frequency",
                    "selection_rank": 1,
                    "term": "first",
                    "selection_source": "term_frequency",
                },
                {
                    "selection_mode": "term_frequency",
                    "selection_rank": 2,
                    "term": "second",
                    "selection_source": "term_frequency",
                },
            ]
        )

        feature_matrices = experiment.build_selected_feature_matrices(
            train_counts,
            csr_matrix(np.array([[6.0, 8.0]])),
            np.array(["first", "second"]),
            selected_terms_df,
            np.array([100.0, 200.0, 300.0]),
            np.array([1000.0]),
        )

        np.testing.assert_allclose(
            feature_matrices["row_l2"]["train"],
            np.array([[0.6, 0.8], [0.0, 0.0], [1.0, 0.0]]),
        )
        np.testing.assert_allclose(
            feature_matrices["row_l2"]["test"],
            np.array([[0.6, 0.8]]),
        )
        self.assertEqual(
            feature_matrices["feature_metadata"]["row_l2"][
                "feature_transform"
            ].tolist(),
            ["raw_frequency_row_l2", "raw_frequency_row_l2"],
        )

    def test_log_tf_row_l2_uses_log_frequency_numerator(self):
        selected_terms_df = pd.DataFrame(
            [
                {
                    "selection_mode": "log_term_frequency",
                    "selection_rank": 1,
                    "term": "first",
                    "selection_source": "log_term_frequency",
                },
                {
                    "selection_mode": "log_term_frequency",
                    "selection_rank": 2,
                    "term": "second",
                    "selection_source": "log_term_frequency",
                },
            ]
        )
        counts = csr_matrix(np.array([[1.0, 4.0], [0.0, 2.0]]))

        feature_matrices = experiment.build_selected_feature_matrices(
            counts,
            counts,
            np.array(["first", "second"]),
            selected_terms_df,
            np.array([10.0, 20.0]),
            np.array([100.0, 200.0]),
        )

        numerator = np.array(
            [
                [1.0, 1.0 + np.log(4.0)],
                [0.0, 1.0 + np.log(2.0)],
            ]
        )
        expected = numerator / np.linalg.norm(
            numerator,
            axis=1,
            keepdims=True,
        )
        np.testing.assert_allclose(
            feature_matrices["row_l2"]["train"],
            expected,
        )
        np.testing.assert_allclose(
            feature_matrices["row_l2"]["test"],
            expected,
        )

    def test_tf_s_difference_uses_full_vocabulary_row_l2_values(self):
        counts = csr_matrix(
            np.array(
                [
                    [3.0, 4.0, 12.0],
                    [0.0, 5.0, 0.0],
                    [6.0, 8.0, 0.0],
                    [0.0, 0.0, 10.0],
                ]
            )
        )
        labels = np.array([1, 1, 0, 0])

        difference_df = experiment.build_tf_s_difference_table(
            counts,
            labels,
            np.array(["first", "second", "third"]),
        )

        tf_s = np.array(
            [
                [3.0 / 13.0, 4.0 / 13.0, 12.0 / 13.0],
                [0.0, 1.0, 0.0],
                [0.6, 0.8, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        expected_label1 = tf_s[:2].mean(axis=0)
        expected_label0 = tf_s[2:].mean(axis=0)

        np.testing.assert_allclose(
            difference_df["label1_value"],
            expected_label1,
        )
        np.testing.assert_allclose(
            difference_df["label0_value"],
            expected_label0,
        )
        np.testing.assert_allclose(
            difference_df["absolute_difference"],
            np.abs(expected_label1 - expected_label0),
        )

    def test_tf_s_features_are_not_renormalized_after_term_selection(self):
        counts = csr_matrix(np.array([[3.0, 4.0, 12.0]]))
        selected_terms_df = pd.DataFrame(
            [
                {
                    "selection_mode": "tf_s",
                    "selection_rank": 1,
                    "term": "first",
                    "selection_source": "tf_s",
                },
                {
                    "selection_mode": "tf_s",
                    "selection_rank": 2,
                    "term": "second",
                    "selection_source": "tf_s",
                },
            ]
        )

        feature_matrices = experiment.build_selected_feature_matrices(
            counts,
            counts,
            np.array(["first", "second", "third"]),
            selected_terms_df,
            np.array([19.0]),
            np.array([19.0]),
        )

        np.testing.assert_allclose(
            feature_matrices["full_vocab_row_l2"]["train"],
            np.array([[3.0 / 13.0, 4.0 / 13.0]]),
        )
        self.assertLess(
            np.linalg.norm(
                feature_matrices["full_vocab_row_l2"]["train"][0]
            ),
            1.0,
        )
        self.assertEqual(
            feature_matrices["feature_metadata"]["full_vocab_row_l2"][
                "feature_transform"
            ].tolist(),
            [
                "full_vocabulary_raw_frequency_row_l2",
                "full_vocabulary_raw_frequency_row_l2",
            ],
        )

    def test_log_tf_s_difference_uses_full_vocabulary_log_row_l2(self):
        counts = csr_matrix(
            np.array(
                [
                    [1.0, 4.0, 9.0],
                    [0.0, 2.0, 0.0],
                    [3.0, 1.0, 0.0],
                    [0.0, 0.0, 5.0],
                ]
            )
        )
        labels = np.array([1, 1, 0, 0])

        difference_df = experiment.build_log_tf_s_difference_table(
            counts,
            labels,
            np.array(["first", "second", "third"]),
        )

        numerator = counts.toarray()
        positive_mask = numerator > 0
        numerator[positive_mask] = 1.0 + np.log(numerator[positive_mask])
        log_tf_s = numerator / np.linalg.norm(
            numerator,
            axis=1,
            keepdims=True,
        )
        expected_label1 = log_tf_s[:2].mean(axis=0)
        expected_label0 = log_tf_s[2:].mean(axis=0)

        np.testing.assert_allclose(
            difference_df["label1_value"],
            expected_label1,
        )
        np.testing.assert_allclose(
            difference_df["label0_value"],
            expected_label0,
        )
        np.testing.assert_allclose(
            difference_df["absolute_difference"],
            np.abs(expected_label1 - expected_label0),
        )

    def test_log_tf_s_features_are_not_renormalized_after_selection(self):
        counts = csr_matrix(np.array([[1.0, 4.0, 9.0]]))
        selected_terms_df = pd.DataFrame(
            [
                {
                    "selection_mode": "log_tf_s",
                    "selection_rank": 1,
                    "term": "first",
                    "selection_source": "log_tf_s",
                },
                {
                    "selection_mode": "log_tf_s",
                    "selection_rank": 2,
                    "term": "second",
                    "selection_source": "log_tf_s",
                },
            ]
        )

        feature_matrices = experiment.build_selected_feature_matrices(
            counts,
            counts,
            np.array(["first", "second", "third"]),
            selected_terms_df,
            np.array([14.0]),
            np.array([14.0]),
        )

        full_numerator = np.array(
            [1.0, 1.0 + np.log(4.0), 1.0 + np.log(9.0)]
        )
        expected = full_numerator[:2] / np.linalg.norm(full_numerator)
        np.testing.assert_allclose(
            feature_matrices["full_vocab_row_l2"]["train"],
            expected.reshape(1, -1),
        )
        self.assertLess(
            np.linalg.norm(
                feature_matrices["full_vocab_row_l2"]["train"][0]
            ),
            1.0,
        )
        self.assertEqual(
            feature_matrices["feature_metadata"]["full_vocab_row_l2"][
                "feature_transform"
            ].tolist(),
            [
                "full_vocabulary_log_frequency_row_l2",
                "full_vocabulary_log_frequency_row_l2",
            ],
        )

    def test_calculate_iqr_outlier_statistics_detects_high_value(self):
        statistics = experiment.calculate_iqr_outlier_statistics(
            [1.0, 2.0, 2.0, 3.0, 100.0]
        )

        self.assertEqual(statistics["outlier_count"], 1)
        self.assertTrue(statistics["has_outlier"])
        self.assertEqual(
            np.flatnonzero(statistics["outlier_mask"]).tolist(),
            [4],
        )

    def test_adds_label_specific_statistics_and_outlier_document_rows(self):
        X_train_count = csr_matrix(
            np.array([[1], [1], [1], [10], [2], [2], [2], [2]], dtype=float)
        )
        train_word_counts = np.full(8, 10.0)
        train_labels = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=int)
        train_doc_ids = np.arange(100, 108, dtype=int)
        selected_terms_df = pd.DataFrame(
            [
                {
                    "selection_mode": "term_frequency",
                    "selection_rank": 1,
                    "term": "token",
                    "selection_source": "term_frequency",
                }
            ]
        )

        diagnosed_df = experiment.add_term_frequency_iqr_diagnostics(
            selected_terms_df,
            X_train_count,
            train_labels,
            np.array(["token"]),
            train_word_counts,
        )
        outlier_rows = experiment.build_term_frequency_iqr_outlier_rows(
            diagnosed_df,
            X_train_count,
            train_labels,
            train_doc_ids,
            np.array(["token"]),
            train_word_counts,
            outer_fold=1,
            feature_count=1,
        )

        self.assertEqual(int(diagnosed_df.loc[0, "tf_label1_outlier_count"]), 1)
        self.assertEqual(int(diagnosed_df.loc[0, "tf_label0_outlier_count"]), 0)
        self.assertTrue(bool(diagnosed_df.loc[0, "tf_label1_has_outlier"]))
        self.assertEqual(len(outlier_rows), 1)
        self.assertEqual(outlier_rows[0]["doc_id"], 103)
        self.assertEqual(outlier_rows[0]["label"], 1)
        self.assertEqual(outlier_rows[0]["outlier_direction"], "high")

    def test_saves_boxplot_and_feature_count_summary_plot(self):
        outer_context = SimpleNamespace(
            X_train_count=csr_matrix(
                np.array(
                    [
                        [1, 0],
                        [2, 1],
                        [10, 1],
                        [1, 2],
                        [2, 2],
                        [2, 8],
                    ],
                    dtype=float,
                )
            ),
            train_word_counts=np.full(6, 10.0),
            train_labels=np.array([1, 1, 1, 0, 0, 0], dtype=int),
            feature_names=np.array(["token", "block"]),
        )
        selected_terms_df = pd.DataFrame(
            [
                {
                    "selection_rank": 1,
                    "term": "token",
                    "selection_source": "term_frequency",
                },
                {
                    "selection_rank": 2,
                    "term": "block",
                    "selection_source": "term_frequency",
                },
            ]
        )
        feature_count_summary_df = pd.DataFrame(
            [
                {
                    "feature_count": 1,
                    "evaluated_term_occurrences": 10,
                    "terms_with_label1_outlier": 8,
                    "terms_with_label0_outlier": 9,
                    "terms_with_any_label_outlier": 10,
                    "both_labels_zero_iqr_term_occurrences": 2,
                    "any_label_outlier_rate": 1.0,
                    "label1_zero_iqr_rate": 0.3,
                    "label0_zero_iqr_rate": 0.4,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            boxplot_path = Path(temp_dir) / "boxplot.png"
            summary_path = Path(temp_dir) / "summary.png"
            experiment.save_term_frequency_iqr_boxplot(
                outer_context,
                selected_terms_df,
                boxplot_path,
                outer_fold=1,
            )
            experiment.save_term_frequency_iqr_summary_plot(
                feature_count_summary_df,
                summary_path,
            )

            self.assertGreater(boxplot_path.stat().st_size, 0)
            self.assertGreater(summary_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
