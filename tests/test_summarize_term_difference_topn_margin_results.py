import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from active.tf_delete_v1_predata.analysis import (
    summarize_term_difference_topn_margin_results as summarizer,
)


class SummarizeTermDifferenceResultsTests(unittest.TestCase):
    def write_result(
        self,
        result_dir,
        feature_count,
        selection_dir,
        feature_dir,
        kernel_dir,
        selection_mode,
        feature_mode,
        kernel,
        accuracy,
    ):
        condition_dir = (
            result_dir
            / f"n{feature_count}"
            / "m"
            / selection_dir
            / feature_dir
            / kernel_dir
        )
        condition_dir.mkdir(parents=True)
        metrics_path = condition_dir / "svm_metrics.csv"
        row = {
            "feature_count": feature_count,
            "selection_mode": selection_mode,
            "feature_mode": feature_mode,
            "kernel": kernel,
            "mean_selected_term_count": feature_count,
            "selected_term_counts": str(feature_count),
            "mean_accuracy": accuracy,
            "mean_recall": 0.8,
            "mean_precision": 0.8,
            "mean_f1": 0.8,
            "best_cs": "1",
            "best_gammas": "0.1" if kernel == "rbf" else "",
        }
        with metrics_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=row)
            writer.writeheader()
            writer.writerow(row)

        with (condition_dir / "svm_params.csv").open(
            "w", encoding="utf-8", newline=""
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["status"])
            writer.writeheader()
            writer.writerow({"status": "valid"})

    def read_rows(self, path):
        with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            return list(csv.DictReader(csv_file))

    def test_writes_best_linear_and_nonlinear_csvs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_dir = Path(temp_dir)
            self.write_result(
                result_dir,
                100,
                "tf",
                "minmax",
                "lin",
                "term_frequency",
                "column_min_max",
                "linear",
                0.95,
            )
            self.write_result(
                result_dir,
                200,
                "tf",
                "minmax",
                "rbf",
                "term_frequency",
                "column_min_max",
                "rbf",
                0.92,
            )
            self.write_result(
                result_dir,
                150,
                "log_tf",
                "minmax",
                "lin",
                "log_term_frequency",
                "column_min_max",
                "linear",
                0.96,
            )
            self.write_result(
                result_dir,
                250,
                "log_tf",
                "minmax",
                "rbf",
                "log_term_frequency",
                "column_min_max",
                "rbf",
                0.93,
            )
            self.write_result(
                result_dir,
                125,
                "tf",
                "row_l2",
                "lin",
                "term_frequency",
                "row_l2",
                "linear",
                0.97,
            )
            self.write_result(
                result_dir,
                225,
                "tf",
                "row_l2",
                "rbf",
                "term_frequency",
                "row_l2",
                "rbf",
                0.94,
            )
            self.write_result(
                result_dir,
                175,
                "log_tf",
                "row_l2",
                "lin",
                "log_term_frequency",
                "row_l2",
                "linear",
                0.98,
            )
            self.write_result(
                result_dir,
                275,
                "log_tf",
                "row_l2",
                "rbf",
                "log_term_frequency",
                "row_l2",
                "rbf",
                0.95,
            )
            self.write_result(
                result_dir,
                300,
                "tf_s",
                "full_vocab_row_l2",
                "lin",
                "tf_s",
                "full_vocab_row_l2",
                "linear",
                0.99,
            )
            self.write_result(
                result_dir,
                350,
                "tf_s",
                "full_vocab_row_l2",
                "rbf",
                "tf_s",
                "full_vocab_row_l2",
                "rbf",
                0.96,
            )
            self.write_result(
                result_dir,
                400,
                "log_tf_s",
                "full_vocab_row_l2",
                "lin",
                "log_tf_s",
                "full_vocab_row_l2",
                "linear",
                0.995,
            )
            self.write_result(
                result_dir,
                450,
                "log_tf_s",
                "full_vocab_row_l2",
                "rbf",
                "log_tf_s",
                "full_vocab_row_l2",
                "rbf",
                0.97,
            )
            self.write_result(
                result_dir,
                500,
                "tf",
                "minmax_df_gt_2pct",
                "lin",
                "term_frequency",
                "column_min_max_df_gt_2pct",
                "linear",
                0.91,
            )
            self.write_result(
                result_dir,
                550,
                "tf",
                "minmax_df_gt_2pct",
                "rbf",
                "term_frequency",
                "column_min_max_df_gt_2pct",
                "rbf",
                0.90,
            )
            self.write_result(
                result_dir,
                600,
                "tf",
                "row_l2_df_gt_2pct",
                "lin",
                "term_frequency",
                "row_l2_df_gt_2pct",
                "linear",
                0.92,
            )
            self.write_result(
                result_dir,
                650,
                "tf",
                "row_l2_df_gt_2pct",
                "rbf",
                "term_frequency",
                "row_l2_df_gt_2pct",
                "rbf",
                0.91,
            )
            self.write_result(
                result_dir,
                700,
                "log_tf",
                "minmax_df_gt_2pct",
                "lin",
                "log_term_frequency",
                "column_min_max_df_gt_2pct",
                "linear",
                0.93,
            )
            self.write_result(
                result_dir,
                750,
                "log_tf",
                "minmax_df_gt_2pct",
                "rbf",
                "log_term_frequency",
                "column_min_max_df_gt_2pct",
                "rbf",
                0.92,
            )
            self.write_result(
                result_dir,
                800,
                "log_tf",
                "row_l2_df_gt_2pct",
                "lin",
                "log_term_frequency",
                "row_l2_df_gt_2pct",
                "linear",
                0.94,
            )
            self.write_result(
                result_dir,
                850,
                "log_tf",
                "row_l2_df_gt_2pct",
                "rbf",
                "log_term_frequency",
                "row_l2_df_gt_2pct",
                "rbf",
                0.93,
            )
            self.write_result(
                result_dir,
                900,
                "tf_s",
                "full_vocab_row_l2_df_gt_2pct",
                "lin",
                "tf_s",
                "full_vocab_row_l2_df_gt_2pct",
                "linear",
                0.95,
            )
            self.write_result(
                result_dir,
                950,
                "tf_s",
                "full_vocab_row_l2_df_gt_2pct",
                "rbf",
                "tf_s",
                "full_vocab_row_l2_df_gt_2pct",
                "rbf",
                0.94,
            )
            self.write_result(
                result_dir,
                1000,
                "log_tf_s",
                "full_vocab_row_l2_df_gt_2pct",
                "lin",
                "log_tf_s",
                "full_vocab_row_l2_df_gt_2pct",
                "linear",
                0.96,
            )
            self.write_result(
                result_dir,
                1050,
                "log_tf_s",
                "full_vocab_row_l2_df_gt_2pct",
                "rbf",
                "log_tf_s",
                "full_vocab_row_l2_df_gt_2pct",
                "rbf",
                0.95,
            )
            self.write_result(
                result_dir,
                50,
                "df",
                "src",
                "lin",
                "document_frequency",
                "source_specific",
                "linear",
                0.99,
            )

            summarizer.rebuild_summaries(result_dir)

            linear_rows = self.read_rows(result_dir / "m" / "best_linear.csv")
            nonlinear_rows = self.read_rows(
                result_dir / "m" / "best_nonlinear.csv"
            )

            self.assertEqual(list(linear_rows[0]), summarizer.AGGREGATED_BEST_COLUMNS)
            self.assertEqual(
                [row["condition"] for row in linear_rows],
                [
                    "相対出現頻度TF × 列Min-Max",
                    "相対出現頻度TF × 行L2",
                    "対数補正TF × 列Min-Max",
                    "対数補正TF × 行L2",
                    "TF_S × 全語彙行L2",
                    "log-TF_S × 全語彙行L2",
                    "相対出現頻度TF × 列Min-Max（DF>2%）",
                    "相対出現頻度TF × 行L2（DF>2%）",
                    "対数補正TF × 列Min-Max（DF>2%）",
                    "対数補正TF × 行L2（DF>2%）",
                    "TF_S × 全語彙行L2（DF>2%）",
                    "log-TF_S × 全語彙行L2（DF>2%）",
                ],
            )
            self.assertEqual(linear_rows[0]["feature_count"], "100")
            self.assertEqual(linear_rows[0]["mean_accuracy"], "0.95")
            self.assertEqual(linear_rows[1]["feature_count"], "125")
            self.assertEqual(linear_rows[2]["feature_count"], "150")
            self.assertEqual(linear_rows[3]["mean_accuracy"], "0.98")
            self.assertEqual(linear_rows[4]["feature_count"], "300")
            self.assertEqual(linear_rows[5]["feature_count"], "400")
            self.assertEqual(linear_rows[6]["feature_count"], "500")
            self.assertEqual(linear_rows[11]["feature_count"], "1000")
            self.assertEqual(
                [row["condition"] for row in nonlinear_rows],
                [
                    "相対出現頻度TF × 列Min-Max",
                    "相対出現頻度TF × 行L2",
                    "対数補正TF × 列Min-Max",
                    "対数補正TF × 行L2",
                    "TF_S × 全語彙行L2",
                    "log-TF_S × 全語彙行L2",
                    "相対出現頻度TF × 列Min-Max（DF>2%）",
                    "相対出現頻度TF × 行L2（DF>2%）",
                    "対数補正TF × 列Min-Max（DF>2%）",
                    "対数補正TF × 行L2（DF>2%）",
                    "TF_S × 全語彙行L2（DF>2%）",
                    "log-TF_S × 全語彙行L2（DF>2%）",
                ],
            )
            self.assertEqual(nonlinear_rows[11]["feature_count"], "1050")
            self.assertEqual(nonlinear_rows[0]["kernel"], "rbf")


if __name__ == "__main__":
    unittest.main()
