from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "data" / "outputs" / "another_data"
DATASET_DIR_NAME = "dcstmnce_final_dataset"
DATASET_LABEL = "Final_Dataset スコアのみ"

FEATURE_MODE_LABELS = {
    "binary": "バイナリー",
    "count": "出現頻度",
}
KERNEL_LABELS = {
    "linear": "線形SVM",
    "rbf": "非線形SVM",
}
SUMMARY_COLUMNS = [
    "データ種別",
    "単語数特徴",
    "SVM",
    "特徴量",
    "スコア（以上）",
    "平均特徴数",
    "Accuracy",
    "Recall",
    "Precision",
    "F1 Score",
    "元ファイル",
]
ORGANIZED_OUTPUT_COLUMNS = [
    "外側の層",
    "特徴数",
    "ハイパーパラメータC",
    "ガンマ",
    "accuracy",
    "recall",
    "prescision",
    "f1_score",
]


@dataclass(frozen=True)
class SummaryRow:
    dataset_label: str
    kernel: str
    feature_mode: str
    score_threshold: str
    mean_feature_count: str
    accuracy: str
    recall: str
    precision: str
    f1_score: str
    source_path: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "データ種別": self.dataset_label,
            "単語数特徴": "-",
            "SVM": self.kernel,
            "特徴量": self.feature_mode,
            "スコア（以上）": self.score_threshold,
            "平均特徴数": self.mean_feature_count,
            "Accuracy": self.accuracy,
            "Recall": self.recall,
            "Precision": self.precision,
            "F1 Score": self.f1_score,
            "元ファイル": str(self.source_path),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="another_data の SVM 結果から summary table と organized_metrics.csv を同時作成します。",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["all", "binary", "count"],
        default="all",
        help="summary table の特徴量の種類で絞り込みます。",
    )
    parser.add_argument(
        "--kernel",
        choices=["all", "linear", "rbf"],
        default="all",
        help="summary table のSVMカーネルで絞り込みます。",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="F1 Score の高い順で上位N件だけ summary table に出力します。0なら全件です。",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="入力フォルダや svm_folds.csv が存在しない場合にエラー終了します。",
    )
    return parser.parse_args()


def dataset_root() -> Path:
    return OUTPUT_ROOT / DATASET_DIR_NAME


def summary_path() -> Path:
    return OUTPUT_ROOT / "summary_tables" / f"{DATASET_DIR_NAME}_metric_summary.csv"


def read_first_csv_row(path: Path) -> dict[str, str] | None:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return next(csv.DictReader(f), None)


def format_float(value: object, digits: int = 3) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.{digits}f}"


def format_number(value: object, decimals: int | None = None) -> str:
    if pd.isna(value) or value == "":
        return ""
    if decimals is None:
        if float(value).is_integer():
            return str(int(value))
        return format(float(value), "g")
    formatted = f"{float(value):.{decimals}f}".rstrip("0").rstrip(".")
    return formatted if formatted else "0"


def format_score_threshold(value: str) -> str:
    if value == "":
        return ""
    parts = [part.strip() for part in value.split(",")]
    if len(parts) > 1:
        return ",".join(format_float(part, 2) for part in parts if part)
    return format_float(value, 2)


def mean_model_feature_count(metric_path: Path) -> str:
    folds_path = metric_path.with_name("svm_folds.csv")
    if not folds_path.exists():
        return ""

    values = []
    with folds_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            raw_value = row.get("model_feature_count", "")
            if raw_value != "":
                values.append(float(raw_value))

    if not values:
        return ""
    return f"{sum(values) / len(values):.1f}"


def resolve_mean_feature_count(metric_path: Path, metric_row: dict[str, str]) -> str:
    model_feature_count = mean_model_feature_count(metric_path)
    if model_feature_count:
        return model_feature_count
    return format_number(metric_row.get("mean_selected_term_count", ""), decimals=1)


def iter_metric_rows(strict: bool) -> list[SummaryRow]:
    root = dataset_root()
    if not root.exists():
        if strict:
            raise FileNotFoundError(f"集計対象フォルダが見つかりません: {root}")
        return []

    rows = []
    for metric_path in sorted(root.rglob("svm_metrics.csv")):
        raw = read_first_csv_row(metric_path)
        if not raw:
            continue

        feature_mode = raw.get("feature_mode", "")
        kernel = raw.get("kernel", "")
        score_threshold = raw.get("candidate_score_thresholds", "") or raw.get("score_threshold", "")
        rows.append(
            SummaryRow(
                dataset_label=DATASET_LABEL,
                kernel=KERNEL_LABELS.get(kernel, kernel),
                feature_mode=FEATURE_MODE_LABELS.get(feature_mode, feature_mode),
                score_threshold=format_score_threshold(score_threshold),
                mean_feature_count=resolve_mean_feature_count(metric_path, raw),
                accuracy=format_float(raw.get("mean_accuracy", "")),
                recall=format_float(raw.get("mean_recall", "")),
                precision=format_float(raw.get("mean_precision", "")),
                f1_score=format_float(raw.get("mean_f1", "")),
                source_path=metric_path,
            )
        )
    return rows


def row_matches_filters(row: SummaryRow, args: argparse.Namespace) -> bool:
    if args.feature_mode != "all" and row.feature_mode != FEATURE_MODE_LABELS[args.feature_mode]:
        return False
    if args.kernel != "all" and row.kernel != KERNEL_LABELS[args.kernel]:
        return False
    return True


def sort_rows(rows: list[SummaryRow]) -> list[SummaryRow]:
    def sort_key(row: SummaryRow) -> tuple[object, ...]:
        f1 = float(row.f1_score) if row.f1_score else -1.0
        first_score = row.score_threshold.split(",", maxsplit=1)[0]
        score = float(first_score) if first_score else -1.0
        return (row.kernel, row.feature_mode, score, f1)

    return sorted(rows, key=sort_key)


def take_top_rows(rows: list[SummaryRow], top: int) -> list[SummaryRow]:
    if top <= 0:
        return rows
    return sorted(rows, key=lambda row: float(row.f1_score or "-1"), reverse=True)[:top]


def write_summary_csv(rows: list[SummaryRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def print_table(rows: list[SummaryRow]) -> None:
    display_columns = SUMMARY_COLUMNS[:-1]
    values = [[row.as_dict()[column] for column in display_columns] for row in rows]
    if values:
        widths = [
            max(len(column), *(len(value) for value in column_values))
            for column, column_values in zip(display_columns, zip(*values), strict=False)
        ]
    else:
        widths = [len(column) for column in display_columns]

    print(" | ".join(column.ljust(width) for column, width in zip(display_columns, widths, strict=False)))
    print("-+-".join("-" * width for width in widths))
    for value_row in values:
        print(" | ".join(value.ljust(width) for value, width in zip(value_row, widths, strict=False)))


def build_summary_output(args: argparse.Namespace) -> tuple[Path, list[SummaryRow]]:
    rows = sort_rows(iter_metric_rows(strict=args.strict))
    rows = [row for row in rows if row_matches_filters(row, args)]
    rows = take_top_rows(rows, args.top)
    output_path = summary_path()
    write_summary_csv(rows, output_path)
    return output_path, rows


def select_feature_count_column(folds_df: pd.DataFrame) -> str:
    if "model_feature_count" in folds_df.columns:
        return "model_feature_count"
    return "selected_term_count"


def format_series(folds_df: pd.DataFrame, column_name: str, decimals: int | None = None) -> pd.Series:
    if column_name not in folds_df.columns:
        return pd.Series([""] * len(folds_df))
    return folds_df[column_name].map(lambda value: format_number(value, decimals))


def mean_formatted(folds_df: pd.DataFrame, column_name: str, decimals: int) -> str:
    if column_name not in folds_df.columns:
        return ""
    return format_number(folds_df[column_name].mean(), decimals)


def build_organized_dataframe(folds_df: pd.DataFrame) -> pd.DataFrame:
    feature_count_column = select_feature_count_column(folds_df)
    output_df = pd.DataFrame(
        {
            "外側の層": folds_df["outer_fold"],
            "特徴数": format_series(folds_df, feature_count_column),
            "ハイパーパラメータC": format_series(folds_df, "c"),
            "ガンマ": format_series(folds_df, "gamma"),
            "accuracy": format_series(folds_df, "accuracy", 3),
            "recall": format_series(folds_df, "recall", 3),
            "prescision": format_series(folds_df, "precision", 3),
            "f1_score": format_series(folds_df, "f1_score", 3),
        }
    )
    average_row = {
        "外側の層": "平均",
        "特徴数": mean_formatted(folds_df, feature_count_column, 1),
        "ハイパーパラメータC": "",
        "ガンマ": "",
        "accuracy": mean_formatted(folds_df, "accuracy", 3),
        "recall": mean_formatted(folds_df, "recall", 3),
        "prescision": mean_formatted(folds_df, "precision", 3),
        "f1_score": mean_formatted(folds_df, "f1_score", 3),
    }
    return pd.concat([output_df, pd.DataFrame([average_row])], ignore_index=True)[ORGANIZED_OUTPUT_COLUMNS]


def save_organized_csv(source_path: Path) -> Path:
    folds_df = pd.read_csv(source_path)
    output_path = source_path.with_name("organized_metrics.csv")
    build_organized_dataframe(folds_df).to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def build_organized_outputs(strict: bool) -> list[Path]:
    root = dataset_root()
    if not root.exists():
        if strict:
            raise FileNotFoundError(f"整理対象フォルダが見つかりません: {root}")
        return []

    fold_paths = sorted(root.rglob("svm_folds.csv"))
    if strict and not fold_paths:
        raise FileNotFoundError(f"No svm_folds.csv found under: {root}")
    return [save_organized_csv(path) for path in fold_paths]


def main() -> int:
    args = parse_args()
    output_path, summary_rows = build_summary_output(args)
    organized_paths = build_organized_outputs(strict=args.strict)

    print_table(summary_rows)
    print()
    print(f"summary saved: {output_path}")
    print(f"summary rows: {len(summary_rows)}")
    print()
    print(f"organized_metrics saved: {len(organized_paths)}")
    for path in organized_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
