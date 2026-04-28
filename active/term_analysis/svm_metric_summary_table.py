from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "data" / "outputs" / "ta"

DATASET_DIR_ALIASES = {
    "dctmnce": "dcstmnce",
    "dcstmnce": "dcstmnce",
    "score": "dcstmnce",
    "dcwc_mncs": "dcwc_mncs",
    "score_word_count": "dcwc_mncs",
}
DATASET_INPUT_ALIASES = {
    "1": "dctmnce",
    "dctmnce": "dctmnce",
    "dcstmnce": "dctmnce",
    "score": "dctmnce",
    "2": "dcwc_mncs",
    "dcwc_mncs": "dcwc_mncs",
    "score_word_count": "dcwc_mncs",
}
WORD_COUNT_INPUT_ALIASES = {
    "1": "raw",
    "raw": "raw",
    "count": "raw",
    "2": "log",
    "log": "log",
    "log1p": "log",
    "3": "all",
    "all": "all",
}
WORD_COUNT_LABELS = {
    "wc": "raw",
    "wcl": "log",
    "wcb": "both",
}
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


@dataclass(frozen=True)
class SummaryRow:
    dataset_label: str
    word_count_mode: str
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
            "単語数特徴": self.word_count_mode,
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
        description="保存済みSVM実験結果から、スコア・平均特徴数・精度指標を一覧化します。",
    )
    parser.add_argument(
        "dataset_arg",
        nargs="?",
        choices=sorted(DATASET_DIR_ALIASES),
        help="集計対象。dctmnce はスコアのみ、dcwc_mncs はスコア＋単語数です。",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_DIR_ALIASES),
        default=None,
        help="集計対象。位置引数を指定した場合は位置引数が優先されます。",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["all", "binary", "count"],
        default="all",
        help="特徴量の種類で絞り込みます。",
    )
    parser.add_argument(
        "--kernel",
        choices=["all", "linear", "rbf"],
        default="all",
        help="SVMカーネルで絞り込みます。",
    )
    parser.add_argument(
        "--word-count-mode",
        choices=["all", "raw", "log", "both"],
        default="all",
        help="dcwc_mncs 用。単語数特徴の種類で絞り込みます。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="出力CSVパス。未指定なら data/outputs/ta/summary_tables/ に保存します。",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="F1 Score の高い順で上位N件だけ出力します。0なら全件です。",
    )
    args = parser.parse_args()
    selected_dataset = args.dataset_arg or args.dataset
    dataset_was_prompted = selected_dataset is None
    args.dataset = selected_dataset or prompt_dataset_choice()
    if dataset_was_prompted and args.dataset in ("dcwc_mncs", "score_word_count") and args.word_count_mode == "all":
        args.word_count_mode = prompt_word_count_mode_choice()
    return args


def prompt_choice(prompt_label: str, option_lines: list[str], alias_to_value: dict[str, str], default_input: str) -> str:
    print(prompt_label)
    for option_line in option_lines:
        print(option_line)

    while True:
        raw = input(f"選択してください [{default_input}]: ").strip().lower()
        if raw == "":
            raw = default_input.lower()
        if raw in alias_to_value:
            return alias_to_value[raw]
        print("入力が正しくありません。候補の番号または名前で入力してください。")


def prompt_dataset_choice() -> str:
    return prompt_choice(
        "\n1. 集計する実験結果を選択してください。",
        [
            "  1: スコアのみ（dctmnce）",
            "  2: スコア＋単語数（dcwc_mncs）",
        ],
        DATASET_INPUT_ALIASES,
        default_input="1",
    )


def prompt_word_count_mode_choice() -> str:
    return prompt_choice(
        "\n2. 単語数特徴の種類を選択してください。",
        [
            "  1: 単語数をそのまま追加（raw）",
            "  2: log1p(単語数) を追加（log）",
            "  3: 全て",
        ],
        WORD_COUNT_INPUT_ALIASES,
        default_input="3",
    )


def normalize_dataset_name(raw_dataset: str) -> str:
    return DATASET_DIR_ALIASES[raw_dataset]


def dataset_label(dataset_dir_name: str) -> str:
    if dataset_dir_name == "dcwc_mncs":
        return "スコア＋単語数"
    return "スコアのみ"


def read_first_csv_row(path: Path) -> dict[str, str] | None:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = csv.DictReader(f)
        return next(rows, None)


def format_float(value: str, digits: int = 3) -> str:
    if value == "":
        return ""
    return f"{float(value):.{digits}f}"


def format_feature_count(value: str) -> str:
    if value == "":
        return ""
    return f"{float(value):.1f}"


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
    # `mean_selected_term_count` は単語特徴だけの平均。
    # `model_feature_count` があれば、単語数特徴を追加した実際のSVM入力特徴数を優先する。
    model_feature_count = mean_model_feature_count(metric_path)
    if model_feature_count:
        return model_feature_count
    return format_feature_count(metric_row.get("mean_selected_term_count", ""))


def parse_word_count_mode(metric_path: Path, dataset_dir_name: str) -> str:
    if dataset_dir_name != "dcwc_mncs":
        return "-"

    parts = list(metric_path.parts)
    try:
        dataset_index = parts.index("dcwc_mncs")
    except ValueError:
        return ""

    if dataset_index + 1 >= len(parts):
        return ""
    short_name = parts[dataset_index + 1]
    return WORD_COUNT_LABELS.get(short_name, short_name)


def row_matches_filters(
    row: SummaryRow,
    feature_mode: str,
    kernel: str,
    word_count_mode: str,
) -> bool:
    if feature_mode != "all" and row.feature_mode != FEATURE_MODE_LABELS[feature_mode]:
        return False
    if kernel != "all" and row.kernel != KERNEL_LABELS[kernel]:
        return False
    if word_count_mode != "all" and row.word_count_mode != word_count_mode:
        return False
    return True


def iter_metric_rows(dataset_dir_name: str) -> Iterable[SummaryRow]:
    dataset_root = OUTPUT_ROOT / dataset_dir_name
    if not dataset_root.exists():
        raise FileNotFoundError(f"集計対象フォルダが見つかりません: {dataset_root}")

    for metric_path in sorted(dataset_root.rglob("svm_metrics.csv")):
        raw = read_first_csv_row(metric_path)
        if not raw:
            continue

        feature_mode = raw.get("feature_mode", "")
        kernel = raw.get("kernel", "")
        yield SummaryRow(
            dataset_label=dataset_label(dataset_dir_name),
            word_count_mode=parse_word_count_mode(metric_path, dataset_dir_name),
            kernel=KERNEL_LABELS.get(kernel, kernel),
            feature_mode=FEATURE_MODE_LABELS.get(feature_mode, feature_mode),
            score_threshold=format_float(raw.get("candidate_score_thresholds", ""), digits=2),
            mean_feature_count=resolve_mean_feature_count(metric_path, raw),
            accuracy=format_float(raw.get("mean_accuracy", "")),
            recall=format_float(raw.get("mean_recall", "")),
            precision=format_float(raw.get("mean_precision", "")),
            f1_score=format_float(raw.get("mean_f1", "")),
            source_path=metric_path,
        )


def sort_rows(rows: list[SummaryRow]) -> list[SummaryRow]:
    def sort_key(row: SummaryRow) -> tuple:
        f1 = float(row.f1_score) if row.f1_score else -1.0
        score = float(row.score_threshold) if row.score_threshold else -1.0
        return (row.dataset_label, row.word_count_mode, row.kernel, row.feature_mode, score, f1)

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
    widths = [
        max(len(column), *(len(value) for value in column_values))
        for column, column_values in zip(display_columns, zip(*values), strict=False)
    ] if values else [len(column) for column in display_columns]

    header = " | ".join(column.ljust(width) for column, width in zip(display_columns, widths, strict=False))
    separator = "-+-".join("-" * width for width in widths)
    print(header)
    print(separator)
    for value_row in values:
        print(" | ".join(value.ljust(width) for value, width in zip(value_row, widths, strict=False)))


def default_output_path(dataset_dir_name: str) -> Path:
    return OUTPUT_ROOT / "summary_tables" / f"{dataset_dir_name}_metric_summary.csv"


def main() -> None:
    args = parse_args()
    dataset_dir_name = normalize_dataset_name(args.dataset)

    rows = sort_rows(list(iter_metric_rows(dataset_dir_name)))
    rows = [
        row
        for row in rows
        if row_matches_filters(row, args.feature_mode, args.kernel, args.word_count_mode)
    ]
    rows = take_top_rows(rows, args.top)

    output_path = args.output or default_output_path(dataset_dir_name)
    write_summary_csv(rows, output_path)

    print_table(rows)
    print()
    print(f"saved: {output_path}")
    print(f"rows : {len(rows)}")


if __name__ == "__main__":
    main()
