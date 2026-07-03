from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from pathlib import Path


EXPERIMENT_DIR_NAME = "term_difference_topn_margin_nested_cv_experiment"
DEFAULT_RESULT_DIR = (
    Path(
        os.environ.get(
            "DIFFERENCES_BETWEEN_LABELS_OUTPUT_ROOT",
            r"D:\D_Student\HayatoTakehana",
        )
    )
    / "data"
    / "outputs"
    / "differences_between_labels"
    / EXPERIMENT_DIR_NAME
)

SUMMARY_COLUMNS = [
    "feature_count",
    "selection_mode",
    "feature_mode",
    "kernel",
    "mean_selected_term_count",
    "selected_term_counts",
    "mean_accuracy",
    "mean_recall",
    "mean_precision",
    "mean_f1",
    "best_cs",
    "best_gammas",
    "valid_param_count",
    "dropped_param_count",
    "timeout_param_count",
]

BEST_RESULT_COLUMNS = SUMMARY_COLUMNS[:-3]
AGGREGATED_BEST_COLUMNS = ["condition", *BEST_RESULT_COLUMNS]
CONDITION_LABELS = {
    "document_frequency": "DFのみ",
    "term_frequency": "TFのみ",
    "tfidf": "TF-IDFのみ",
    "document_tfidf_unique": "DF+TF-IDF重複なし",
    "document_tfidf_duplicate": "DF+TF-IDF重複あり",
    "combined_unique": "DF+TF重複なし",
    "combined_duplicate": "DF+TF重複あり",
}
KERNEL_AGGREGATE_FILENAMES = {
    "linear": "best_linear.csv",
    "rbf": "best_nonlinear.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "既存の svm_metrics.csv と svm_params.csv から、"
            "条件ごとの summary.csv だけを再生成します。"
        )
    )
    parser.add_argument(
        "result_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help=f"実験結果フォルダ。省略時: {DEFAULT_RESULT_DIR}",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="CSVの欠損や不正な配置を検出した場合に処理を中止します。",
    )
    return parser.parse_args()


def read_first_csv_row(path: Path) -> dict[str, str] | None:
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        return next(csv.DictReader(csv_file), None)


def normalize_feature_count(value: str) -> int | str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if number.is_integer():
        return int(number)
    return value


def count_param_statuses(params_path: Path) -> dict[str, int] | None:
    if not params_path.exists():
        return None

    with params_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        counts = Counter(
            row.get("status", "").strip()
            for row in csv.DictReader(csv_file)
            if row.get("status", "").strip()
        )
    return {
        "valid_param_count": counts["valid"],
        "dropped_param_count": counts["dropped"],
        "timeout_param_count": counts["timeout"],
    }


def resolve_summary_path(result_dir: Path, metrics_path: Path) -> Path:
    relative_parts = metrics_path.relative_to(result_dir).parts
    if (
        len(relative_parts) < 4
        or not relative_parts[0].startswith("n")
        or relative_parts[1] != "m"
    ):
        raise ValueError(
            "想定する n<特徴数>/m/<条件>/svm_metrics.csv 配置ではありません: "
            f"{metrics_path}"
        )

    condition_parts = relative_parts[2:-1]
    return result_dir / "m" / Path(*condition_parts) / "summary.csv"


def build_summary_row(
    metrics_row: dict[str, str],
    status_counts: dict[str, int] | None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "feature_count": normalize_feature_count(metrics_row.get("feature_count", "")),
        "selection_mode": metrics_row.get("selection_mode", ""),
        "feature_mode": metrics_row.get("feature_mode", ""),
        "kernel": metrics_row.get("kernel", ""),
        "mean_selected_term_count": metrics_row.get("mean_selected_term_count", ""),
        "selected_term_counts": metrics_row.get("selected_term_counts", ""),
        "mean_accuracy": metrics_row.get("mean_accuracy", ""),
        "mean_recall": metrics_row.get("mean_recall", ""),
        "mean_precision": metrics_row.get("mean_precision", ""),
        "mean_f1": metrics_row.get("mean_f1", ""),
        "best_cs": metrics_row.get("best_cs", ""),
        "best_gammas": metrics_row.get("best_gammas", ""),
        "valid_param_count": "",
        "dropped_param_count": "",
        "timeout_param_count": "",
    }
    if status_counts is not None:
        row.update(status_counts)
    return row


def feature_count_sort_key(row: dict[str, object]) -> tuple[int, float | str]:
    feature_count = row["feature_count"]
    if isinstance(feature_count, int):
        return (0, float(feature_count))
    try:
        return (0, float(str(feature_count)))
    except ValueError:
        return (1, str(feature_count))


def select_best_row(
    rows: list[dict[str, object]],
) -> dict[str, object] | None:
    best_row = None
    best_accuracy = float("-inf")
    best_feature_count = float("inf")

    for row in rows:
        try:
            accuracy = float(str(row["mean_accuracy"]))
            feature_count = float(str(row["feature_count"]))
        except (KeyError, TypeError, ValueError):
            continue

        if accuracy > best_accuracy or (
            accuracy == best_accuracy and feature_count < best_feature_count
        ):
            best_row = row
            best_accuracy = accuracy
            best_feature_count = feature_count

    return best_row


def condition_sort_key(row: dict[str, object]) -> tuple[int, str]:
    selection_mode = str(row.get("selection_mode", ""))
    condition_order = {
        selection_mode: index
        for index, selection_mode in enumerate(CONDITION_LABELS)
    }
    return (
        condition_order.get(selection_mode, len(condition_order)),
        selection_mode,
    )


def write_summary_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_aggregated_best_csv(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=AGGREGATED_BEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "condition": CONDITION_LABELS.get(
                        str(row.get("selection_mode", "")),
                        str(row.get("selection_mode", "")),
                    ),
                    **{
                        column: row.get(column, "")
                        for column in BEST_RESULT_COLUMNS
                    },
                }
            )


def save_aggregated_best_csvs(
    result_dir: Path,
    rows_by_summary_path: dict[Path, list[dict[str, object]]],
) -> list[Path]:
    best_rows_by_kernel: dict[str, list[dict[str, object]]] = defaultdict(list)
    for rows in rows_by_summary_path.values():
        best_row = select_best_row(rows)
        if best_row is not None:
            best_rows_by_kernel[str(best_row.get("kernel", ""))].append(best_row)

    saved_paths = []
    for kernel, filename in KERNEL_AGGREGATE_FILENAMES.items():
        rows = best_rows_by_kernel.get(kernel, [])
        rows.sort(key=condition_sort_key)
        output_path = result_dir / "m" / filename
        write_aggregated_best_csv(rows, output_path)
        saved_paths.append(output_path)
        print(f"saved: {output_path} ({len(rows)} rows)")
    return saved_paths


def handle_problem(message: str, strict: bool) -> None:
    if strict:
        raise RuntimeError(message)
    print(f"warning: {message}")


def rebuild_summaries(result_dir: Path, strict: bool = False) -> list[Path]:
    result_dir = result_dir.expanduser().resolve()
    if not result_dir.is_dir():
        raise FileNotFoundError(f"実験結果フォルダが見つかりません: {result_dir}")

    metrics_paths = sorted(result_dir.glob("n*/m/**/svm_metrics.csv"))
    if not metrics_paths:
        raise FileNotFoundError(
            f"集計対象の svm_metrics.csv が見つかりません: {result_dir}"
        )

    rows_by_summary_path: dict[Path, list[dict[str, object]]] = defaultdict(list)
    for metrics_path in metrics_paths:
        metrics_row = read_first_csv_row(metrics_path)
        if metrics_row is None:
            handle_problem(f"データ行がないためスキップします: {metrics_path}", strict)
            continue

        try:
            summary_path = resolve_summary_path(result_dir, metrics_path)
        except ValueError as exc:
            handle_problem(str(exc), strict)
            continue

        params_path = metrics_path.with_name("svm_params.csv")
        status_counts = count_param_statuses(params_path)
        if status_counts is None:
            handle_problem(
                f"svm_params.csv がないためパラメータ件数を空欄にします: {params_path}",
                strict,
            )

        rows_by_summary_path[summary_path].append(
            build_summary_row(metrics_row, status_counts)
        )

    if not rows_by_summary_path:
        raise RuntimeError("summary.csv に書き出せる結果がありません。")

    saved_paths = []
    for summary_path, rows in sorted(rows_by_summary_path.items()):
        rows.sort(key=feature_count_sort_key)
        write_summary_csv(rows, summary_path)
        saved_paths.append(summary_path)
        print(f"saved: {summary_path} ({len(rows)} rows)")
    saved_paths.extend(save_aggregated_best_csvs(result_dir, rows_by_summary_path))
    return saved_paths


def main() -> None:
    args = parse_args()
    saved_paths = rebuild_summaries(args.result_dir, strict=args.strict)
    print(f"completed: {len(saved_paths)} CSV files")


if __name__ == "__main__":
    main()
