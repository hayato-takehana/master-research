from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def _bootstrap_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "active").exists() and (candidate / "archive").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise RuntimeError("Project root could not be found.")


PROJECT_ROOT = _bootstrap_project_root()

from active.term_analysis.doc_count_score_threshold_margin_nested_cv_experiment import (  # noqa: E402
    N_SPLITS,
    RANDOM_STATE,
    SplitFeatureContext,
    load_corpus,
)


OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "ta" / "nearest_pdf"

# 引数なしで実行する場合はここを書き換えてください。
DEFAULT_TEST_PDF = "63_0.pdf"
DEFAULT_SCORE_THRESHOLD = 0.11
DEFAULT_FEATURE_MODE = "binary"  # "count" or "binary"
DEFAULT_RBF_GAMMA = 0.01
DEFAULT_TOP_N = 10
DEFAULT_OUTPUT = str(OUTPUT_DIR / "nearest_pdf_in_svm_feature_space.csv")


def normalize_pdf_name(value: str) -> str:
    value = str(value).strip()
    if value.lower().endswith(".pdf"):
        return Path(value).name
    return f"{value}.pdf"


def find_pdf_id(pdf_names: np.ndarray, pdf_name_or_number: str) -> int:
    target = normalize_pdf_name(pdf_name_or_number)
    matches = np.flatnonzero(pdf_names == target)
    if len(matches) == 1:
        return int(matches[0])
    if len(matches) > 1:
        raise ValueError(f"PDF name is not unique: {target}")

    partial_matches = [str(name) for name in pdf_names if str(pdf_name_or_number).strip() in str(name)]
    if partial_matches:
        raise ValueError(
            f"PDF not found as an exact name: {target}\n"
            f"Partial matches: {', '.join(partial_matches[:20])}"
        )
    raise ValueError(f"PDF not found: {target}")


def iter_outer_splits(documents: np.ndarray, labels: np.ndarray):
    outer_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for outer_fold, (train_index, test_index) in enumerate(outer_cv.split(documents, labels), start=1):
        yield outer_fold, train_index, test_index


def dense_matrix(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)


def dense_row_values(matrix, row_index: int) -> np.ndarray:
    row = matrix[row_index]
    if hasattr(row, "toarray"):
        return np.asarray(row.toarray()).ravel()
    return np.asarray(row).ravel()


def find_outer_context_for_test_pdf(
    documents: np.ndarray,
    labels: np.ndarray,
    doc_ids: np.ndarray,
    test_doc_id: int,
) -> tuple[int, SplitFeatureContext]:
    matched = None
    for outer_fold, train_index, test_index in iter_outer_splits(documents, labels):
        if test_doc_id not in set(map(int, test_index)):
            continue
        if matched is not None:
            raise RuntimeError(f"Expected one matching outer fold, but found at least two: {matched[0]}, {outer_fold}")
        matched = (
            outer_fold,
            SplitFeatureContext(
                documents[train_index],
                labels[train_index],
                documents[test_index],
                labels[test_index],
                doc_ids[train_index],
                doc_ids[test_index],
            ),
        )

    if matched is None:
        raise RuntimeError("No outer fold has the requested PDF in the test data.")
    return matched


def compute_nearest_rows(
    context: SplitFeatureContext,
    outer_fold: int,
    pdf_names: np.ndarray,
    test_doc_id: int,
    score_threshold: float,
    feature_mode: str,
    rbf_gamma: float,
    top_n: int,
) -> list[dict]:
    threshold_bundle = context.get_threshold_bundle(score_threshold)
    selected_matrix = threshold_bundle["feature_matrices"][feature_mode]
    selected_term_count = int(threshold_bundle["selected_term_count"])

    test_row_index = int(np.flatnonzero(context.test_doc_ids == test_doc_id)[0])
    train_values = dense_matrix(selected_matrix["train"])
    test_values = dense_row_values(selected_matrix["test"], test_row_index)

    diff = train_values - test_values
    squared_linear_distances = np.sum(diff * diff, axis=1)
    linear_distances = np.sqrt(squared_linear_distances)

    rbf_kernel_values = np.exp(-rbf_gamma * squared_linear_distances)
    rbf_feature_distances = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * rbf_kernel_values))

    rows = []
    for method, distances in (
        ("linear_feature_space", linear_distances),
        ("rbf_feature_space", rbf_feature_distances),
    ):
        nearest_order = np.argsort(distances, kind="mergesort")[:top_n]
        for rank, train_row_index in enumerate(nearest_order, start=1):
            train_doc_id = int(context.train_doc_ids[int(train_row_index)])
            rows.append(
                {
                    "method": method,
                    "rank": rank,
                    "outer_fold": outer_fold,
                    "score_threshold": score_threshold,
                    "feature_mode": feature_mode,
                    "selected_term_count": selected_term_count,
                    "rbf_gamma": rbf_gamma if method == "rbf_feature_space" else "",
                    "test_document_id": int(test_doc_id),
                    "test_pdf_name": str(pdf_names[test_doc_id]),
                    "test_label": int(context.test_labels[test_row_index]),
                    "train_document_id": train_doc_id,
                    "train_pdf_name": str(pdf_names[train_doc_id]),
                    "train_label": int(context.train_labels[int(train_row_index)]),
                    "distance": float(distances[int(train_row_index)]),
                    "linear_squared_distance": float(squared_linear_distances[int(train_row_index)]),
                }
            )
    return rows


def save_nearest_pdf_rows(rows: list[dict], output_path: Path) -> pd.DataFrame:
    output_path = output_path if output_path.is_absolute() else PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_path.resolve(), index=False, encoding="utf-8-sig")
    return result_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Find training PDFs nearest to a requested outer-test PDF in linear feature space "
            "and RBF kernel feature space, without running SVM evaluation."
        )
    )
    parser.add_argument("test_pdf", nargs="?", help="PDF name or number that must be in the outer test data.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help=f"Score threshold. Default: {DEFAULT_SCORE_THRESHOLD}",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["count", "binary"],
        default=DEFAULT_FEATURE_MODE,
        help=f"Feature value type. Default: {DEFAULT_FEATURE_MODE}",
    )
    parser.add_argument(
        "--rbf-gamma",
        type=float,
        default=DEFAULT_RBF_GAMMA,
        help=f"RBF gamma used for kernel-space distance. Default: {DEFAULT_RBF_GAMMA}",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of nearest training PDFs to output for each method. Default: {DEFAULT_TOP_N}",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output CSV path.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    test_pdf = args.test_pdf if args.test_pdf else DEFAULT_TEST_PDF
    if not test_pdf:
        parser.error("Specify TEST_PDF or set DEFAULT_TEST_PDF.")
    if args.rbf_gamma <= 0:
        parser.error("--rbf-gamma must be positive.")
    if args.top_n <= 0:
        parser.error("--top-n must be positive.")

    documents, labels, doc_ids, pdf_names = load_corpus()
    test_doc_id = find_pdf_id(pdf_names, test_pdf)
    outer_fold, context = find_outer_context_for_test_pdf(documents, labels, doc_ids, test_doc_id)
    rows = compute_nearest_rows(
        context=context,
        outer_fold=outer_fold,
        pdf_names=pdf_names,
        test_doc_id=test_doc_id,
        score_threshold=args.threshold,
        feature_mode=args.feature_mode,
        rbf_gamma=args.rbf_gamma,
        top_n=args.top_n,
    )
    result_df = save_nearest_pdf_rows(rows, Path(args.output))

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    print(f"saved: {output_path.resolve()}")
    print(f"test PDF: {normalize_pdf_name(test_pdf)}")
    print(f"outer fold: {outer_fold}")
    print(f"threshold: {args.threshold}")
    print(f"feature mode: {args.feature_mode}")
    print(f"rbf gamma: {args.rbf_gamma}")
    for method in ("linear_feature_space", "rbf_feature_space"):
        top_row = result_df[result_df["method"] == method].sort_values("rank").iloc[0]
        print(
            f"{method} nearest: "
            f"{top_row['train_pdf_name']} "
            f"(distance={top_row['distance']:.6f}, label={int(top_row['train_label'])})"
        )
    if not math.isclose(float(result_df[result_df["method"] == "linear_feature_space"].iloc[0]["linear_squared_distance"]), 0.0):
        print("note: for an RBF kernel, nearest ranking is monotonic with the original squared Euclidean distance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
