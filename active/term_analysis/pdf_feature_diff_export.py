from __future__ import annotations

import argparse
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


OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "ta" / "pdf_feature_diff"

# 引数なしで実行する場合はここを書き換えてください。
DEFAULT_TRAIN_PDF = "104_0.pdf"
DEFAULT_TEST_PDF = "105_0.pdf"
DEFAULT_SCORE_THRESHOLD = 0.11
DEFAULT_FEATURE_MODE = "binary"  # "count" or "binary"
DEFAULT_OUTPUT = str(OUTPUT_DIR / "pdf_feature_values.csv")


def normalize_pdf_name(value: str) -> str:
    return Path(value.strip()).name


def find_pdf_id(pdf_names: np.ndarray, pdf_name: str) -> int:
    target = normalize_pdf_name(pdf_name)
    matches = np.flatnonzero(pdf_names == target)
    if len(matches) == 1:
        return int(matches[0])
    if len(matches) > 1:
        raise ValueError(f"PDF name is not unique: {target}")

    partial_matches = [str(name) for name in pdf_names if target in str(name)]
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


def is_requested_train_test_split(
    train_doc_id: int,
    test_doc_id: int,
    train_index: np.ndarray,
    test_index: np.ndarray,
) -> bool:
    train_ids = set(map(int, train_index))
    test_ids = set(map(int, test_index))
    return train_doc_id in train_ids and test_doc_id in test_ids


def dense_row_values(matrix, row_index: int) -> np.ndarray:
    row = matrix[row_index]
    if hasattr(row, "toarray"):
        return np.asarray(row.toarray()).ravel()
    return np.asarray(row).ravel()


def build_feature_value_rows(
    context: SplitFeatureContext,
    score_threshold: float,
    feature_mode: str,
    train_doc_id: int,
    test_doc_id: int,
    pdf_names: np.ndarray,
) -> list[dict]:
    train_row_index = int(np.flatnonzero(context.train_doc_ids == train_doc_id)[0])
    test_row_index = int(np.flatnonzero(context.test_doc_ids == test_doc_id)[0])

    threshold_bundle = context.get_threshold_bundle(score_threshold)
    selected_terms_df = threshold_bundle["selected_terms_df"].reset_index(drop=True)
    selected_matrix = threshold_bundle["feature_matrices"][feature_mode]

    train_values = dense_row_values(selected_matrix["train"], train_row_index)
    test_values = dense_row_values(selected_matrix["test"], test_row_index)
    train_pdf_name = str(pdf_names[train_doc_id])
    test_pdf_name = str(pdf_names[test_doc_id])

    rows = []
    for feature_index, term_row in selected_terms_df.iterrows():
        rows.append(
            {
                "単語": term_row["term"],
                train_pdf_name: float(train_values[feature_index]),
                test_pdf_name: float(test_values[feature_index]),
            }
        )
    return rows


def save_feature_values(
    documents: np.ndarray,
    labels: np.ndarray,
    doc_ids: np.ndarray,
    pdf_names: np.ndarray,
    train_doc_id: int,
    test_doc_id: int,
    score_threshold: float,
    feature_mode: str,
    output_path: Path,
) -> pd.DataFrame:
    output_path = output_path if output_path.is_absolute() else PROJECT_ROOT / output_path
    matched_fold = None
    rows = None

    for outer_fold, train_index, test_index in iter_outer_splits(documents, labels):
        if not is_requested_train_test_split(train_doc_id, test_doc_id, train_index, test_index):
            continue
        if matched_fold is not None:
            raise RuntimeError(f"Expected one matching outer fold, but found at least two: {matched_fold}, {outer_fold}")

        matched_fold = outer_fold
        context = SplitFeatureContext(
            documents[train_index],
            labels[train_index],
            documents[test_index],
            labels[test_index],
            doc_ids[train_index],
            doc_ids[test_index],
        )
        rows = build_feature_value_rows(
            context,
            score_threshold,
            feature_mode,
            train_doc_id,
            test_doc_id,
            pdf_names,
        )

    if matched_fold is None or rows is None:
        raise RuntimeError("No outer fold matched the requested train/test PDF condition.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_path.resolve(), index=False, encoding="utf-8-sig")
    result_df.attrs["matched_fold"] = matched_fold
    return result_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Find the outer 10-fold split where TRAIN_PDF is in train and TEST_PDF is in test, "
            "then export selected feature values without running SVM."
        )
    )
    parser.add_argument("train_pdf", nargs="?", help="PDF name that must be in the outer training data.")
    parser.add_argument("test_pdf", nargs="?", help="PDF name that must be in the outer test data.")
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
        help=f"Feature value type to compare. Default: {DEFAULT_FEATURE_MODE}",
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

    if args.train_pdf and args.test_pdf:
        train_pdf = args.train_pdf
        test_pdf = args.test_pdf
    elif not args.train_pdf and not args.test_pdf and DEFAULT_TRAIN_PDF and DEFAULT_TEST_PDF:
        train_pdf = DEFAULT_TRAIN_PDF
        test_pdf = DEFAULT_TEST_PDF
    else:
        parser.error("Specify TRAIN_PDF TEST_PDF, or set DEFAULT_TRAIN_PDF and DEFAULT_TEST_PDF.")

    documents, labels, doc_ids, pdf_names = load_corpus()
    train_doc_id = find_pdf_id(pdf_names, train_pdf)
    test_doc_id = find_pdf_id(pdf_names, test_pdf)

    result_df = save_feature_values(
        documents=documents,
        labels=labels,
        doc_ids=doc_ids,
        pdf_names=pdf_names,
        train_doc_id=train_doc_id,
        test_doc_id=test_doc_id,
        score_threshold=args.threshold,
        feature_mode=args.feature_mode,
        output_path=Path(args.output),
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    print(f"saved: {output_path.resolve()}")
    print(f"rows: {len(result_df)}")
    print(f"outer fold: {result_df.attrs['matched_fold']}")
    print(f"train PDF: {normalize_pdf_name(train_pdf)}")
    print(f"test PDF : {normalize_pdf_name(test_pdf)}")
    print(f"threshold: {args.threshold}")
    print(f"feature mode: {args.feature_mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
