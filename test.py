from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def _bootstrap_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "active").exists() and (candidate / "archive").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise RuntimeError("Project root could not be found.")


PROJECT_ROOT = _bootstrap_project_root()

from project_runtime import bootstrap_project_paths  # noqa: E402

bootstrap_project_paths(PROJECT_ROOT)

from dataset_loader import load_document_filenames, load_documents  # noqa: E402
from text_vectorizer import Tf_idf  # noqa: E402


# 必要に応じてここを書き換えてください。
SPECIFIED_TERMS = ["partner", "root", "deleg", "tree", "determinist", "pair", "sale", "legal", "merkl", "signatur", "paramet", "person", "websit", "presal"]
REAL_SCAM = False
KEEP_DIGITS = False
PERCENT_MODE = "drop"
# 実行時間優先のテストなので False にしています。既存実験に近づけたい場合は True に変更してください。
USE_STEMMING = False
MIN_LEN = 0
N_SPLITS = 10
RANDOM_STATE = 42
CLASS_WEIGHT = "balanced"
TOL = 1e-3

# 実行時間優先のため、候補は少なめにしています。
KERNELS = ["linear", "rbf"]
LINEAR_C_VALUES = [0.01, 0.1, 1, 10, 100]
RBF_C_VALUES = [ 0.01, 0.1, 1, 10, 100]
RBF_GAMMA_VALUES = [0.001, 0.01, 0.1, 1, 10, 'scale', 'auto']

OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "ta" / "specified_word_word_count_svm_test"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "summary.csv"
FOLD_OUTPUT_PATH = OUTPUT_DIR / "outer_fold_results.csv"
FEATURE_OUTPUT_PATH = OUTPUT_DIR / "document_features.csv"


def log_progress(message: str) -> None:
    print(message, flush=True)


def load_corpus() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label1_documents, label0_documents = load_documents(
        real_scam=REAL_SCAM,
        keep_digits=KEEP_DIGITS,
        percent_mode=PERCENT_MODE,
    )
    label1_names, label0_names = load_document_filenames(real_scam=REAL_SCAM, sort_files=False)

    documents = np.array(label1_documents + label0_documents, dtype=object)
    labels = np.array([1] * len(label1_documents) + [0] * len(label0_documents), dtype=int)
    pdf_names = np.array(label1_names + label0_names, dtype=object)

    if len(documents) != len(pdf_names):
        raise RuntimeError(
            "document count and filename count do not match: "
            f"documents={len(documents)}, pdf_names={len(pdf_names)}"
        )

    return documents, labels, pdf_names


def preprocess_documents(documents: np.ndarray) -> list[str]:
    return Tf_idf(list(documents), True, MIN_LEN, use_stemming=USE_STEMMING).processed_docs


def normalize_terms(terms: list[str]) -> list[dict[str, object]]:
    normalized_terms: list[dict[str, object]] = []
    for original_term in terms:
        processed_term = Tf_idf([original_term], True, MIN_LEN, use_stemming=USE_STEMMING).processed_docs[0]
        tokens = processed_term.split()
        if not tokens:
            log_progress(f"[warning] skipped empty term after preprocessing: {original_term}")
            continue
        normalized_terms.append(
            {
                "original": original_term,
                "processed": " ".join(tokens),
                "tokens": tuple(tokens),
            }
        )

    if not normalized_terms:
        raise ValueError("No usable specified terms remain after preprocessing.")

    return normalized_terms


def contains_term(document_tokens: list[str], term_tokens: tuple[str, ...]) -> int:
    if len(term_tokens) == 1:
        return int(term_tokens[0] in set(document_tokens))

    term_length = len(term_tokens)
    for start in range(0, len(document_tokens) - term_length + 1):
        if tuple(document_tokens[start : start + term_length]) == term_tokens:
            return 1
    return 0


def build_feature_matrix(
    documents: np.ndarray,
    processed_documents: list[str],
    normalized_terms: list[dict[str, object]],
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    document_tokens = [processed_document.split() for processed_document in processed_documents]
    document_token_sets = [set(tokens) for tokens in document_tokens]

    binary_columns = []
    feature_names = []
    for term in normalized_terms:
        term_tokens = term["tokens"]
        if len(term_tokens) == 1:
            values = [int(term_tokens[0] in token_set) for token_set in document_token_sets]
        else:
            values = [contains_term(tokens, term_tokens) for tokens in document_tokens]
        binary_columns.append(values)
        feature_names.append(f"has_{term['processed']}")

    word_counts = np.array([len(str(document).split()) for document in documents], dtype=float).reshape(-1, 1)
    scaled_word_counts = StandardScaler().fit_transform(word_counts).ravel()

    X_binary = np.array(binary_columns, dtype=float).T
    X = np.column_stack([X_binary, scaled_word_counts])
    feature_names.append("word_count_scaled")

    feature_df = pd.DataFrame(X, columns=feature_names)
    feature_df.insert(0, "word_count", word_counts.ravel().astype(int))

    return X, feature_names, feature_df


def parameter_grid(kernel: str) -> list[dict[str, object]]:
    if kernel == "linear":
        return [{"kernel": "linear", "C": c} for c in LINEAR_C_VALUES]
    if kernel == "rbf":
        return [
            {"kernel": "rbf", "C": c, "gamma": gamma}
            for c, gamma in itertools.product(RBF_C_VALUES, RBF_GAMMA_VALUES)
        ]
    raise ValueError(f"Unsupported kernel: {kernel}")


def build_model(params: dict[str, object]) -> svm.SVC:
    return svm.SVC(
        kernel=params["kernel"],
        C=params["C"],
        gamma=params.get("gamma", "scale"),
        class_weight=CLASS_WEIGHT,
        tol=TOL,
    )


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def select_best_params_by_inner_cv(
    X_outer_train: np.ndarray,
    y_outer_train: np.ndarray,
    kernel: str,
) -> tuple[dict[str, object], float]:
    inner_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    candidates = parameter_grid(kernel)
    candidate_scores = []

    for candidate_index, params in enumerate(candidates):
        fold_scores = []
        for inner_train_index, inner_valid_index in inner_cv.split(X_outer_train, y_outer_train):
            model = build_model(params)
            model.fit(X_outer_train[inner_train_index], y_outer_train[inner_train_index])
            y_valid_pred = model.predict(X_outer_train[inner_valid_index])
            fold_scores.append(accuracy_score(y_outer_train[inner_valid_index], y_valid_pred))

        candidate_scores.append(
            {
                "candidate_index": candidate_index,
                "params": params,
                "mean_inner_accuracy": float(np.mean(fold_scores)),
            }
        )

    candidate_scores.sort(
        key=lambda row: (
            row["mean_inner_accuracy"],
            -row["candidate_index"],
        ),
        reverse=True,
    )
    best = candidate_scores[0]
    return best["params"], best["mean_inner_accuracy"]


def run_nested_cv(X: np.ndarray, labels: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    outer_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []

    for kernel in KERNELS:
        log_progress(f"[nested cv] kernel={kernel}")
        for outer_fold, (train_index, test_index) in enumerate(outer_cv.split(X, labels), start=1):
            start = time.perf_counter()
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            best_params, best_inner_accuracy = select_best_params_by_inner_cv(X_train, y_train, kernel)
            model = build_model(best_params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = score_predictions(y_test, y_pred)

            row = {
                "kernel": kernel,
                "outer_fold": outer_fold,
                "best_params": repr(best_params),
                "best_inner_accuracy": best_inner_accuracy,
                "test_size": int(len(test_index)),
                "elapsed_seconds": round(time.perf_counter() - start, 3),
                **metrics,
            }
            fold_rows.append(row)
            log_progress(
                f"[nested cv] kernel={kernel} outer_fold={outer_fold}/{N_SPLITS} "
                f"accuracy={metrics['accuracy']:.4f} best={best_params}"
            )

    fold_df = pd.DataFrame(fold_rows)
    summary_rows = []
    for kernel, group in fold_df.groupby("kernel", sort=False):
        summary_rows.append(
            {
                "kernel": kernel,
                "specified_terms": ",".join(SPECIFIED_TERMS),
                "feature_count": int(X.shape[1]),
                "n_documents": int(len(labels)),
                "n_label1": int(np.sum(labels == 1)),
                "n_label0": int(np.sum(labels == 0)),
                "mean_accuracy": float(group["accuracy"].mean()),
                "std_accuracy": float(group["accuracy"].std(ddof=0)),
                "mean_recall": float(group["recall"].mean()),
                "mean_precision": float(group["precision"].mean()),
                "mean_f1": float(group["f1"].mean()),
                "best_params_by_fold": "|".join(group["best_params"].astype(str)),
            }
        )

    return pd.DataFrame(summary_rows), fold_df


def save_outputs(
    summary_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    labels: np.ndarray,
    pdf_names: np.ndarray,
    normalized_terms: list[dict[str, object]],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_feature_df = feature_df.copy()
    output_feature_df.insert(0, "label", labels)
    output_feature_df.insert(0, "pdf_name", pdf_names)

    normalized_term_columns = [
        {
            "original_term": term["original"],
            "processed_term": term["processed"],
        }
        for term in normalized_terms
    ]
    term_df = pd.DataFrame(normalized_term_columns)

    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    fold_df.to_csv(FOLD_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    output_feature_df.to_csv(FEATURE_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    term_df.to_csv(OUTPUT_DIR / "specified_terms.csv", index=False, encoding="utf-8-sig")


def main() -> int:
    log_progress("[main] loading corpus")
    documents, labels, pdf_names = load_corpus()

    log_progress("[main] building specified-word binary features and normalized word-count feature")
    processed_documents = preprocess_documents(documents)
    normalized_terms = normalize_terms(SPECIFIED_TERMS)
    X, feature_names, feature_df = build_feature_matrix(documents, processed_documents, normalized_terms)

    log_progress(f"[main] documents={len(labels)}, features={feature_names}")
    log_progress(f"[main] label1={int(np.sum(labels == 1))}, label0={int(np.sum(labels == 0))}")

    summary_df, fold_df = run_nested_cv(X, labels)
    save_outputs(summary_df, fold_df, feature_df, labels, pdf_names, normalized_terms)

    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nsaved summary: {SUMMARY_OUTPUT_PATH}")
    print(f"saved fold results: {FOLD_OUTPUT_PATH}")
    print(f"saved document features: {FEATURE_OUTPUT_PATH}")
    print(f"saved specified terms: {OUTPUT_DIR / 'specified_terms.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
