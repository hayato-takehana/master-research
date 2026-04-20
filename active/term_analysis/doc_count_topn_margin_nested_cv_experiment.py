from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
import os
import sys
import time


def _bootstrap_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "active").exists() and (candidate / "archive").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise RuntimeError("Project root could not be found.")


PROJECT_ROOT = _bootstrap_project_root()

from project_runtime import bootstrap_project_paths, get_output_dir, redirect_relative_outputs

bootstrap_project_paths(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
SAVE_DIR = get_output_dir(__file__, PROJECT_ROOT)
redirect_relative_outputs(SAVE_DIR)

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from dataset_loader import load_documents
from text_vectorizer import Tf_idf


TOP_N_VALUES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
MIN_LEN = 0
MIN_DF = 0.0
USE_STEMMING = True
KEEP_DIGITS = False
PERCENT_MODE = "drop"

LINEAR_C_VALUES = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
RBF_C_VALUES = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
RBF_GAMMA_VALUES = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
MARGIN_THRESHOLD = 0.9990
N_SPLITS = 10
RANDOM_STATE = 42
CLASS_WEIGHT = "balanced"
TOL = 1e-10
CANDIDATE_PROGRESS_INTERVAL = 10

FEATURE_MODE_DIR_NAMES = {
    "binary": "binary",
    "count": "count",
}
KERNEL_DIR_NAMES = {
    "linear": "linear_svm",
    "rbf": "nonlinear_svm",
}
METRICS_DIR_NAME = "metrics"
SELECTED_TERMS_DIR_NAME = "selected_terms"
FEATURE_COUNT_SUMMARY_DIR_NAME = "feature_count_summary"
BEST_RESULT_COLUMNS = [
    "feature_count",
    "feature_mode",
    "kernel",
    "mean_selected_term_count",
    "mean_accuracy",
    "mean_recall",
    "mean_precision",
    "mean_f1",
    "best_cs",
    "best_gammas",
]


def prompt_choice(prompt_label, option_lines, alias_to_value, default_input):
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


def prompt_experiment_configuration():
    kernel_mode = prompt_choice(
        "\n1. 実行するSVMを選択してください。",
        [
            "  1: 線形SVMのみ",
            "  2: RBF非線形SVMのみ",
            "  3: 両方",
        ],
        {
            "1": "linear",
            "linear": "linear",
            "lin": "linear",
            "2": "rbf",
            "rbf": "rbf",
            "nonlinear": "rbf",
            "3": "both",
            "both": "both",
            "all": "both",
        },
        default_input="3",
    )

    feature_mode = prompt_choice(
        "\n2. 使う特徴量を選択してください。",
        [
            "  1: 出現頻度のみ",
            "  2: バイナリー変数のみ",
            "  3: 両方",
        ],
        {
            "1": "count",
            "count": "count",
            "frequency": "count",
            "2": "binary",
            "binary": "binary",
            "bin": "binary",
            "3": "both",
            "both": "both",
            "all": "both",
        },
        default_input="3",
    )

    return {
        "kernel_mode": kernel_mode,
        "feature_mode": feature_mode,
    }


def print_experiment_configuration(config):
    kernel_label_map = {
        "linear": "線形SVMのみ",
        "rbf": "RBF非線形SVMのみ",
        "both": "線形SVM + RBF非線形SVM",
    }
    feature_label_map = {
        "count": "出現頻度のみ",
        "binary": "バイナリー変数のみ",
        "both": "出現頻度 + バイナリー変数",
    }

    log_progress("\n[config] selected runtime options")
    log_progress(f"[config] svm mode     : {kernel_label_map[config['kernel_mode']]}")
    log_progress(f"[config] feature mode : {feature_label_map[config['feature_mode']]}")
    log_progress("[config] top_n mode    : fixed feature-count experiments only")


def load_corpus():
    documents_1, documents_0 = load_documents(
        real_scam=False,
        keep_digits=KEEP_DIGITS,
        percent_mode=PERCENT_MODE,
    )
    documents = np.array(documents_1 + documents_0, dtype=object)
    labels = np.array([1] * len(documents_1) + [0] * len(documents_0), dtype=int)
    return documents, labels


def log_progress(message):
    print(message, flush=True)


def format_elapsed_seconds(elapsed_seconds):
    elapsed_seconds = int(round(elapsed_seconds))
    minutes, seconds = divmod(elapsed_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def vectorize_train_test_documents(train_docs, test_docs):
    train_vectorizer = Tf_idf(list(train_docs), True, MIN_LEN, use_stemming=USE_STEMMING)
    X_train_count, feature_names, fitted_count_vectorizer = train_vectorizer.term_frequency(
        MIN_DF,
        ngram_range=(1, 1),
    )

    test_vectorizer = Tf_idf(list(test_docs), True, MIN_LEN, use_stemming=USE_STEMMING)
    X_test_count = fitted_count_vectorizer.transform(test_vectorizer.processed_docs)

    return X_train_count.tocsr(), X_test_count.tocsr(), np.array(feature_names)


def _nonzero_doc_counts(X):
    if hasattr(X, "getnnz"):
        return np.asarray(X.getnnz(axis=0)).ravel()
    return np.asarray(np.sum(X != 0, axis=0)).ravel()


def build_doc_count_score_table(X, labels, feature_names):
    mask1 = labels == 1
    mask0 = labels == 0
    X_label1 = X[mask1]
    X_label0 = X[mask0]
    n_label1 = X_label1.shape[0]
    n_label0 = X_label0.shape[0]

    label1_doc_count = _nonzero_doc_counts(X_label1)
    label0_doc_count = _nonzero_doc_counts(X_label0)

    score_label1 = (label1_doc_count / n_label1) - (label0_doc_count / n_label0)
    score_label0 = (label0_doc_count / n_label0) - (label1_doc_count / n_label1)

    return pd.DataFrame(
        {
            "term": feature_names,
            "label1_doc_count": label1_doc_count,
            "label0_doc_count": label0_doc_count,
            "label1_total_docs": n_label1,
            "label0_total_docs": n_label0,
            "label1_score": score_label1,
            "label0_score": score_label0,
            "doc_count_diff_label1_minus_label0": label1_doc_count - label0_doc_count,
            "doc_count_diff_label0_minus_label1": label0_doc_count - label1_doc_count,
        }
    )


def select_top_terms(score_df, top_n):
    if top_n % 2 != 0:
        raise ValueError("TOP_N は偶数を指定してください。ラベル1側とラベル0側で等分します。")

    top_n_per_label = top_n // 2

    label1_top = (
        score_df.sort_values(
            by=["label1_score", "label1_doc_count", "label0_doc_count", "term"],
            ascending=[False, False, True, True],
        )
        .head(top_n_per_label)
        .reset_index(drop=True)
    )
    label0_top = (
        score_df.sort_values(
            by=["label0_score", "label0_doc_count", "label1_doc_count", "term"],
            ascending=[False, False, True, True],
        )
        .head(top_n_per_label)
        .reset_index(drop=True)
    )
    return label1_top, label0_top


def build_selected_feature_matrices(X_train_count, X_test_count, feature_names, label1_top, label0_top):
    selected_terms = list(label1_top["term"]) + list(label0_top["term"])
    selected_terms_unique = list(dict.fromkeys(selected_terms))
    vocab_index = {term: idx for idx, term in enumerate(feature_names)}
    selected_indices = [vocab_index[term] for term in selected_terms_unique]

    X_train_selected = X_train_count[:, selected_indices]
    X_test_selected = X_test_count[:, selected_indices]

    return {
        "selected_terms": selected_terms_unique,
        "count": {
            "train": X_train_selected,
            "test": X_test_selected,
        },
        "binary": {
            "train": (X_train_selected != 0).astype(np.int8),
            "test": (X_test_selected != 0).astype(np.int8),
        },
    }


class SplitFeatureContext:
    def __init__(self, train_docs, train_labels, test_docs, test_labels):
        self.train_docs = np.array(train_docs, dtype=object)
        self.train_labels = np.array(train_labels, dtype=int)
        self.test_docs = np.array(test_docs, dtype=object)
        self.test_labels = np.array(test_labels, dtype=int)
        self.X_train_count, self.X_test_count, self.feature_names = vectorize_train_test_documents(
            self.train_docs,
            self.test_docs,
        )
        self.score_df = build_doc_count_score_table(self.X_train_count, self.train_labels, self.feature_names)
        self.top_n_cache = {}

    def get_top_n_bundle(self, top_n):
        if top_n not in self.top_n_cache:
            label1_top, label0_top = select_top_terms(self.score_df, top_n)
            feature_matrices = build_selected_feature_matrices(
                self.X_train_count,
                self.X_test_count,
                self.feature_names,
                label1_top,
                label0_top,
            )
            self.top_n_cache[top_n] = {
                "feature_count": top_n,
                "label1_selected": label1_top,
                "label0_selected": label0_top,
                "selected_term_count": len(feature_matrices["selected_terms"]),
                "feature_matrices": feature_matrices,
            }
        return self.top_n_cache[top_n]


def build_outer_fold_contexts(documents, labels):
    outer_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    outer_fold_contexts = []
    build_start = time.perf_counter()

    log_progress(f"[split setup] building outer/inner feature contexts for {N_SPLITS}-fold nested CV")

    for outer_fold, (train_index, test_index) in enumerate(outer_cv.split(documents, labels), start=1):
        outer_fold_start = time.perf_counter()
        log_progress(f"[split setup] outer fold {outer_fold}/{N_SPLITS}: preparing train/test feature contexts")

        outer_train_docs = documents[train_index]
        outer_train_labels = labels[train_index]
        outer_test_docs = documents[test_index]
        outer_test_labels = labels[test_index]

        outer_context = SplitFeatureContext(
            outer_train_docs,
            outer_train_labels,
            outer_test_docs,
            outer_test_labels,
        )

        inner_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        inner_contexts = []
        for inner_train_index, inner_test_index in inner_cv.split(outer_train_docs, outer_train_labels):
            inner_contexts.append(
                SplitFeatureContext(
                    outer_train_docs[inner_train_index],
                    outer_train_labels[inner_train_index],
                    outer_train_docs[inner_test_index],
                    outer_train_labels[inner_test_index],
                )
            )

        outer_fold_contexts.append(
            {
                "outer_fold": outer_fold,
                "outer_context": outer_context,
                "inner_contexts": inner_contexts,
            }
        )
        log_progress(
            f"[split setup] outer fold {outer_fold}/{N_SPLITS} completed "
            f"({len(inner_contexts)} inner contexts, elapsed {format_elapsed_seconds(time.perf_counter() - outer_fold_start)})"
        )

    log_progress(
        f"[split setup] all feature contexts completed "
        f"(elapsed {format_elapsed_seconds(time.perf_counter() - build_start)})"
    )
    return outer_fold_contexts


def _build_svc(kernel, c, gamma=None):
    kwargs = {
        "kernel": kernel,
        "C": c,
        "class_weight": CLASS_WEIGHT,
        "tol": TOL,
    }
    if gamma is not None:
        kwargs["gamma"] = gamma
    return svm.SVC(**kwargs)


def _compute_margins(model, X_train, y_train):
    decision_values = model.decision_function(X_train)
    y_signed = np.where(y_train == 0, -1, 1)
    return y_signed * decision_values


def format_candidate_label(feature_count, c, gamma=None):
    if gamma is None:
        return f"top={feature_count:05d}, C={c}"
    return f"top={feature_count:05d}, C={c}, gamma={gamma}"


def fit_and_evaluate_split_context(split_context, feature_count, feature_mode, kernel, c, gamma=None):
    feature_bundle = split_context.get_top_n_bundle(feature_count)
    selected_term_count = feature_bundle["selected_term_count"]
    if selected_term_count == 0:
        return None

    X_train = feature_bundle["feature_matrices"][feature_mode]["train"]
    X_test = feature_bundle["feature_matrices"][feature_mode]["test"]

    model = _build_svc(kernel, c, gamma=gamma)
    model.fit(X_train, split_context.train_labels)

    margins = _compute_margins(model, X_train, split_context.train_labels)
    if not np.all(margins >= MARGIN_THRESHOLD):
        return None

    y_pred = model.predict(X_test)
    return {
        "selected_term_count": selected_term_count,
        "accuracy": float(accuracy_score(split_context.test_labels, y_pred)),
        "recall": float(recall_score(split_context.test_labels, y_pred, zero_division=0)),
        "precision": float(precision_score(split_context.test_labels, y_pred, zero_division=0)),
        "f1_score": float(f1_score(split_context.test_labels, y_pred, zero_division=0)),
    }


def evaluate_candidate_across_outer_folds(outer_fold_contexts, feature_count, feature_mode, kernel, c, gamma=None):
    candidate_label = format_candidate_label(feature_count, c, gamma)
    fold_records = []

    for fold_bundle in outer_fold_contexts:
        inner_scores = []
        for inner_context in fold_bundle["inner_contexts"]:
            inner_result = fit_and_evaluate_split_context(
                inner_context,
                feature_count,
                feature_mode,
                kernel,
                c,
                gamma,
            )
            if inner_result is None:
                return None
            inner_scores.append(inner_result["accuracy"])

        outer_result = fit_and_evaluate_split_context(
            fold_bundle["outer_context"],
            feature_count,
            feature_mode,
            kernel,
            c,
            gamma,
        )
        if outer_result is None:
            return None

        fold_records.append(
            {
                "outer_fold": fold_bundle["outer_fold"],
                "feature_count": feature_count,
                "selected_term_count": outer_result["selected_term_count"],
                "c": c,
                "gamma": gamma,
                "param_label": candidate_label,
                "inner_score": float(sum(inner_scores) / len(inner_scores)),
                "accuracy": outer_result["accuracy"],
                "recall": outer_result["recall"],
                "precision": outer_result["precision"],
                "f1_score": outer_result["f1_score"],
            }
        )

    return fold_records


def round_half_up(value, digits=3):
    if value in ("", None):
        return ""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError):
        return value
    if not np.isfinite(numeric_value):
        return numeric_value
    quantize_exp = Decimal("1").scaleb(-digits)
    try:
        return float(Decimal(str(numeric_value)).quantize(quantize_exp, rounding=ROUND_HALF_UP))
    except InvalidOperation:
        return numeric_value


def build_empty_result(feature_count, feature_mode, kernel):
    result = {
        "feature_mode": feature_mode,
        "kernel": kernel,
        "feature_count": feature_count,
        "margin_threshold": MARGIN_THRESHOLD,
        "valid_param_labels": [],
        "dropped_param_labels": [],
        "selected_records": [],
        "best_cs": [],
        "selected_term_counts": [],
    }
    if kernel == "rbf":
        result["best_gammas"] = []
    return result


def run_nested_cv_for_fixed_feature_count(outer_fold_contexts, feature_count, feature_mode, kernel):
    c_values = LINEAR_C_VALUES if kernel == "linear" else RBF_C_VALUES
    gamma_candidates = [None] if kernel == "linear" else RBF_GAMMA_VALUES
    total_candidates = len(c_values) * len(gamma_candidates)
    run_start = time.perf_counter()
    processed_candidates = 0
    valid_candidate_results = {}
    dropped_param_labels = []

    log_progress(
        f"[condition start] {format_condition_label(feature_mode, kernel)} / top{feature_count}: "
        f"{total_candidates} candidate settings"
    )

    for c in c_values:
        for gamma in gamma_candidates:
            candidate_label = format_candidate_label(feature_count, c, gamma)
            candidate_records = evaluate_candidate_across_outer_folds(
                outer_fold_contexts,
                feature_count,
                feature_mode,
                kernel,
                c,
                gamma,
            )
            processed_candidates += 1
            if candidate_records is None:
                dropped_param_labels.append(candidate_label)
            else:
                valid_candidate_results[(c, gamma)] = candidate_records

            if (
                processed_candidates % CANDIDATE_PROGRESS_INTERVAL == 0
                or processed_candidates == total_candidates
            ):
                log_progress(
                    f"[condition progress] {format_condition_label(feature_mode, kernel)} / top{feature_count}: "
                    f"{processed_candidates}/{total_candidates} candidates processed "
                    f"(elapsed {format_elapsed_seconds(time.perf_counter() - run_start)})"
                )

    result = build_empty_result(feature_count, feature_mode, kernel)
    result["valid_param_labels"] = [
        format_candidate_label(feature_count, c, gamma)
        for c, gamma in valid_candidate_results.keys()
    ]
    result["dropped_param_labels"] = dropped_param_labels

    if not valid_candidate_results:
        return result

    selected_records = []
    for fold_idx in range(len(outer_fold_contexts)):
        best_record = None
        for records in valid_candidate_results.values():
            candidate = records[fold_idx]
            if best_record is None:
                best_record = candidate
                continue
            if candidate["inner_score"] > best_record["inner_score"]:
                best_record = candidate
                continue
            if (
                candidate["inner_score"] == best_record["inner_score"]
                and candidate["param_label"] < best_record["param_label"]
            ):
                best_record = candidate
        selected_records.append(best_record)

    result["selected_records"] = selected_records
    result["best_cs"] = [record["c"] for record in selected_records]
    if kernel == "rbf":
        result["best_gammas"] = [record["gamma"] for record in selected_records]
    result["selected_term_counts"] = [record["selected_term_count"] for record in selected_records]
    result["mean_selected_term_count"] = float(np.mean(result["selected_term_counts"]))
    result["mean_accuracy"] = float(np.mean([record["accuracy"] for record in selected_records]))
    result["mean_recall"] = float(np.mean([record["recall"] for record in selected_records]))
    result["mean_precision"] = float(np.mean([record["precision"] for record in selected_records]))
    result["mean_f1"] = float(np.mean([record["f1_score"] for record in selected_records]))
    return result


def save_nested_cv_result_csvs(result, output_dir, output_prefix):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"{output_prefix}_final_metrics.csv"
    selected_path = output_dir / f"{output_prefix}_selected_fold_results.csv"
    params_path = output_dir / f"{output_prefix}_param_status.csv"

    metrics_row = {
        "feature_mode": result["feature_mode"],
        "kernel": result["kernel"],
        "feature_count": result["feature_count"],
        "margin_threshold": result["margin_threshold"],
        "mean_selected_term_count": round_half_up(result.get("mean_selected_term_count", "")),
        "mean_accuracy": round_half_up(result.get("mean_accuracy", "")),
        "mean_recall": round_half_up(result.get("mean_recall", "")),
        "mean_precision": round_half_up(result.get("mean_precision", "")),
        "mean_f1": round_half_up(result.get("mean_f1", "")),
        "best_cs": ",".join(map(str, result.get("best_cs", []))),
        "best_gammas": ",".join(map(str, result.get("best_gammas", []))),
    }
    pd.DataFrame([metrics_row]).to_csv(metrics_path, index=False, encoding="utf-8-sig")

    selected_rows = []
    for record in result.get("selected_records", []):
        selected_rows.append(
            {
                "outer_fold": record["outer_fold"],
                "feature_count": record["feature_count"],
                "selected_term_count": record["selected_term_count"],
                "param_label": record["param_label"],
                "c": record["c"],
                "gamma": record["gamma"],
                "inner_score": round_half_up(record["inner_score"]),
                "accuracy": round_half_up(record["accuracy"]),
                "recall": round_half_up(record["recall"]),
                "precision": round_half_up(record["precision"]),
                "f1_score": round_half_up(record["f1_score"]),
            }
        )
    pd.DataFrame(
        selected_rows,
        columns=[
            "outer_fold",
            "feature_count",
            "selected_term_count",
            "param_label",
            "c",
            "gamma",
            "inner_score",
            "accuracy",
            "recall",
            "precision",
            "f1_score",
        ],
    ).to_csv(selected_path, index=False, encoding="utf-8-sig")

    param_rows = [{"status": "valid", "param_label": label} for label in result.get("valid_param_labels", [])]
    param_rows.extend(
        {"status": "dropped", "param_label": label} for label in result.get("dropped_param_labels", [])
    )
    pd.DataFrame(param_rows, columns=["status", "param_label"]).to_csv(params_path, index=False, encoding="utf-8-sig")

    result["saved_csv_paths"] = {
        "metrics": str(metrics_path),
        "selected_records": str(selected_path),
        "param_status": str(params_path),
    }


def save_outer_fold_selected_terms(result, outer_fold_contexts, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    for record in result.get("selected_records", []):
        outer_fold = record["outer_fold"]
        feature_count = record["feature_count"]
        feature_bundle = outer_fold_contexts[outer_fold - 1]["outer_context"].get_top_n_bundle(feature_count)

        file_path = output_dir / f"outer_fold{outer_fold:02d}_top{feature_count}_selected_terms.csv"
        pd.DataFrame({"term": feature_bundle["feature_matrices"]["selected_terms"]}).to_csv(
            file_path,
            index=False,
            encoding="utf-8-sig",
        )
        index_rows.append(
            {
                "outer_fold": outer_fold,
                "feature_count": feature_count,
                "selected_term_count": feature_bundle["selected_term_count"],
                "file_path": str(file_path),
            }
        )

    index_path = output_dir / "selected_terms_index.csv"
    pd.DataFrame(
        index_rows,
        columns=["outer_fold", "feature_count", "selected_term_count", "file_path"],
    ).to_csv(index_path, index=False, encoding="utf-8-sig")
    return {
        "selected_terms_index": str(index_path),
    }


def save_feature_count_term_count_outputs(outer_fold_contexts, feature_count_candidates, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    for fold_bundle in outer_fold_contexts:
        outer_fold = fold_bundle["outer_fold"]
        outer_context = fold_bundle["outer_context"]
        for feature_count in feature_count_candidates:
            feature_bundle = outer_context.get_top_n_bundle(feature_count)
            fold_rows.append(
                {
                    "outer_fold": outer_fold,
                    "feature_count": feature_count,
                    "selected_term_count": feature_bundle["selected_term_count"],
                    "label1_selected_term_count": len(feature_bundle["label1_selected"]),
                    "label0_selected_term_count": len(feature_bundle["label0_selected"]),
                }
            )

    fold_counts_path = output_dir / "outer_fold_feature_counts.csv"
    pd.DataFrame(fold_rows).to_csv(fold_counts_path, index=False, encoding="utf-8-sig")

    summary_rows = []
    for feature_count in feature_count_candidates:
        feature_rows = [row for row in fold_rows if row["feature_count"] == feature_count]
        summary_rows.append(
            {
                "feature_count": feature_count,
                "mean_selected_term_count": round_half_up(np.mean([row["selected_term_count"] for row in feature_rows])),
                "mean_label1_selected_term_count": round_half_up(
                    np.mean([row["label1_selected_term_count"] for row in feature_rows])
                ),
                "mean_label0_selected_term_count": round_half_up(
                    np.mean([row["label0_selected_term_count"] for row in feature_rows])
                ),
                "selected_term_counts": ",".join(str(row["selected_term_count"]) for row in feature_rows),
                "label1_selected_term_counts": ",".join(str(row["label1_selected_term_count"]) for row in feature_rows),
                "label0_selected_term_counts": ",".join(str(row["label0_selected_term_count"]) for row in feature_rows),
            }
        )

    summary_path = output_dir / "feature_count_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    return {
        "outer_fold_feature_counts": str(fold_counts_path),
        "feature_count_summary": str(summary_path),
    }


def build_summary_row(result):
    row = {
        "feature_count": result["feature_count"],
        "feature_mode": result["feature_mode"],
        "kernel": result["kernel"],
        "mean_selected_term_count": "",
        "mean_accuracy": "",
        "mean_recall": "",
        "mean_precision": "",
        "mean_f1": "",
        "best_cs": "",
        "best_gammas": "",
        "valid_param_count": len(result.get("valid_param_labels", [])),
        "dropped_param_count": len(result.get("dropped_param_labels", [])),
    }

    if result.get("selected_records"):
        row.update(
            {
                "mean_selected_term_count": round_half_up(result["mean_selected_term_count"]),
                "mean_accuracy": round_half_up(result["mean_accuracy"]),
                "mean_recall": round_half_up(result["mean_recall"]),
                "mean_precision": round_half_up(result["mean_precision"]),
                "mean_f1": round_half_up(result["mean_f1"]),
                "best_cs": ",".join(map(str, result.get("best_cs", []))),
                "best_gammas": ",".join(map(str, result.get("best_gammas", []))),
            }
        )

    return row


def get_condition_output_dir(output_root, feature_mode, kernel):
    condition_dir = output_root / FEATURE_MODE_DIR_NAMES[feature_mode] / KERNEL_DIR_NAMES[kernel]
    condition_dir.mkdir(parents=True, exist_ok=True)
    return condition_dir


def get_active_conditions(kernel_mode, feature_mode):
    feature_modes = ["binary", "count"] if feature_mode == "both" else [feature_mode]
    kernels = ["linear", "rbf"] if kernel_mode == "both" else [kernel_mode]

    conditions = []
    for kernel in kernels:
        for current_feature_mode in feature_modes:
            conditions.append((current_feature_mode, kernel))
    return conditions


def save_global_summary_outputs(summary_rows_by_condition, best_results_by_condition, output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for feature_mode, kernel in summary_rows_by_condition.keys():
        condition_dir = get_condition_output_dir(output_root, feature_mode, kernel)
        summary_path = condition_dir / "svm_feature_count_summary.csv"
        best_path = condition_dir / "svm_best_result.csv"

        pd.DataFrame(summary_rows_by_condition[(feature_mode, kernel)]).to_csv(
            summary_path,
            index=False,
            encoding="utf-8-sig",
        )

        best_row = best_results_by_condition[(feature_mode, kernel)]
        if best_row is None:
            pd.DataFrame(columns=BEST_RESULT_COLUMNS).to_csv(best_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame([{column: best_row.get(column, "") for column in BEST_RESULT_COLUMNS}]).to_csv(
                best_path,
                index=False,
                encoding="utf-8-sig",
            )


def format_condition_label(feature_mode, kernel):
    feature_mode_label = "バイナリ特徴" if feature_mode == "binary" else "出現頻度特徴"
    kernel_label = "線形SVM" if kernel == "linear" else "非線形SVM"
    return f"{feature_mode_label} / {kernel_label}"


def print_result(result, heading):
    print(f"\n{heading}")
    if not result.get("selected_records"):
        print(f"margin >= {result['margin_threshold']} を全foldで満たす組み合わせがありませんでした。")
        return

    print(f"accuracy : {result['mean_accuracy']:.4f}")
    print(f"recall   : {result['mean_recall']:.4f}")
    print(f"precision: {result['mean_precision']:.4f}")
    print(f"f1       : {result['mean_f1']:.4f}")
    print(f"selected term counts: {result['selected_term_counts']}")
    print(f"selected C values: {result['best_cs']}")
    if "best_gammas" in result:
        print(f"selected gamma values: {result['best_gammas']}")
    print(f"globally valid params: {result['valid_param_labels']}")
    if "saved_csv_paths" in result:
        print(f"saved metrics csv: {result['saved_csv_paths']['metrics']}")
        print(f"saved selected-records csv: {result['saved_csv_paths']['selected_records']}")
        print(f"saved param-status csv: {result['saved_csv_paths']['param_status']}")


if __name__ == "__main__":
    runtime_config = prompt_experiment_configuration()
    print_experiment_configuration(runtime_config)

    documents, labels = load_corpus()
    active_conditions = get_active_conditions(
        runtime_config["kernel_mode"],
        runtime_config["feature_mode"],
    )
    outer_fold_contexts = build_outer_fold_contexts(documents, labels)

    feature_count_paths = save_feature_count_term_count_outputs(
        outer_fold_contexts,
        TOP_N_VALUES,
        SAVE_DIR / FEATURE_COUNT_SUMMARY_DIR_NAME,
    )
    log_progress(f"saved feature-count csv: {feature_count_paths['outer_fold_feature_counts']}")
    log_progress(f"saved feature-count summary csv: {feature_count_paths['feature_count_summary']}")

    summary_rows_by_condition = {condition: [] for condition in active_conditions}
    best_results_by_condition = {condition: None for condition in active_conditions}

    for feature_count in TOP_N_VALUES:
        feature_root = SAVE_DIR / f"top{feature_count}"

        for feature_mode, kernel in active_conditions:
            result = run_nested_cv_for_fixed_feature_count(
                outer_fold_contexts,
                feature_count,
                feature_mode,
                kernel,
            )

            metrics_dir = get_condition_output_dir(feature_root / METRICS_DIR_NAME, feature_mode, kernel)
            save_nested_cv_result_csvs(result, metrics_dir, "svm")

            selected_terms_dir = get_condition_output_dir(feature_root / SELECTED_TERMS_DIR_NAME, feature_mode, kernel)
            selected_term_paths = save_outer_fold_selected_terms(
                result,
                outer_fold_contexts,
                selected_terms_dir,
            )

            print_result(result, f"{format_condition_label(feature_mode, kernel)} / top{feature_count}")
            print(f"saved selected terms index csv: {selected_term_paths['selected_terms_index']}")

            summary_row = build_summary_row(result)
            summary_rows_by_condition[(feature_mode, kernel)].append(summary_row)

            if result.get("selected_records"):
                current_best = best_results_by_condition[(feature_mode, kernel)]
                if current_best is None or result["mean_accuracy"] > current_best["raw_accuracy"]:
                    best_results_by_condition[(feature_mode, kernel)] = {
                        "feature_count": result["feature_count"],
                        "feature_mode": result["feature_mode"],
                        "kernel": result["kernel"],
                        "mean_selected_term_count": round_half_up(result["mean_selected_term_count"]),
                        "mean_accuracy": round_half_up(result["mean_accuracy"]),
                        "mean_recall": round_half_up(result["mean_recall"]),
                        "mean_precision": round_half_up(result["mean_precision"]),
                        "mean_f1": round_half_up(result["mean_f1"]),
                        "best_cs": ",".join(map(str, result.get("best_cs", []))),
                        "best_gammas": ",".join(map(str, result.get("best_gammas", []))),
                        "raw_accuracy": result["mean_accuracy"],
                    }

    save_global_summary_outputs(
        summary_rows_by_condition,
        best_results_by_condition,
        SAVE_DIR / METRICS_DIR_NAME,
    )
