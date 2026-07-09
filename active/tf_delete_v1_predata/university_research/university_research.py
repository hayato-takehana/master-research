from __future__ import annotations

import concurrent.futures as futures
import multiprocessing as mp
from pathlib import Path
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _bootstrap_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "active").exists() and (candidate / "archive").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise RuntimeError("Project root could not be found.")


PROJECT_ROOT = _bootstrap_project_root()

from project_runtime import bootstrap_project_paths, get_output_dir, redirect_relative_outputs  # noqa: E402

bootstrap_project_paths(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
SAVE_DIR = get_output_dir(__file__, PROJECT_ROOT)
redirect_relative_outputs(SAVE_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn import svm  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from dataset_loader import (  # noqa: E402
    PRE_RESEARCH_DELETE_V1_DATASET_NAMES,
    load_documents,
)
from text_vectorizer import Tf_idf  # noqa: E402


REAL_SCAM = False
KEEP_DIGITS = False
PERCENT_MODE = "drop"
USE_STEMMING = True
MIN_LEN = 3
MIN_DF = 0.02

N_SPLITS = 10
RANDOM_STATE = 42
CLASS_WEIGHT = "balanced"
TOL = 1e-10
MARGIN_THRESHOLD = 0.9990

LINEAR_C_VALUES = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
RBF_C_VALUES = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
RBF_GAMMA_VALUES = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]

KERNELS = ("linear", "rbf")
KERNEL_DIR_NAMES = {
    "linear": "linear",
    "rbf": "rbf",
}
SPLIT_CONTEXT_WORKERS_ENV = "UNIVERSITY_RESEARCH_SPLIT_WORKERS"
DEFAULT_SPLIT_CONTEXT_WORKER_LIMIT = 10
CANDIDATE_WORKERS_ENV = "UNIVERSITY_RESEARCH_CANDIDATE_WORKERS"
DEFAULT_CANDIDATE_WORKER_LIMIT = 10

SUMMARY_COLUMNS = [
    "model",
    "kernel",
    "margin_threshold",
    "TP",
    "TN",
    "FP",
    "FN",
    "accuracy",
    "recall",
    "precision",
    "f1",
    "mean_accuracy",
    "mean_recall",
    "mean_precision",
    "mean_f1",
    "best_cs",
    "best_gammas",
    "feature_counts",
    "mean_feature_count",
    "valid_param_count",
    "dropped_param_count",
]

FOLD_COLUMNS = [
    "outer_fold",
    "kernel",
    "feature_count",
    "c",
    "gamma",
    "param_label",
    "inner_score",
    "TP",
    "TN",
    "FP",
    "FN",
    "accuracy",
    "recall",
    "precision",
    "f1",
]


def log_progress(message: str) -> None:
    print(message, flush=True)


def resolve_worker_count(env_name, default_worker_limit, job_count):
    """環境変数またはCPU数から、同時実行するプロセス数を決める。"""
    if job_count <= 1:
        return 1

    env_value = os.environ.get(env_name, "").strip()
    if env_value:
        try:
            requested_workers = int(env_value)
        except ValueError:
            log_progress(
                f"[parallel warning] {env_name}={env_value!r} is not an integer; "
                "using 1 worker"
            )
            return 1
        if requested_workers < 1:
            log_progress(
                f"[parallel warning] {env_name} must be >= 1; using 1 worker"
            )
            return 1
        return min(requested_workers, job_count)

    cpu_count = os.cpu_count() or 1
    return max(
        1,
        min(default_worker_limit, max(1, cpu_count - 1), job_count),
    )


def print_full_corpus_tfidf_feature_count(documents) -> None:
    """全文書でTF-IDFを作った場合の特徴数を参考値として表示する。"""
    tf_idf = Tf_idf(
        list(documents),
        True,
        MIN_LEN,
        use_stemming=USE_STEMMING,
    )
    X, _, _ = tf_idf.tf_idf(
        MIN_DF,
        ngram_range=(1, 1),
    )
    print(
        f"参考特徴数（全文書でTF-IDF、単語数）: {X.shape[1]}",
        flush=True,
    )


def load_corpus() -> tuple[np.ndarray, np.ndarray]:
    documents_1, documents_0 = load_documents(
        real_scam=REAL_SCAM,
        keep_digits=KEEP_DIGITS,
        percent_mode=PERCENT_MODE,
        dataset_names=PRE_RESEARCH_DELETE_V1_DATASET_NAMES,
    )
    documents = np.array(documents_1 + documents_0, dtype=object)
    labels = np.array(
        [1] * len(documents_1) + [0] * len(documents_0),
        dtype=int,
    )
    print(f"label1 documents: {len(documents_1)}")
    print(f"label0 documents: {len(documents_0)}")
    return documents, labels


class SplitTfidfContext:
    """1つのsplitについて、学習側だけでTF-IDFをfitした行列を保持する。"""

    def __init__(
        self,
        train_documents,
        train_labels,
        test_documents,
        test_labels,
    ):
        self.train_labels = np.asarray(train_labels, dtype=int)
        self.test_labels = np.asarray(test_labels, dtype=int)

        train_tf_idf = Tf_idf(
            list(train_documents),
            True,
            MIN_LEN,
            use_stemming=USE_STEMMING,
        )
        (
            self.X_train,
            feature_names,
            fitted_vectorizer,
        ) = train_tf_idf.tf_idf(
            MIN_DF,
            ngram_range=(1, 1),
        )

        test_tf_idf = Tf_idf(
            list(test_documents),
            True,
            MIN_LEN,
            use_stemming=USE_STEMMING,
        )
        self.X_test = fitted_vectorizer.transform(test_tf_idf.processed_docs)
        self.feature_count = len(feature_names)


def build_outer_fold_context_from_indices(
    outer_fold,
    outer_train_index,
    outer_test_index,
    documents,
    labels,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
):
    """1つのouter foldに必要なouter/inner TF-IDFコンテキストを構築する。"""
    fold_start = time.perf_counter()
    outer_train_documents = documents[outer_train_index]
    outer_train_labels = labels[outer_train_index]

    outer_context = SplitTfidfContext(
        outer_train_documents,
        outer_train_labels,
        documents[outer_test_index],
        labels[outer_test_index],
    )

    inner_cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    inner_contexts = []
    for inner_train_index, inner_test_index in inner_cv.split(
        outer_train_documents,
        outer_train_labels,
    ):
        inner_contexts.append(
            SplitTfidfContext(
                outer_train_documents[inner_train_index],
                outer_train_labels[inner_train_index],
                outer_train_documents[inner_test_index],
                outer_train_labels[inner_test_index],
            )
        )

    return {
        "outer_fold": outer_fold,
        "outer_context": outer_context,
        "inner_contexts": inner_contexts,
        "setup_elapsed_seconds": time.perf_counter() - fold_start,
    }


_SPLIT_WORKER_DOCUMENTS = None
_SPLIT_WORKER_LABELS = None


def _init_split_context_worker(documents, labels):
    """各常駐ワーカーへcorpusを一度だけ渡す。"""
    global _SPLIT_WORKER_DOCUMENTS
    global _SPLIT_WORKER_LABELS

    _SPLIT_WORKER_DOCUMENTS = documents
    _SPLIT_WORKER_LABELS = labels


def _build_outer_fold_context_worker(fold_payload):
    if _SPLIT_WORKER_DOCUMENTS is None or _SPLIT_WORKER_LABELS is None:
        raise RuntimeError("split context worker was not initialized")
    outer_fold, outer_train_index, outer_test_index, n_splits, random_state = (
        fold_payload
    )
    return build_outer_fold_context_from_indices(
        outer_fold,
        outer_train_index,
        outer_test_index,
        _SPLIT_WORKER_DOCUMENTS,
        _SPLIT_WORKER_LABELS,
        n_splits,
        random_state,
    )


def build_outer_fold_contexts(
    documents,
    labels,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
):
    outer_cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    fold_payloads = [
        (
            outer_fold,
            outer_train_index,
            outer_test_index,
            n_splits,
            random_state,
        )
        for outer_fold, (outer_train_index, outer_test_index) in enumerate(
            outer_cv.split(documents, labels),
            start=1,
        )
    ]
    worker_count = resolve_worker_count(
        SPLIT_CONTEXT_WORKERS_ENV,
        DEFAULT_SPLIT_CONTEXT_WORKER_LIMIT,
        len(fold_payloads),
    )
    log_progress(
        f"[split setup] workers: {worker_count} "
        f"(override with {SPLIT_CONTEXT_WORKERS_ENV})"
    )

    if worker_count == 1:
        outer_fold_contexts = [
            build_outer_fold_context_from_indices(
                outer_fold,
                outer_train_index,
                outer_test_index,
                documents,
                labels,
                split_count,
                split_random_state,
            )
            for (
                outer_fold,
                outer_train_index,
                outer_test_index,
                split_count,
                split_random_state,
            ) in fold_payloads
        ]
    else:
        outer_fold_contexts = []
        executor = futures.ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp.get_context("spawn"),
            initializer=_init_split_context_worker,
            initargs=(documents, labels),
        )
        try:
            future_to_fold = {
                executor.submit(_build_outer_fold_context_worker, payload): payload[0]
                for payload in fold_payloads
            }
            for completed_count, future in enumerate(
                futures.as_completed(future_to_fold),
                start=1,
            ):
                fold_context = future.result()
                outer_fold_contexts.append(fold_context)
                log_progress(
                    f"[split setup] outer fold "
                    f"{fold_context['outer_fold']}/{n_splits} completed "
                    f"({completed_count}/{len(fold_payloads)}, "
                    f"elapsed_seconds={fold_context['setup_elapsed_seconds']:.1f})"
                )
        finally:
            executor.shutdown(cancel_futures=True)

    outer_fold_contexts.sort(key=lambda context: context["outer_fold"])
    return outer_fold_contexts


def build_svc(kernel, c, gamma=None):
    kwargs = {
        "kernel": kernel,
        "C": c,
        "class_weight": CLASS_WEIGHT,
        "tol": TOL,
    }
    if gamma is not None:
        kwargs["gamma"] = gamma
    return svm.SVC(**kwargs)


def compute_signed_margins(model, X_train, y_train):
    decision_values = np.asarray(model.decision_function(X_train)).ravel()
    signed_labels = np.where(np.asarray(y_train) == 0, -1, 1)
    return signed_labels * decision_values


def format_param_label(c, gamma=None):
    if gamma is None:
        return f"C={c}"
    return f"C={c}, gamma={gamma}"


def fit_and_evaluate_context(
    split_context,
    kernel,
    c,
    gamma=None,
    margin_threshold=MARGIN_THRESHOLD,
):
    model = build_svc(kernel, c, gamma)
    model.fit(split_context.X_train, split_context.train_labels)

    margins = compute_signed_margins(
        model,
        split_context.X_train,
        split_context.train_labels,
    )
    if not np.all(margins >= margin_threshold):
        return None

    predictions = model.predict(split_context.X_test)
    tn, fp, fn, tp = confusion_matrix(
        split_context.test_labels,
        predictions,
        labels=[0, 1],
    ).ravel()
    return {
        "feature_count": split_context.feature_count,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "accuracy": float(accuracy_score(split_context.test_labels, predictions)),
        "recall": float(recall_score(split_context.test_labels, predictions, zero_division=0)),
        "precision": float(precision_score(split_context.test_labels, predictions, zero_division=0)),
        "f1": float(f1_score(split_context.test_labels, predictions, zero_division=0)),
    }


def evaluate_candidate_across_outer_folds(
    outer_fold_contexts,
    kernel,
    c,
    gamma=None,
    margin_threshold=MARGIN_THRESHOLD,
):
    """全inner/outer学習データでmargin条件を満たす候補だけを返す。"""
    param_label = format_param_label(c, gamma)
    fold_records = []

    for fold_bundle in outer_fold_contexts:
        inner_scores = []
        for inner_context in fold_bundle["inner_contexts"]:
            inner_result = fit_and_evaluate_context(
                inner_context,
                kernel,
                c,
                gamma,
                margin_threshold,
            )
            if inner_result is None:
                return None
            inner_scores.append(inner_result["accuracy"])

        outer_result = fit_and_evaluate_context(
            fold_bundle["outer_context"],
            kernel,
            c,
            gamma,
            margin_threshold,
        )
        if outer_result is None:
            return None

        fold_records.append(
            {
                "outer_fold": fold_bundle["outer_fold"],
                "kernel": kernel,
                "c": c,
                "gamma": gamma,
                "param_label": param_label,
                "inner_score": float(np.mean(inner_scores)),
                **outer_result,
            }
        )

    return fold_records


_CANDIDATE_WORKER_OUTER_FOLD_CONTEXTS = None
_CANDIDATE_WORKER_KERNEL = None
_CANDIDATE_WORKER_MARGIN_THRESHOLD = None


def _init_candidate_worker(outer_fold_contexts, kernel, margin_threshold):
    """各常駐ワーカーへTF-IDFコンテキストを一度だけ渡す。"""
    global _CANDIDATE_WORKER_OUTER_FOLD_CONTEXTS
    global _CANDIDATE_WORKER_KERNEL
    global _CANDIDATE_WORKER_MARGIN_THRESHOLD

    _CANDIDATE_WORKER_OUTER_FOLD_CONTEXTS = outer_fold_contexts
    _CANDIDATE_WORKER_KERNEL = kernel
    _CANDIDATE_WORKER_MARGIN_THRESHOLD = margin_threshold


def _evaluate_candidate_worker(candidate):
    if _CANDIDATE_WORKER_OUTER_FOLD_CONTEXTS is None:
        raise RuntimeError("candidate worker was not initialized")
    c, gamma = candidate
    candidate_records = evaluate_candidate_across_outer_folds(
        _CANDIDATE_WORKER_OUTER_FOLD_CONTEXTS,
        _CANDIDATE_WORKER_KERNEL,
        c,
        gamma,
        _CANDIDATE_WORKER_MARGIN_THRESHOLD,
    )
    return c, gamma, candidate_records


def build_empty_result(kernel, margin_threshold=MARGIN_THRESHOLD):
    return {
        "model": f"{kernel}_svm",
        "kernel": kernel,
        "margin_threshold": margin_threshold,
        "valid_param_labels": [],
        "dropped_param_labels": [],
        "selected_records": [],
        "best_cs": [],
        "feature_counts": [],
        "mean_feature_count": "",
    }


def aggregate_selected_records(
    kernel,
    selected_records,
    margin_threshold=MARGIN_THRESHOLD,
):
    result = build_empty_result(kernel, margin_threshold)
    result["selected_records"] = selected_records
    result["best_cs"] = [record["c"] for record in selected_records]
    result["feature_counts"] = [record["feature_count"] for record in selected_records]
    result["mean_feature_count"] = float(np.mean(result["feature_counts"]))
    if kernel == "rbf":
        result["best_gammas"] = [record["gamma"] for record in selected_records]

    for metric in ("TP", "TN", "FP", "FN"):
        result[metric] = int(sum(record[metric] for record in selected_records))

    total = result["TP"] + result["TN"] + result["FP"] + result["FN"]
    result["accuracy"] = (
        (result["TP"] + result["TN"]) / total
        if total
        else 0.0
    )
    result["recall"] = (
        result["TP"] / (result["TP"] + result["FN"])
        if result["TP"] + result["FN"]
        else 0.0
    )
    result["precision"] = (
        result["TP"] / (result["TP"] + result["FP"])
        if result["TP"] + result["FP"]
        else 0.0
    )
    result["f1"] = (
        2 * result["precision"] * result["recall"]
        / (result["precision"] + result["recall"])
        if result["precision"] + result["recall"]
        else 0.0
    )

    for metric in ("accuracy", "recall", "precision", "f1"):
        result[f"mean_{metric}"] = float(
            np.mean([record[metric] for record in selected_records])
        )

    return result


def run_margin_constrained_nested_cv(
    outer_fold_contexts,
    kernel,
    c_values=None,
    gamma_values=None,
    margin_threshold=MARGIN_THRESHOLD,
):
    if c_values is None:
        c_values = LINEAR_C_VALUES if kernel == "linear" else RBF_C_VALUES
    if kernel == "linear":
        gamma_candidates = [None]
    else:
        gamma_candidates = RBF_GAMMA_VALUES if gamma_values is None else gamma_values

    candidates = [
        (c, gamma)
        for c in c_values
        for gamma in gamma_candidates
    ]
    total_candidates = len(candidates)
    valid_candidate_results = {}
    dropped_param_labels = []
    processed_candidates = 0
    start_time = time.perf_counter()
    worker_count = resolve_worker_count(
        CANDIDATE_WORKERS_ENV,
        DEFAULT_CANDIDATE_WORKER_LIMIT,
        total_candidates,
    )

    log_progress(
        f"\n[{kernel} start] {total_candidates} parameter settings, "
        f"margin >= {margin_threshold}, workers={worker_count} "
        f"(override with {CANDIDATE_WORKERS_ENV})"
    )

    def record_candidate_result(c, gamma, candidate_records):
        nonlocal processed_candidates
        processed_candidates += 1
        if candidate_records is None:
            dropped_param_labels.append(format_param_label(c, gamma))
        else:
            valid_candidate_results[(c, gamma)] = candidate_records

        if (
            processed_candidates == 1
            or processed_candidates == total_candidates
            or processed_candidates % 10 == 0
        ):
            elapsed = time.perf_counter() - start_time
            log_progress(
                f"[{kernel} progress] {processed_candidates}/{total_candidates} "
                f"(valid={len(valid_candidate_results)}, "
                f"dropped={len(dropped_param_labels)}, "
                f"elapsed_seconds={elapsed:.1f})"
            )

    if worker_count == 1:
        for c, gamma in candidates:
            candidate_records = evaluate_candidate_across_outer_folds(
                outer_fold_contexts,
                kernel,
                c,
                gamma,
                margin_threshold,
            )
            record_candidate_result(c, gamma, candidate_records)
    else:
        executor = futures.ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp.get_context("spawn"),
            initializer=_init_candidate_worker,
            initargs=(outer_fold_contexts, kernel, margin_threshold),
        )
        try:
            future_to_candidate = {
                executor.submit(_evaluate_candidate_worker, candidate): candidate
                for candidate in candidates
            }
            for future in futures.as_completed(future_to_candidate):
                c, gamma, candidate_records = future.result()
                record_candidate_result(c, gamma, candidate_records)
        finally:
            executor.shutdown(cancel_futures=True)

    if not valid_candidate_results:
        result = build_empty_result(kernel, margin_threshold)
        dropped_param_label_set = set(dropped_param_labels)
        result["dropped_param_labels"] = [
            format_param_label(c, gamma)
            for c, gamma in candidates
            if format_param_label(c, gamma) in dropped_param_label_set
        ]
        return result

    selected_records = []
    for fold_index in range(len(outer_fold_contexts)):
        best_record = None
        for c, gamma in candidates:
            candidate_records = valid_candidate_results.get((c, gamma))
            if candidate_records is None:
                continue
            candidate = candidate_records[fold_index]
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

    result = aggregate_selected_records(
        kernel,
        selected_records,
        margin_threshold,
    )
    result["valid_param_labels"] = [
        format_param_label(c, gamma)
        for c, gamma in candidates
        if (c, gamma) in valid_candidate_results
    ]
    dropped_param_label_set = set(dropped_param_labels)
    result["dropped_param_labels"] = [
        format_param_label(c, gamma)
        for c, gamma in candidates
        if format_param_label(c, gamma) in dropped_param_label_set
    ]
    return result


def result_to_summary_row(result):
    return {
        "model": result["model"],
        "kernel": result["kernel"],
        "margin_threshold": result["margin_threshold"],
        "TP": result.get("TP", ""),
        "TN": result.get("TN", ""),
        "FP": result.get("FP", ""),
        "FN": result.get("FN", ""),
        "accuracy": result.get("accuracy", ""),
        "recall": result.get("recall", ""),
        "precision": result.get("precision", ""),
        "f1": result.get("f1", ""),
        "mean_accuracy": result.get("mean_accuracy", ""),
        "mean_recall": result.get("mean_recall", ""),
        "mean_precision": result.get("mean_precision", ""),
        "mean_f1": result.get("mean_f1", ""),
        "best_cs": ",".join(map(str, result.get("best_cs", []))),
        "best_gammas": ",".join(map(str, result.get("best_gammas", []))),
        "feature_counts": ",".join(map(str, result.get("feature_counts", []))),
        "mean_feature_count": result.get("mean_feature_count", ""),
        "valid_param_count": len(result.get("valid_param_labels", [])),
        "dropped_param_count": len(result.get("dropped_param_labels", [])),
    }


def save_result(result):
    output_dir = SAVE_DIR / KERNEL_DIR_NAMES[result["kernel"]]
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "metrics.csv"
    folds_path = output_dir / "folds.csv"
    params_path = output_dir / "params.csv"

    pd.DataFrame(
        [result_to_summary_row(result)],
        columns=SUMMARY_COLUMNS,
    ).to_csv(summary_path, index=False, encoding="utf-8-sig")

    pd.DataFrame(
        result.get("selected_records", []),
        columns=FOLD_COLUMNS,
    ).to_csv(folds_path, index=False, encoding="utf-8-sig")

    param_rows = [
        {"status": "valid", "param_label": label}
        for label in result.get("valid_param_labels", [])
    ]
    param_rows.extend(
        {"status": "dropped", "param_label": label}
        for label in result.get("dropped_param_labels", [])
    )
    pd.DataFrame(
        param_rows,
        columns=["status", "param_label"],
    ).to_csv(params_path, index=False, encoding="utf-8-sig")

    return {
        "metrics": summary_path,
        "folds": folds_path,
        "params": params_path,
    }


def print_result(result, saved_paths):
    model_label = "線形SVM" if result["kernel"] == "linear" else "非線形SVM（RBF）"
    print(f"\n{model_label}")
    if not result.get("selected_records"):
        print(
            f"全inner/outer foldでmargin >= {result['margin_threshold']}を満たす"
            "パラメータがありませんでした。"
        )
    else:
        print(f"TP: {result['TP']}")
        print(f"TN: {result['TN']}")
        print(f"FP: {result['FP']}")
        print(f"FN: {result['FN']}")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Recall: {result['recall']:.4f}")
        print(f"Precision: {result['precision']:.4f}")
        print(f"F1 Score: {result['f1']:.4f}")
        print(f"selected C values: {result['best_cs']}")
        if result["kernel"] == "rbf":
            print(f"selected gamma values: {result['best_gammas']}")
        print(
            f"外側{N_SPLITS}foldの平均特徴数（単語数）: "
            f"{result['mean_feature_count']:.1f}"
        )
    print(f"saved metrics: {saved_paths['metrics']}")
    print(f"saved folds: {saved_paths['folds']}")
    print(f"saved params: {saved_paths['params']}")


def main() -> int:
    start_time = time.time()
    documents, labels = load_corpus()
    print_full_corpus_tfidf_feature_count(documents)
    log_progress("[main] building leakage-free outer/inner TF-IDF contexts")
    outer_fold_contexts = build_outer_fold_contexts(documents, labels)

    results = []
    for kernel in KERNELS:
        result = run_margin_constrained_nested_cv(
            outer_fold_contexts,
            kernel,
        )
        saved_paths = save_result(result)
        print_result(result, saved_paths)
        results.append(result)

    summary_path = SAVE_DIR / "summary.csv"
    pd.DataFrame(
        [result_to_summary_row(result) for result in results],
        columns=SUMMARY_COLUMNS,
    ).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\nsaved summary: {summary_path}")
    print(f"elapsed_seconds: {time.time() - start_time:.2f}")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
