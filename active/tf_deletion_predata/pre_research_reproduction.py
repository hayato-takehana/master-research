from __future__ import annotations

import concurrent.futures as futures
import multiprocessing as mp
from pathlib import Path
import os
import sys
import tempfile
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

import pandas as pd  # noqa: E402
from sklearn import svm  # noqa: E402
from sklearn.base import BaseEstimator, TransformerMixin  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.utils.validation import check_is_fitted  # noqa: E402

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
RESULT_CSV = SAVE_DIR / "pre_research_svm_reproduction_results.csv"
OUTER_FOLD_WORKERS_ENV = "PRE_RESEARCH_OUTER_FOLD_WORKERS"
DEFAULT_OUTER_FOLD_WORKER_LIMIT = 10
SVM_PARAM_GRID = {
    "svc__kernel": ["linear", "rbf", "sigmoid"],
    "svc__C": [0.1, 0.5, 1, 3, 5],
    "svc__gamma": [0.01, 0.001, 0.0001],
}


def print_full_corpus_tfidf_feature_count(documents: list[str]) -> None:
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


class PriorResearchTfidfTransformer(TransformerMixin, BaseEstimator):
    """学習foldだけで前処理・語彙・IDFを学習するTF-IDF変換器。"""

    def __init__(self, min_len=MIN_LEN, min_df=MIN_DF, use_stemming=USE_STEMMING):
        self.min_len = min_len
        self.min_df = min_df
        self.use_stemming = use_stemming

    def _fit_documents(self, documents):
        tf_idf = Tf_idf(
            list(documents),
            True,
            self.min_len,
            use_stemming=self.use_stemming,
        )
        X, feature_names, vectorizer = tf_idf.tf_idf(
            self.min_df,
            ngram_range=(1, 1),
        )
        self.feature_names_ = feature_names
        self.vectorizer_ = vectorizer
        return X

    def fit(self, documents, labels=None):
        self._fit_documents(documents)
        return self

    def fit_transform(self, documents, labels=None, **fit_params):
        return self._fit_documents(documents)

    def transform(self, documents):
        check_is_fitted(self, ("feature_names_", "vectorizer_"))
        tf_idf = Tf_idf(
            list(documents),
            True,
            self.min_len,
            use_stemming=self.use_stemming,
        )
        return self.vectorizer_.transform(tf_idf.processed_docs)


def load_corpus() -> tuple[list[str], list[int]]:
    documents_1, documents_0 = load_documents(
        real_scam=REAL_SCAM,
        keep_digits=KEEP_DIGITS,
        percent_mode=PERCENT_MODE,
        dataset_names=PRE_RESEARCH_DELETE_V1_DATASET_NAMES,
    )

    print(len(documents_1))
    print(len(documents_0))

    documents = documents_1 + documents_0
    labels = [1] * len(documents_1) + [0] * len(documents_0)
    return documents, labels


def build_svm_pipeline(memory=None) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                PriorResearchTfidfTransformer(),
            ),
            (
                "svc",
                svm.SVC(class_weight="balanced", random_state=RANDOM_STATE),
            ),
        ],
        memory=memory,
    )


def resolve_worker_count(env_name: str, default_worker_limit: int, job_count: int) -> int:
    """環境変数またはCPU数からプロセス数を決める。"""
    if job_count <= 1:
        return 1

    env_value = os.environ.get(env_name, "").strip()
    if env_value:
        try:
            requested_workers = int(env_value)
        except ValueError:
            print(
                f"[parallel warning] {env_name}={env_value!r} is not an integer; "
                "using 1 worker",
                flush=True,
            )
            return 1
        if requested_workers < 1:
            print(
                f"[parallel warning] {env_name} must be >= 1; using 1 worker",
                flush=True,
            )
            return 1
        return min(requested_workers, job_count)

    cpu_count = os.cpu_count() or 1
    return max(
        1,
        min(default_worker_limit, max(1, cpu_count - 1), job_count),
    )


def evaluate_outer_fold(
    outer_fold: int,
    train_indices,
    test_indices,
    documents: list[str],
    labels: list[int],
    n_splits: int = N_SPLITS,
    param_grid=None,
) -> dict[str, object]:
    """1つの外側foldでinner GridSearchと最終予測を行う。"""
    if param_grid is None:
        param_grid = SVM_PARAM_GRID
    fold_start = time.perf_counter()
    train_documents = [documents[int(index)] for index in train_indices]
    train_labels = [labels[int(index)] for index in train_indices]
    test_documents = [documents[int(index)] for index in test_indices]

    inner_cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    with tempfile.TemporaryDirectory(
        prefix=f"pre_research_fold_{outer_fold:02d}_"
    ) as cache_dir:
        grid_search = GridSearchCV(
            estimator=build_svm_pipeline(memory=cache_dir),
            param_grid=param_grid,
            cv=inner_cv,
            scoring="accuracy",
            n_jobs=1,
        )
        grid_search.fit(train_documents, train_labels)
        predictions = grid_search.predict(test_documents)

    feature_count = len(
        grid_search.best_estimator_.named_steps["tfidf"].feature_names_
    )
    return {
        "outer_fold": outer_fold,
        "test_indices": [int(index) for index in test_indices],
        "predictions": [int(prediction) for prediction in predictions],
        "feature_count": feature_count,
        "elapsed_seconds": time.perf_counter() - fold_start,
    }


_OUTER_WORKER_DOCUMENTS = None
_OUTER_WORKER_LABELS = None
_OUTER_WORKER_N_SPLITS = None
_OUTER_WORKER_PARAM_GRID = None


def _init_outer_fold_worker(documents, labels, n_splits, param_grid) -> None:
    """各常駐ワーカーへデータと実験設定を一度だけ渡す。"""
    global _OUTER_WORKER_DOCUMENTS
    global _OUTER_WORKER_LABELS
    global _OUTER_WORKER_N_SPLITS
    global _OUTER_WORKER_PARAM_GRID

    _OUTER_WORKER_DOCUMENTS = documents
    _OUTER_WORKER_LABELS = labels
    _OUTER_WORKER_N_SPLITS = n_splits
    _OUTER_WORKER_PARAM_GRID = param_grid


def _evaluate_outer_fold_worker(fold_payload) -> dict[str, object]:
    if _OUTER_WORKER_DOCUMENTS is None or _OUTER_WORKER_LABELS is None:
        raise RuntimeError("outer fold worker was not initialized")
    outer_fold, train_indices, test_indices = fold_payload
    return evaluate_outer_fold(
        outer_fold,
        train_indices,
        test_indices,
        _OUTER_WORKER_DOCUMENTS,
        _OUTER_WORKER_LABELS,
        _OUTER_WORKER_N_SPLITS,
        _OUTER_WORKER_PARAM_GRID,
    )


def run_outer_folds(
    documents: list[str],
    labels: list[int],
) -> tuple[list[int], list[int]]:
    """外側foldを常駐プロセスプールで実行し、予測と特徴数を返す。"""
    outer_cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    fold_payloads = [
        (outer_fold, train_indices, test_indices)
        for outer_fold, (train_indices, test_indices) in enumerate(
            outer_cv.split(documents, labels),
            start=1,
        )
    ]
    worker_count = resolve_worker_count(
        OUTER_FOLD_WORKERS_ENV,
        DEFAULT_OUTER_FOLD_WORKER_LIMIT,
        len(fold_payloads),
    )
    print(
        f"[parallel] outer-fold workers: {worker_count} "
        f"(override with {OUTER_FOLD_WORKERS_ENV})",
        flush=True,
    )

    if worker_count == 1:
        fold_results = [
            evaluate_outer_fold(
                outer_fold,
                train_indices,
                test_indices,
                documents,
                labels,
                N_SPLITS,
                SVM_PARAM_GRID,
            )
            for outer_fold, train_indices, test_indices in fold_payloads
        ]
    else:
        fold_results = []
        executor = futures.ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp.get_context("spawn"),
            initializer=_init_outer_fold_worker,
            initargs=(documents, labels, N_SPLITS, SVM_PARAM_GRID),
        )
        try:
            future_to_fold = {
                executor.submit(_evaluate_outer_fold_worker, payload): payload[0]
                for payload in fold_payloads
            }
            for completed_count, future in enumerate(
                futures.as_completed(future_to_fold),
                start=1,
            ):
                result = future.result()
                fold_results.append(result)
                print(
                    f"[outer fold] {result['outer_fold']}/{N_SPLITS} completed "
                    f"({completed_count}/{len(fold_payloads)}, "
                    f"elapsed_seconds={result['elapsed_seconds']:.1f})",
                    flush=True,
                )
        finally:
            executor.shutdown(cancel_futures=True)

    fold_results.sort(key=lambda result: int(result["outer_fold"]))
    predictions = [None] * len(documents)
    feature_counts = []
    for result in fold_results:
        feature_counts.append(int(result["feature_count"]))
        for index, prediction in zip(
            result["test_indices"],
            result["predictions"],
        ):
            predictions[int(index)] = int(prediction)
    if any(prediction is None for prediction in predictions):
        raise RuntimeError("outer-fold predictions do not cover all documents")
    return [int(prediction) for prediction in predictions], feature_counts


def evaluate_svm_prior_research(documents: list[str], labels: list[int]) -> dict[str, object]:
    predictions, outer_feature_counts = run_outer_folds(
        documents,
        labels,
    )
    mean_outer_feature_count = sum(outer_feature_counts) / len(outer_feature_counts)

    TP = TN = FP = FN = 0
    for true, pred in zip(labels, predictions):
        if true == 1 and pred == 1:
            TP += 1
        elif true == 0 and pred == 0:
            TN += 1
        elif true == 0 and pred == 1:
            FP += 1
        elif true == 1 and pred == 0:
            FN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    print("SVMの先行研究の精度")
    print(f"True Positives (TP): {TP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"外側{N_SPLITS}foldの平均特徴数（単語数）: {mean_outer_feature_count:.1f}")

    return {
        "model": "SVMの先行研究の精度",
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "outer_feature_counts": ",".join(map(str, outer_feature_counts)),
        "mean_outer_feature_count": mean_outer_feature_count,
    }


def main() -> int:
    start_time = time.time()
    documents, labels = load_corpus()
    print_full_corpus_tfidf_feature_count(documents)
    result = evaluate_svm_prior_research(documents, labels)

    result_df = pd.DataFrame([result])
    result_df.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
    print(f"saved: {RESULT_CSV}")
    print(f"elapsed_seconds: {time.time() - start_time:.2f}")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
