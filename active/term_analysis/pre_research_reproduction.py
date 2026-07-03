from __future__ import annotations

from pathlib import Path
import os
import sys
import tempfile
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

from project_runtime import bootstrap_project_paths, get_output_dir, redirect_relative_outputs  # noqa: E402

bootstrap_project_paths(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
SAVE_DIR = get_output_dir(__file__, PROJECT_ROOT)
redirect_relative_outputs(SAVE_DIR)

import pandas as pd  # noqa: E402
from sklearn import svm  # noqa: E402
from sklearn.base import BaseEstimator, TransformerMixin  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.utils.validation import check_is_fitted  # noqa: E402

from dataset_loader import load_documents  # noqa: E402
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


def evaluate_svm_prior_research(documents: list[str], labels: list[int]) -> dict[str, object]:
    param_grid = {
        "svc__kernel": ["linear", "rbf", "sigmoid"],
        "svc__C": [0.1, 0.5, 1, 3, 5],
        "svc__gamma": [0.01, 0.001, 0.0001],
    }

    inner_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    outer_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Pipelineのキャッシュにより、同一inner splitのTF-IDF再計算を候補ごとに繰り返さない。
    with tempfile.TemporaryDirectory(prefix="pre_research_tfidf_") as cache_dir:
        pipeline = build_svm_pipeline(memory=cache_dir)
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="accuracy",
        )
        predictions = cross_val_predict(grid_search, documents, labels, cv=outer_cv)

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
    }


def main() -> int:
    start_time = time.time()
    documents, labels = load_corpus()
    result = evaluate_svm_prior_research(documents, labels)

    result_df = pd.DataFrame([result])
    result_df.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
    print(f"saved: {RESULT_CSV}")
    print(f"elapsed_seconds: {time.time() - start_time:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
