from __future__ import annotations

from pathlib import Path
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

from project_runtime import bootstrap_project_paths, get_output_dir, redirect_relative_outputs  # noqa: E402

bootstrap_project_paths(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
SAVE_DIR = get_output_dir(__file__, PROJECT_ROOT)
redirect_relative_outputs(SAVE_DIR)

import pandas as pd  # noqa: E402
from sklearn import svm  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict  # noqa: E402

from text_vectorizer import Tf_idf  # noqa: E402


USE_STEMMING = True
MIN_LEN = 3
MIN_DF = 0.02
N_SPLITS = 10
RANDOM_STATE = 42
RESULT_CSV = SAVE_DIR / "pre_research_svm_reproduction_results.csv"
FINAL_DATASET_PATH = PROJECT_ROOT / "Final_Dataset.csv"
TEXT_COLUMN = "Fillings"
LABEL_COLUMN = "Fraud"


def load_corpus() -> tuple[list[str], list[int]]:
    dataset = pd.read_csv(FINAL_DATASET_PATH, usecols=[TEXT_COLUMN, LABEL_COLUMN])
    dataset = dataset.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).reset_index(drop=True)

    normalized_labels = dataset[LABEL_COLUMN].astype(str).str.strip().str.lower()
    label_map = {"yes": 1, "no": 0}
    unknown_labels = sorted(set(normalized_labels) - set(label_map))
    if unknown_labels:
        raise ValueError(
            f"{LABEL_COLUMN} column contains unsupported labels: {unknown_labels}. "
            "Expected labels are 'yes' and 'no'."
        )

    documents_1 = dataset.loc[normalized_labels == "yes", TEXT_COLUMN].astype(str).tolist()
    documents_0 = dataset.loc[normalized_labels == "no", TEXT_COLUMN].astype(str).tolist()

    print(len(documents_1))
    print(len(documents_0))

    documents = documents_1 + documents_0
    labels = [1] * len(documents_1) + [0] * len(documents_0)
    return documents, labels


def build_tfidf_features(documents: list[str]):
    tf_idf = Tf_idf(
        documents,
        True,
        MIN_LEN,
        use_stemming=USE_STEMMING,
    )
    X, feature_names, vectorizer = tf_idf.tf_idf(MIN_DF, ngram_range=(1, 1))
    print("Number of features:", len(feature_names))
    return X, feature_names, vectorizer


def evaluate_svm_prior_research(X, labels: list[int]) -> dict[str, object]:
    param_grid = {
        "kernel": ["linear", "rbf", "sigmoid"],
        "C": [0.1, 0.5, 1, 3, 5],
        "gamma": [0.01, 0.001, 0.0001],
    }
    clf = svm.SVC(class_weight="balanced", random_state=RANDOM_STATE)

    inner_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=inner_cv, scoring="accuracy")

    outer_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    predictions = cross_val_predict(grid_search, X, labels, cv=outer_cv)

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
    X, _, _ = build_tfidf_features(documents)
    result = evaluate_svm_prior_research(X, labels)

    result_df = pd.DataFrame([result])
    result_df.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
    print(f"saved: {RESULT_CSV}")
    print(f"elapsed_seconds: {time.time() - start_time:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
