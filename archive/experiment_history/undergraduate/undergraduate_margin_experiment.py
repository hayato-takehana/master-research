from pathlib import Path
import csv
import os
import sys


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

from pdf_text_loader import Text_road_and_dell
from text_vectorizer import Tf_idf
from dataset_loader import load_document_filenames

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn import svm


MIN_LEN = 3
MIN_DF = 0.02
USE_STEMMING = True
KEEP_DIGITS = False
PERCENT_MODE = "drop"
REMOVE_INDEX = 3
MARGIN_THRESHOLD = 0.9990

text_sagi = Text_road_and_dell(
    "document_詐欺_先行研究.pkl",
    "詐欺_先行研究",
    keep_digits=KEEP_DIGITS,
    percent_mode=PERCENT_MODE,
)
documents_1 = text_sagi.read_PDF()

text_no_sagi = Text_road_and_dell(
    "document_詐欺じゃない_先行研究.pkl",
    "詐欺じゃない_先行研究",
    keep_digits=KEEP_DIGITS,
    percent_mode=PERCENT_MODE,
)
documents_0 = text_no_sagi.read_PDF()

documents = documents_1 + documents_0
document_names_1, document_names_0 = load_document_filenames(real_scam=False)
document_names = document_names_1 + document_names_0

tf_idf = Tf_idf(documents, True, MIN_LEN, use_stemming=USE_STEMMING)
labels = tf_idf.labels(documents_1)
X, feature_names, vectorizer = tf_idf.tf_idf(MIN_DF)
X = X.toarray()





#X = np.delete(X, REMOVE_INDEX, axis=0)
#labels.pop(REMOVE_INDEX)
#document_names.pop(REMOVE_INDEX)
#X = np.delete(X, REMOVE_INDEX, axis=0)
#labels.pop(REMOVE_INDEX)
#document_names.pop(REMOVE_INDEX)

labels = np.array(labels)


def report_margin_violations(title, indices, y_true, margins, document_names):
    violation_mask = margins < MARGIN_THRESHOLD
    violation_local_indices = np.where(violation_mask)[0]

    print(f"\n{title}")
    if len(violation_local_indices) == 0:
        print("マージン違反はありません。")
        return

    print("マージン違反データ一覧")
    for local_idx in violation_local_indices:
        global_idx = int(indices[local_idx])
        label = int(y_true[local_idx])
        pdf_name = document_names[global_idx] if global_idx < len(document_names) else ""
        margin_value = float(margins[local_idx])
        print(
            f"配列番号={global_idx}, label={label}, margin={margin_value:.6f}, pdf={pdf_name}"
        )


def save_final_metrics_csv(filename, model_name, accuracies, recalls, precisions, f1_scores, best_cs, best_gammas=None):
    output_path = SAVE_DIR / filename
    row = {
        "model_name": model_name,
        "mean_accuracy": sum(accuracies) / len(accuracies) if accuracies else "",
        "mean_recall": sum(recalls) / len(recalls) if recalls else "",
        "mean_precision": sum(precisions) / len(precisions) if precisions else "",
        "mean_f1_score": sum(f1_scores) / len(f1_scores) if f1_scores else "",
        "best_cs": ",".join(map(str, best_cs)),
        "best_gammas": ",".join(map(str, best_gammas)) if best_gammas is not None else "",
    }

    with open(output_path, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

def save_selected_fold_results_csv(filename, model_name, selected_records):
    output_path = SAVE_DIR / filename
    fieldnames = [
        "model_name",
        "outer_fold",
        "param_label",
        "inner_score",
        "accuracy",
        "recall",
        "precision",
        "f1_score",
    ]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in selected_records:
            writer.writerow(
                {
                    "model_name": model_name,
                    "outer_fold": record["outer_fold"],
                    "param_label": record["param_label"],
                    "inner_score": record["inner_score"],
                    "accuracy": record["accuracy"],
                    "recall": record["recall"],
                    "precision": record["precision"],
                    "f1_score": record["f1_score"],
                }
            )


def build_svc(kernel, c, gamma=None):
    kwargs = {
        "kernel": kernel,
        "C": c,
        "class_weight": "balanced",
        "tol": 1e-10,
    }
    if gamma is not None:
        kwargs["gamma"] = gamma
    return svm.SVC(**kwargs)


def compute_margins(model, X_train, y_train):
    decision_values = model.decision_function(X_train)
    y_signed = np.where(y_train == 0, -1, 1)
    margins = y_signed * decision_values
    return margins, np.all(margins >= MARGIN_THRESHOLD)


def format_param_label(c, gamma=None):
    if gamma is None:
        return f"C={c}"
    return f"C={c}, gamma={gamma}"


def evaluate_parameter_across_folds(
    model_title,
    kernel,
    c,
    gamma,
    X,
    y,
    outer_splits,
    document_names,
):
    fold_records = []
    param_label = format_param_label(c, gamma)

    for outer_fold, (train_index, test_index) in enumerate(outer_splits, start=1):
        inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        inner_scores = []
        X_train_outer = X[train_index]
        y_train_outer = y[train_index]

        for inner_train_index, inner_test_index in inner_cv.split(X_train_outer, y_train_outer):
            inner_train_global = train_index[inner_train_index]
            inner_test_global = train_index[inner_test_index]

            X_train_inner = X[inner_train_global]
            y_train_inner = y[inner_train_global]
            X_test_inner = X[inner_test_global]
            y_test_inner = y[inner_test_global]

            model_inner = build_svc(kernel, c, gamma)
            model_inner.fit(X_train_inner, y_train_inner)
            margins_inner, inner_ok = compute_margins(model_inner, X_train_inner, y_train_inner)

            if not inner_ok:
                report_margin_violations(
                    f"{model_title} 内側CVでのマージン違反 ({param_label}, outer_fold={outer_fold})",
                    inner_train_global,
                    y_train_inner,
                    margins_inner,
                    document_names,
                )
                return None

            inner_scores.append(model_inner.score(X_test_inner, y_test_inner))

        model_outer = build_svc(kernel, c, gamma)
        model_outer.fit(X_train_outer, y_train_outer)
        margins_outer, outer_ok = compute_margins(model_outer, X_train_outer, y_train_outer)

        if not outer_ok:
            report_margin_violations(
                f"{model_title} 外側CVでのマージン違反 ({param_label}, outer_fold={outer_fold})",
                train_index,
                y_train_outer,
                margins_outer,
                document_names,
            )
            return None

        y_pred = model_outer.predict(X[test_index])
        fold_records.append(
            {
                "outer_fold": outer_fold,
                "c": c,
                "gamma": gamma,
                "param_label": param_label,
                "inner_score": sum(inner_scores) / len(inner_scores),
                "accuracy": accuracy_score(y[test_index], y_pred),
                "recall": recall_score(y[test_index], y_pred),
                "precision": precision_score(y[test_index], y_pred),
                "f1_score": f1_score(y[test_index], y_pred),
            }
        )

    return fold_records


def run_margin_constrained_nested_cv(
    model_name,
    model_title,
    kernel,
    param_grid,
    X,
    y,
    document_names,
    metrics_filename,
    selected_results_filename,
):
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    outer_splits = list(outer_cv.split(X, y))

    valid_param_results = {}
    dropped_param_labels = []

    for c, gamma in param_grid:
        param_label = format_param_label(c, gamma)
        fold_records = evaluate_parameter_across_folds(
            model_title=model_title,
            kernel=kernel,
            c=c,
            gamma=gamma,
            X=X,
            y=y,
            outer_splits=outer_splits,
            document_names=document_names,
        )

        if fold_records is None:
            dropped_param_labels.append(param_label)
            continue

        valid_param_results[(c, gamma)] = fold_records

    if not valid_param_results:
        print(f"{model_title}: 内側または外側でマージン条件を満たす組み合わせがありませんでした。")
        save_final_metrics_csv(metrics_filename, model_name, [], [], [], [], [], [])
        save_selected_fold_results_csv(selected_results_filename, model_name, [])
        return

    globally_valid_param_labels = [format_param_label(c, gamma) for c, gamma in valid_param_results.keys()]
    print(f"\n{model_title}: 全foldでマージン条件を満たした組み合わせ")
    print(globally_valid_param_labels)
    if dropped_param_labels:
        print(f"{model_title}: 除外された組み合わせ数 = {len(dropped_param_labels)}")

    selected_records = []
    for fold_idx in range(len(outer_splits)):
        best_record = None
        for records in valid_param_results.values():
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

    accuracies = [record["accuracy"] for record in selected_records]
    recalls = [record["recall"] for record in selected_records]
    precisions = [record["precision"] for record in selected_records]
    f1_scores = [record["f1_score"] for record in selected_records]
    best_cs = [record["c"] for record in selected_records]
    best_gammas = None
    if kernel == "rbf":
        best_gammas = [record["gamma"] for record in selected_records]

    print(f"\n{model_title} ダブルクロスバリデーションでの結果")
    print(sum(accuracies) / len(accuracies))
    print(sum(recalls) / len(recalls))
    print(sum(precisions) / len(precisions))
    print(sum(f1_scores) / len(f1_scores))
    print(best_cs)
    if best_gammas is not None:
        print(best_gammas)

    save_final_metrics_csv(
        metrics_filename,
        model_name,
        accuracies,
        recalls,
        precisions,
        f1_scores,
        best_cs,
        best_gammas,
    )
    save_selected_fold_results_csv(selected_results_filename, model_name, selected_records)


print("SVM実行前のデータ形状")
print(f"文書数: {X.shape[0]}")
print(f"特徴数: {X.shape[1]}")

LINEAR_C_VALUES = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
RBF_C_VALUES = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
RBF_GAMMA_VALUES = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]

run_margin_constrained_nested_cv(
    model_name="linear_svm",
    model_title="線形SVM",
    kernel="linear",
    param_grid=[(c, None) for c in LINEAR_C_VALUES],
    X=X,
    y=labels,
    document_names=document_names,
    metrics_filename="linear_svm_final_metrics.csv",
    selected_results_filename="linear_svm_selected_fold_results.csv",
)

run_margin_constrained_nested_cv(
    model_name="rbf_svm",
    model_title="非線形SVM",
    kernel="rbf",
    param_grid=[(c, gamma) for c in RBF_C_VALUES for gamma in RBF_GAMMA_VALUES],
    X=X,
    y=labels,
    document_names=document_names,
    metrics_filename="rbf_svm_final_metrics.csv",
    selected_results_filename="rbf_svm_selected_fold_results.csv",
)
