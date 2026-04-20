from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
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

import numpy as np
import pandas as pd

from dataset_loader import load_documents
from robust_margin_svm import run_robust_margin_nested_cv
from text_vectorizer import Tf_idf


TOP_N_VALUES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
MIN_LEN = 0
MIN_DF = 0.0
USE_STEMMING = True
KEEP_DIGITS = False
PERCENT_MODE = "drop"

RUN_LINEAR = True
RUN_RBF = True

LINEAR_C_VALUES = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
RBF_C_VALUES = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
RBF_GAMMA_VALUES = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
MARGIN_THRESHOLD = 0.9990
FEATURE_MODE_DIR_NAMES = {
    "binary": "binary",
    "count": "count",
}
KERNEL_DIR_NAMES = {
    "linear": "linear_svm",
    "rbf": "nonlinear_svm",
}
BEST_RESULT_COLUMNS = [
    "feature_count",
    "feature_mode",
    "kernel",
    "mean_accuracy",
    "mean_recall",
    "mean_precision",
    "mean_f1",
    "best_cs",
    "best_gammas",
]


def load_feature_source():
    documents_1, documents_0 = load_documents(
        real_scam=False,
        keep_digits=KEEP_DIGITS,
        percent_mode=PERCENT_MODE,
    )
    documents = documents_1 + documents_0

    tf_idf = Tf_idf(documents, True, MIN_LEN, use_stemming=USE_STEMMING)
    labels = np.array(tf_idf.labels(documents_1), dtype=int)
    X_tfidf, feature_names, _ = tf_idf.tf_idf(MIN_DF, ngram_range=(1, 1))
    X_count, count_feature_names, _ = tf_idf.term_frequency(MIN_DF, ngram_range=(1, 1))

    if list(feature_names) != list(count_feature_names):
        raise ValueError("TF-IDF と出現回数で語彙順が一致しません。")

    return X_tfidf.toarray(), X_count.toarray(), labels, np.array(feature_names), documents_1, documents_0


def build_doc_count_score_table(X, labels, feature_names):
    mask1 = labels == 1
    mask0 = labels == 0
    X_label1 = X[mask1]
    X_label0 = X[mask0]
    n_label1 = X_label1.shape[0]
    n_label0 = X_label0.shape[0]

    label1_doc_count = np.sum(X_label1 != 0, axis=0)
    label0_doc_count = np.sum(X_label0 != 0, axis=0)

    score_label1 = (label1_doc_count / n_label1) - (label0_doc_count / n_label0)
    score_label0 = (label0_doc_count / n_label0) - (label1_doc_count / n_label1)

    score_df = pd.DataFrame(
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

    return score_df


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


def build_binary_features(X, feature_names, label1_top, label0_top):
    selected_terms = list(label1_top["term"]) + list(label0_top["term"])
    selected_terms_unique = list(dict.fromkeys(selected_terms))
    vocab_index = {term: idx for idx, term in enumerate(feature_names)}
    selected_indices = [vocab_index[term] for term in selected_terms_unique]
    X_selected = X[:, selected_indices]
    X_binary = (X_selected != 0).astype(int)
    return X_binary, selected_terms_unique


def build_count_features(X_count, feature_names, label1_top, label0_top):
    selected_terms = list(label1_top["term"]) + list(label0_top["term"])
    selected_terms_unique = list(dict.fromkeys(selected_terms))
    vocab_index = {term: idx for idx, term in enumerate(feature_names)}
    selected_indices = [vocab_index[term] for term in selected_terms_unique]
    X_selected = X_count[:, selected_indices]
    return X_selected, selected_terms_unique


def get_feature_output_dirs(feature_count):
    feature_root_dir = SAVE_DIR / f"top{feature_count}"
    features_dir = feature_root_dir / "features"
    metrics_dir = feature_root_dir / "metrics"
    features_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return feature_root_dir, features_dir, metrics_dir


def get_condition_metrics_dir(metrics_root_dir, feature_mode, kernel):
    condition_dir = metrics_root_dir / FEATURE_MODE_DIR_NAMES[feature_mode] / KERNEL_DIR_NAMES[kernel]
    condition_dir.mkdir(parents=True, exist_ok=True)
    return condition_dir


def get_active_conditions():
    conditions = []
    if RUN_LINEAR:
        conditions.extend([("binary", "linear"), ("count", "linear")])
    if RUN_RBF:
        conditions.extend([("binary", "rbf"), ("count", "rbf")])
    return conditions


def save_feature_selection_outputs(score_df, label1_top, label0_top, selected_terms, features_dir):
    score_df.to_csv(features_dir / "doc_count_score_table.csv", index=False, encoding="utf-8-sig")
    label1_top.to_csv(features_dir / "top_label1_terms.csv", index=False, encoding="utf-8-sig")
    label0_top.to_csv(features_dir / "top_label0_terms.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"selected_term": selected_terms}).to_csv(
        features_dir / "selected_terms_combined.csv",
        index=False,
        encoding="utf-8-sig",
    )


def round_half_up(value, digits=3):
    if value in ("", None):
        return ""
    quantize_exp = Decimal("1").scaleb(-digits)
    return float(Decimal(str(value)).quantize(quantize_exp, rounding=ROUND_HALF_UP))


def build_summary_row(feature_count, feature_mode, kernel, result):
    row = {
        "feature_count": feature_count,
        "feature_mode": feature_mode,
        "kernel": kernel,
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
                "mean_accuracy": round_half_up(result["mean_accuracy"]),
                "mean_recall": round_half_up(result["mean_recall"]),
                "mean_precision": round_half_up(result["mean_precision"]),
                "mean_f1": round_half_up(result["mean_f1"]),
                "best_cs": ",".join(map(str, result.get("best_cs", []))),
                "best_gammas": ",".join(map(str, result.get("best_gammas", []))) if "best_gammas" in result else "",
            }
        )

    return row


def save_svm_summary_csv(summary_rows, output_path):
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def save_best_result_csv(best_row, output_path):
    if best_row is None:
        pd.DataFrame(columns=BEST_RESULT_COLUMNS).to_csv(output_path, index=False, encoding="utf-8-sig")
        return
    best_result_to_save = {column: best_row.get(column, "") for column in BEST_RESULT_COLUMNS}
    pd.DataFrame([best_result_to_save]).to_csv(output_path, index=False, encoding="utf-8-sig")


def format_condition_label(feature_mode, kernel):
    feature_mode_label = "バイナリ特徴" if feature_mode == "binary" else "出現頻度特徴"
    kernel_label = "線形SVM" if kernel == "linear" else "非線形SVM"
    return f"{feature_mode_label} / {kernel_label}"


def save_global_metrics_outputs(summary_rows_by_condition, best_results_by_condition):
    global_metrics_root = SAVE_DIR / "metrics"
    global_metrics_root.mkdir(parents=True, exist_ok=True)

    for feature_mode, kernel in get_active_conditions():
        condition_dir = get_condition_metrics_dir(global_metrics_root, feature_mode, kernel)
        save_svm_summary_csv(
            summary_rows_by_condition[(feature_mode, kernel)],
            condition_dir / "svm_feature_count_summary.csv",
        )
        save_best_result_csv(
            best_results_by_condition[(feature_mode, kernel)],
            condition_dir / "svm_best_result.csv",
        )


def print_result(result):
    if result is None:
        return
    if not result.get("selected_records"):
        print(f"\n{result['kernel']} SVM の結果")
        print(f"margin >= {result['margin_threshold']} を全foldで満たす組み合わせがありませんでした。")
        if "saved_csv_paths" in result:
            print(f"saved metrics csv: {result['saved_csv_paths']['metrics']}")
            print(f"saved selected-records csv: {result['saved_csv_paths']['selected_records']}")
            print(f"saved param-status csv: {result['saved_csv_paths']['param_status']}")
        return
    print(f"\n{result['kernel']} SVM の結果")
    print(f"accuracy : {result['mean_accuracy']:.4f}")
    print(f"recall   : {result['mean_recall']:.4f}")
    print(f"precision: {result['mean_precision']:.4f}")
    print(f"f1       : {result['mean_f1']:.4f}")
    print(f"selected C values: {result['best_cs']}")
    if "best_gammas" in result:
        print(f"selected gamma values: {result['best_gammas']}")
    print(f"globally valid params: {result['valid_param_labels']}")
    if "saved_csv_paths" in result:
        print(f"saved metrics csv: {result['saved_csv_paths']['metrics']}")
        print(f"saved selected-records csv: {result['saved_csv_paths']['selected_records']}")
        print(f"saved param-status csv: {result['saved_csv_paths']['param_status']}")


if __name__ == "__main__":
    X_source, X_count_source, labels, feature_names, documents_1, documents_0 = load_feature_source()
    active_conditions = get_active_conditions()
    summary_rows_by_condition = {condition: [] for condition in active_conditions}
    best_results_by_condition = {condition: None for condition in active_conditions}

    for feature_count in TOP_N_VALUES:
        _, features_dir, metrics_root_dir = get_feature_output_dirs(feature_count)
        score_df = build_doc_count_score_table(X_source, labels, feature_names)
        label1_top, label0_top = select_top_terms(score_df, feature_count)
        X_binary, selected_terms = build_binary_features(X_source, feature_names, label1_top, label0_top)
        X_count_selected, count_selected_terms = build_count_features(X_count_source, feature_names, label1_top, label0_top)

        save_feature_selection_outputs(score_df, label1_top, label0_top, selected_terms, features_dir)

        print("\n文書出現数差スコアによる特徴選択")
        print(f"指定した総特徴候補数: {feature_count}")
        print(f"ラベル1側の上位件数: {len(label1_top)}")
        print(f"ラベル0側の上位件数: {len(label0_top)}")
        print(f"重複除去後の特徴数: {X_binary.shape[1]}")
        print(f"SVM入力の文書数: {X_binary.shape[0]}")
        print("\nラベル1側の上位語")
        print(label1_top[["term", "label1_doc_count", "label0_doc_count", "label1_score"]].to_string(index=False))
        print("\nラベル0側の上位語")
        print(label0_top[["term", "label0_doc_count", "label1_doc_count", "label0_score"]].to_string(index=False))

        pd.DataFrame(X_binary, columns=selected_terms).assign(label=labels).to_csv(
            features_dir / "binary_feature_matrix.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(X_count_selected, columns=count_selected_terms).assign(label=labels).to_csv(
            features_dir / "count_feature_matrix.csv",
            index=False,
            encoding="utf-8-sig",
        )

        experiment_results = []

        if RUN_LINEAR:
            print("\nバイナリ特徴での linear SVM")
            linear_result = run_robust_margin_nested_cv(
                X=X_binary,
                y=labels,
                kernel="linear",
                c_values=LINEAR_C_VALUES,
                margin_threshold=MARGIN_THRESHOLD,
                output_dir=get_condition_metrics_dir(metrics_root_dir, "binary", "linear"),
                output_prefix="svm",
            )
            print_result(linear_result)
            experiment_results.append(("binary", "linear", linear_result))

            print("\n出現回数特徴での linear SVM")
            linear_count_result = run_robust_margin_nested_cv(
                X=X_count_selected,
                y=labels,
                kernel="linear",
                c_values=LINEAR_C_VALUES,
                margin_threshold=MARGIN_THRESHOLD,
                output_dir=get_condition_metrics_dir(metrics_root_dir, "count", "linear"),
                output_prefix="svm",
            )
            print_result(linear_count_result)
            experiment_results.append(("count", "linear", linear_count_result))

        if RUN_RBF:
            print("\nバイナリ特徴での rbf SVM")
            rbf_result = run_robust_margin_nested_cv(
                X=X_binary,
                y=labels,
                kernel="rbf",
                c_values=RBF_C_VALUES,
                gamma_values=RBF_GAMMA_VALUES,
                margin_threshold=MARGIN_THRESHOLD,
                output_dir=get_condition_metrics_dir(metrics_root_dir, "binary", "rbf"),
                output_prefix="svm",
            )
            print_result(rbf_result)
            experiment_results.append(("binary", "rbf", rbf_result))

            print("\n出現回数特徴での rbf SVM")
            rbf_count_result = run_robust_margin_nested_cv(
                X=X_count_selected,
                y=labels,
                kernel="rbf",
                c_values=RBF_C_VALUES,
                gamma_values=RBF_GAMMA_VALUES,
                margin_threshold=MARGIN_THRESHOLD,
                output_dir=get_condition_metrics_dir(metrics_root_dir, "count", "rbf"),
                output_prefix="svm",
            )
            print_result(rbf_count_result)
            experiment_results.append(("count", "rbf", rbf_count_result))

        for feature_mode, kernel, result in experiment_results:
            summary_row = build_summary_row(feature_count, feature_mode, kernel, result)
            summary_rows_by_condition[(feature_mode, kernel)].append(summary_row)

            if result.get("selected_records"):
                candidate_accuracy = result["mean_accuracy"]
                current_best_result = best_results_by_condition[(feature_mode, kernel)]
                if current_best_result is None or candidate_accuracy > current_best_result["raw_accuracy"]:
                    best_results_by_condition[(feature_mode, kernel)] = {
                        "feature_count": feature_count,
                        "feature_mode": feature_mode,
                        "kernel": kernel,
                        "mean_accuracy": round_half_up(result["mean_accuracy"]),
                        "mean_recall": round_half_up(result["mean_recall"]),
                        "mean_precision": round_half_up(result["mean_precision"]),
                        "mean_f1": round_half_up(result["mean_f1"]),
                        "best_cs": ",".join(map(str, result.get("best_cs", []))),
                        "best_gammas": ",".join(map(str, result.get("best_gammas", []))) if "best_gammas" in result else "",
                        "raw_accuracy": candidate_accuracy,
                    }

    save_global_metrics_outputs(summary_rows_by_condition, best_results_by_condition)

    any_best_result = False
    for feature_mode, kernel in active_conditions:
        best_result = best_results_by_condition[(feature_mode, kernel)]
        print(f"\n{format_condition_label(feature_mode, kernel)} の最高設定")
        if best_result is None:
            print("全foldでマージン条件を満たす設定がありませんでした。")
            continue
        any_best_result = True
        best_result_to_print = {k: v for k, v in best_result.items() if k != "raw_accuracy"}
        print(best_result_to_print)

    if not any_best_result:
        print("\n全条件でマージン条件を満たす設定がありませんでした。")
