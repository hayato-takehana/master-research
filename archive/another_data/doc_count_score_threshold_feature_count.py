from __future__ import annotations

from pathlib import Path
import argparse
import os
import sys
import time


def _bootstrap_project_root() -> Path:
    """`active` と `archive` を含むプロジェクトルートを探索して import パスへ追加する。"""
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "active").exists() and (candidate / "archive").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise RuntimeError("Project root could not be found.")


PROJECT_ROOT = _bootstrap_project_root()

from project_runtime import bootstrap_project_paths, redirect_relative_outputs  # noqa: E402

bootstrap_project_paths(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
SAVE_DIR = PROJECT_ROOT / "data" / "outputs" / "another_data" / "doc_count_score_threshold_feature_counts"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
redirect_relative_outputs(SAVE_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from text_vectorizer import Tf_idf  # noqa: E402


ABS_SCORE_THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
MIN_LEN = 0
MIN_DF = 0.0
USE_STEMMING = True
FINAL_DATASET_PATH = PROJECT_ROOT / "Final_Dataset.csv"
TEXT_COLUMN = "Fillings"
LABEL_COLUMN = "Fraud"


def load_corpus():
    """Final_Dataset.csv から文書本文・ラベル・文書IDを読み込む。"""
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

    documents = dataset[TEXT_COLUMN].astype(str).to_numpy(dtype=object)
    labels = normalized_labels.map(label_map).to_numpy(dtype=int)
    doc_ids = np.arange(len(documents), dtype=int)

    return documents, labels, doc_ids


def log_progress(message):
    """長時間処理の進捗を即時表示する。"""
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


def vectorize_all_documents(documents):
    """全データで語彙を学習し、既存実験と同じ出現回数特徴行列を作る。"""
    vectorizer = Tf_idf(list(documents), True, MIN_LEN, use_stemming=USE_STEMMING)
    X_count, feature_names, fitted_count_vectorizer = vectorizer.term_frequency(
        MIN_DF,
        ngram_range=(1, 1),
    )
    return X_count.tocsr(), np.array(feature_names), fitted_count_vectorizer


def _nonzero_doc_counts(X):
    """各特徴語が「何文書に出たか」を列ごとに数える。"""
    if hasattr(X, "getnnz"):
        return np.asarray(X.getnnz(axis=0)).ravel()
    return np.asarray(np.sum(X != 0, axis=0)).ravel()


def build_doc_count_score_table(X, labels, feature_names):
    """文書出現率差にもとづく語の偏りスコア表を作る。

    `label1_score` は「label1 文書に出る割合 - label0 文書に出る割合」、
    `label0_score` はその逆向きの値である。
    """
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

    dominant_label = np.where(score_label1 > 0, "label1", np.where(score_label1 < 0, "label0", "neutral"))
    selected_score_source = np.where(
        score_label1 > 0,
        "label1_score",
        np.where(score_label1 < 0, "label0_score", ""),
    )
    selected_score = np.where(score_label1 > 0, score_label1, np.where(score_label1 < 0, score_label0, 0.0))

    return pd.DataFrame(
        {
            "term": feature_names,
            "label1_doc_count": label1_doc_count,
            "label0_doc_count": label0_doc_count,
            "label1_total_docs": n_label1,
            "label0_total_docs": n_label0,
            "label1_score": score_label1,
            "label0_score": score_label0,
            "abs_score": np.abs(score_label1),
            "dominant_label": dominant_label,
            "selected_score_source": selected_score_source,
            "selected_score": selected_score,
            "doc_count_diff_label1_minus_label0": label1_doc_count - label0_doc_count,
            "doc_count_diff_label0_minus_label1": label0_doc_count - label1_doc_count,
        }
    )


def sort_score_table(score_df):
    return score_df.sort_values(
        by=["abs_score", "selected_score", "label1_doc_count", "label0_doc_count", "term"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)


def build_threshold_feature_count_summary(score_df, score_thresholds):
    """指定した abs_score 閾値以上に残る特徴語数を集計する。"""
    rows = []
    for score_threshold in score_thresholds:
        selected = score_df.loc[score_df["abs_score"] >= score_threshold]
        label1_selected = selected.loc[selected["dominant_label"] == "label1"]
        label0_selected = selected.loc[selected["dominant_label"] == "label0"]
        rows.append(
            {
                "score_threshold": score_threshold,
                "selected_term_count": len(selected),
                "label1_selected_term_count": len(label1_selected),
                "label0_selected_term_count": len(label0_selected),
                "neutral_selected_term_count": len(selected.loc[selected["dominant_label"] == "neutral"]),
                "max_abs_score": score_df["abs_score"].max(),
                "min_selected_abs_score": selected["abs_score"].min() if len(selected) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_score_feature_count_curve(score_df):
    """全ユニーク abs_score を閾値にした場合の特徴語数を作る。"""
    score_for_curve_df = score_df.assign(abs_score_for_curve=score_df["abs_score"].round(12))
    counts_by_score = (
        score_for_curve_df.groupby(["abs_score_for_curve", "dominant_label"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={"label1": "label1_term_count_at_score", "label0": "label0_term_count_at_score"})
    )

    for column in ["label1_term_count_at_score", "label0_term_count_at_score", "neutral"]:
        if column not in counts_by_score.columns:
            counts_by_score[column] = 0

    curve = counts_by_score.sort_index(ascending=False).reset_index()
    curve = curve.rename(columns={"abs_score_for_curve": "abs_score", "neutral": "neutral_term_count_at_score"})
    curve["term_count_at_score"] = (
        curve["label1_term_count_at_score"]
        + curve["label0_term_count_at_score"]
        + curve["neutral_term_count_at_score"]
    )
    curve["selected_term_count_at_or_above_score"] = curve["term_count_at_score"].cumsum()
    curve["label1_selected_term_count_at_or_above_score"] = curve["label1_term_count_at_score"].cumsum()
    curve["label0_selected_term_count_at_or_above_score"] = curve["label0_term_count_at_score"].cumsum()
    curve["neutral_selected_term_count_at_or_above_score"] = curve["neutral_term_count_at_score"].cumsum()

    return curve[
        [
            "abs_score",
            "selected_term_count_at_or_above_score",
            "label1_selected_term_count_at_or_above_score",
            "label0_selected_term_count_at_or_above_score",
            "neutral_selected_term_count_at_or_above_score",
            "term_count_at_score",
            "label1_term_count_at_score",
            "label0_term_count_at_score",
            "neutral_term_count_at_score",
        ]
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Final_Dataset.csv 全体から既存 nested CV 実験と同じ方法で特徴を作成し、"
            "abs_score 閾値ごとの特徴数を出力する。"
        )
    )
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=ABS_SCORE_THRESHOLDS,
        help="確認したい abs_score 閾値。例: --thresholds 0.005 0.01 0.02",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SAVE_DIR,
        help=f"CSV の保存先。デフォルト: {SAVE_DIR}",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    log_progress("[main] loading corpus")
    documents, labels, _ = load_corpus()
    log_progress(
        f"[main] documents={len(documents)}, label1={int(np.sum(labels == 1))}, label0={int(np.sum(labels == 0))}"
    )

    log_progress("[main] building count features from all documents")
    X_count, feature_names, _ = vectorize_all_documents(documents)
    log_progress(f"[main] total feature count before score threshold: {len(feature_names)}")

    log_progress("[main] building document-count score table")
    score_df = sort_score_table(build_doc_count_score_table(X_count, labels, feature_names))
    summary_df = build_threshold_feature_count_summary(score_df, sorted(set(args.thresholds)))
    curve_df = build_score_feature_count_curve(score_df)

    score_table_path = output_dir / "all_term_doc_count_scores.csv"
    threshold_summary_path = output_dir / "threshold_feature_counts.csv"
    curve_path = output_dir / "score_feature_count_curve.csv"

    score_df.to_csv(score_table_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(threshold_summary_path, index=False, encoding="utf-8-sig")
    curve_df.to_csv(curve_path, index=False, encoding="utf-8-sig")

    log_progress("\n[threshold summary]")
    print(summary_df.to_string(index=False))
    log_progress(f"\nsaved all score table: {score_table_path}")
    log_progress(f"saved threshold summary: {threshold_summary_path}")
    log_progress(f"saved score/count curve: {curve_path}")
    log_progress(f"elapsed: {format_elapsed_seconds(time.perf_counter() - start_time)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
