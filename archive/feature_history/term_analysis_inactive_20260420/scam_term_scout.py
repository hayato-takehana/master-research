import os
import sys
from pathlib import Path


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
OUTPUT_DIR = get_output_dir(__file__, PROJECT_ROOT)
redirect_relative_outputs(OUTPUT_DIR)

import numpy as np
import pandas as pd

from dataset_loader import load_documents
from text_vectorizer import Tf_idf


RANKING_GROUPS = {
    "label1_focus": [
        ("tfidf_diff", "label1_mean_tfidf_minus_label0_mean_tfidf"),
        ("label1_mean_tfidf", "label1_mean_tfidf"),
        ("count_diff", "label1_total_count_minus_label0_total_count"),
        ("doc_diff", "label1_doc_count_minus_label0_doc_count"),
        ("label1_total_count", "label1_total_count"),
        ("label1_doc_count", "label1_doc_count"),
        ("mean_count_diff", "label1_mean_count_minus_label0_mean_count"),
        ("doc_rate_diff", "label1_doc_rate_minus_label0_doc_rate"),
    ],
    "label0_focus": [
        ("tfidf_diff", "label0_mean_tfidf_minus_label1_mean_tfidf"),
        ("label0_mean_tfidf", "label0_mean_tfidf"),
        ("count_diff", "label0_total_count_minus_label1_total_count"),
        ("doc_diff", "label0_doc_count_minus_label1_doc_count"),
        ("label0_total_count", "label0_total_count"),
        ("label0_doc_count", "label0_doc_count"),
        ("mean_count_diff", "label0_mean_count_minus_label1_mean_count"),
        ("doc_rate_diff", "label0_doc_rate_minus_label1_doc_rate"),
    ],
}


def _to_1d(array_like):
    if hasattr(array_like, "toarray"):
        return np.asarray(array_like.toarray()).ravel()
    return np.asarray(array_like).ravel()


def _compute_metrics(tfidf_matrix, count_matrix, labels, feature_names):
    labels = np.asarray(labels).astype(int)
    mask1 = labels == 1
    mask0 = labels == 0

    tfidf_label1 = tfidf_matrix[mask1]
    tfidf_label0 = tfidf_matrix[mask0]
    count_label1 = count_matrix[mask1]
    count_label0 = count_matrix[mask0]

    n_label1 = int(mask1.sum())
    n_label0 = int(mask0.sum())

    label1_mean_tfidf = _to_1d(tfidf_label1.mean(axis=0))
    label0_mean_tfidf = _to_1d(tfidf_label0.mean(axis=0))

    label1_total_count = _to_1d(count_label1.sum(axis=0))
    label0_total_count = _to_1d(count_label0.sum(axis=0))

    label1_doc_count = _to_1d((count_label1 > 0).sum(axis=0))
    label0_doc_count = _to_1d((count_label0 > 0).sum(axis=0))

    label1_mean_count = label1_total_count / max(n_label1, 1)
    label0_mean_count = label0_total_count / max(n_label0, 1)

    label1_doc_rate = label1_doc_count / max(n_label1, 1)
    label0_doc_rate = label0_doc_count / max(n_label0, 1)

    metrics_df = pd.DataFrame(
        {
            "term": feature_names,
            "label1_mean_tfidf": label1_mean_tfidf,
            "label0_mean_tfidf": label0_mean_tfidf,
            "label1_mean_tfidf_minus_label0_mean_tfidf": label1_mean_tfidf - label0_mean_tfidf,
            "label0_mean_tfidf_minus_label1_mean_tfidf": label0_mean_tfidf - label1_mean_tfidf,
            "label1_total_count": label1_total_count,
            "label0_total_count": label0_total_count,
            "label1_total_count_minus_label0_total_count": label1_total_count - label0_total_count,
            "label0_total_count_minus_label1_total_count": label0_total_count - label1_total_count,
            "label1_doc_count": label1_doc_count,
            "label0_doc_count": label0_doc_count,
            "label1_doc_count_minus_label0_doc_count": label1_doc_count - label0_doc_count,
            "label0_doc_count_minus_label1_doc_count": label0_doc_count - label1_doc_count,
            "label1_mean_count": label1_mean_count,
            "label0_mean_count": label0_mean_count,
            "label1_mean_count_minus_label0_mean_count": label1_mean_count - label0_mean_count,
            "label0_mean_count_minus_label1_mean_count": label0_mean_count - label1_mean_count,
            "label1_doc_rate": label1_doc_rate,
            "label0_doc_rate": label0_doc_rate,
            "label1_doc_rate_minus_label0_doc_rate": label1_doc_rate - label0_doc_rate,
            "label0_doc_rate_minus_label1_doc_rate": label0_doc_rate - label1_doc_rate,
        }
    )
    return metrics_df


def _save_rankings(metrics_df, output_dir, top_k, ranking_specs, focus_name):
    ranking_results = []

    for short_name, column_name in ranking_specs:
        ranked = (
            metrics_df.sort_values(by=column_name, ascending=False)
            .reset_index(drop=True)
            .copy()
        )
        ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
        top_df = ranked.head(top_k)
        top_df.to_csv(output_dir / f"top_{short_name}.csv", index=False, encoding="utf-8-sig")

        ranking_results.append(
            top_df[["rank", "term"]].assign(
                focus_name=focus_name,
                ranking_name=short_name,
                ranking_column=column_name,
            )
        )

    overlap_df = pd.concat(ranking_results, ignore_index=True)
    common_terms = (
        overlap_df.groupby("term")
        .agg(
            appearance_count=("ranking_name", "count"),
            ranking_names=("ranking_name", lambda values: ", ".join(sorted(values))),
            best_rank=("rank", "min"),
            average_rank=("rank", "mean"),
        )
        .reset_index()
        .sort_values(by=["appearance_count", "best_rank", "average_rank"], ascending=[False, True, True])
    )
    common_terms.to_csv(output_dir / "common_terms_across_rankings.csv", index=False, encoding="utf-8-sig")
    return common_terms


def analyze_ngram(documents, labels, ngram_size, min_df, top_k, output_root):
    tf_idf = Tf_idf(documents, True, 1)
    tfidf_matrix, feature_names, _ = tf_idf.tf_idf(min_df=min_df, ngram_range=(ngram_size, ngram_size))
    count_matrix, count_feature_names, _ = tf_idf.term_frequency(min_df=min_df, ngram_range=(ngram_size, ngram_size))

    if list(feature_names) != list(count_feature_names):
        raise ValueError(f"TF-IDF と出現回数で語彙順が一致しません: ngram={ngram_size}")

    ngram_dir = output_root / f"ngram_{ngram_size}"
    ngram_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = _compute_metrics(tfidf_matrix, count_matrix, labels, feature_names)
    metrics_df.insert(0, "ngram", ngram_size)
    metrics_df.to_csv(ngram_dir / "all_metrics.csv", index=False, encoding="utf-8-sig")

    common_terms_by_focus = []
    for focus_name, ranking_specs in RANKING_GROUPS.items():
        focus_dir = ngram_dir / focus_name
        focus_dir.mkdir(parents=True, exist_ok=True)
        common_terms = _save_rankings(
            metrics_df,
            focus_dir,
            top_k=top_k,
            ranking_specs=ranking_specs,
            focus_name=focus_name,
        )
        common_terms.insert(0, "ngram", ngram_size)
        common_terms.insert(1, "focus_name", focus_name)
        common_terms_by_focus.append(common_terms)

    return metrics_df, pd.concat(common_terms_by_focus, ignore_index=True)


if __name__ == "__main__":
    USE_REAL_SCAM = False
    KEEP_DIGITS = True
    PERCENT_MODE = "word"
    MIN_DF = 0.0
    MAX_NGRAM = 5
    TOP_K = 100

    print("文書を読み込みます...")
    document_label1, document_label0 = load_documents(
        real_scam=USE_REAL_SCAM,
        keep_digits=KEEP_DIGITS,
        percent_mode=PERCENT_MODE,
    )
    documents = document_label1 + document_label0
    labels = np.array([1] * len(document_label1) + [0] * len(document_label0), dtype=int)

    print(f"ラベル1文書数: {len(document_label1)}")
    print(f"ラベル0文書数: {len(document_label0)}")

    all_metrics = []
    all_common_terms = []

    for ngram_size in range(1, MAX_NGRAM + 1):
        print(f"\n=== {ngram_size}-gram を分析します ===")
        metrics_df, common_terms = analyze_ngram(
            documents=documents,
            labels=labels,
            ngram_size=ngram_size,
            min_df=MIN_DF,
            top_k=TOP_K,
            output_root=OUTPUT_DIR,
        )
        all_metrics.append(metrics_df)
        all_common_terms.append(common_terms)

    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    combined_metrics.to_csv(OUTPUT_DIR / "all_metrics_all_ngrams.csv", index=False, encoding="utf-8-sig")

    combined_common_terms = pd.concat(all_common_terms, ignore_index=True)
    combined_common_terms.to_csv(OUTPUT_DIR / "common_terms_across_rankings_all_ngrams.csv", index=False, encoding="utf-8-sig")

    overall_common_terms = (
        combined_common_terms.groupby(["focus_name", "term"])
        .agg(
            total_appearance_count=("appearance_count", "sum"),
            ngram_count=("ngram", "nunique"),
            ngrams=("ngram", lambda values: ", ".join(str(v) for v in sorted(set(values)))),
            best_rank=("best_rank", "min"),
        )
        .reset_index()
        .sort_values(by=["focus_name", "total_appearance_count", "ngram_count", "best_rank"], ascending=[True, False, False, True])
    )
    overall_common_terms.to_csv(
        OUTPUT_DIR / "overall_common_terms_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n分析が完了しました。")
    print(f"出力先: {OUTPUT_DIR}")
