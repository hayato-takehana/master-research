from __future__ import annotations

from pathlib import Path
import argparse
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")


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
OUTPUT_ROOT = Path(
    os.environ.get(
        "DIFFERENCES_BETWEEN_LABELS_OUTPUT_ROOT",
        r"D:\D_Student\HayatoTakehana",
    )
)
SAVE_DIR = OUTPUT_ROOT / "data" / "outputs" / "tf_delete_v1_predata" / "term_differences"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
redirect_relative_outputs(SAVE_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from dataset_loader import (  # noqa: E402
    PRE_RESEARCH_DELETE_V1_DATASET_NAMES,
    load_documents,
)
from text_vectorizer import Tf_idf  # noqa: E402


REAL_SCAM = False
KEEP_DIGITS = False
PERCENT_MODE = "drop"
MIN_LEN = 0
MIN_DF = 0.0
USE_STEMMING = True
TOP_N = 10


def configure_japanese_plot_font():
    """利用可能な日本語フォントをグラフ全体に設定する。"""
    available_font_names = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in (
        "Noto Sans JP",
        "Yu Gothic",
        "Meiryo",
        "MS Gothic",
        "Noto Sans CJK JP",
        "IPAexGothic",
        "IPAGothic",
    ):
        if font_name in available_font_names:
            plt.rcParams["font.family"] = font_name
            return


configure_japanese_plot_font()


def load_corpus():
    """delete_v1先行研究コーパスから文書本文・ラベル・文書IDを読み込む。"""
    documents_1, documents_0 = load_documents(
        real_scam=REAL_SCAM,
        keep_digits=KEEP_DIGITS,
        percent_mode=PERCENT_MODE,
        dataset_names=PRE_RESEARCH_DELETE_V1_DATASET_NAMES,
    )

    documents = np.array(documents_1 + documents_0, dtype=object)
    labels = np.array([1] * len(documents_1) + [0] * len(documents_0), dtype=int)
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
    """全データで語彙を学習し、出現回数行列と全単語リストを取得する。"""
    vectorizer = Tf_idf(list(documents), True, MIN_LEN, use_stemming=USE_STEMMING)
    X_count, feature_names, fitted_count_vectorizer = vectorizer.term_frequency(
        MIN_DF,
        ngram_range=(1, 1),
    )
    return (
        X_count.tocsr(),
        np.array(feature_names),
        fitted_count_vectorizer,
    )


def build_all_terms_table(feature_names):
    return pd.DataFrame(
        {
            "feature_index": np.arange(len(feature_names), dtype=int),
            "term": feature_names,
        }
    )


def _sparse_column_sums(matrix):
    return np.asarray(matrix.sum(axis=0)).ravel().astype(float)


def _document_lengths_from_counts(X_count):
    return np.asarray(X_count.sum(axis=1)).ravel().astype(float)


def _mean_document_length_normalized_values(X_by_label, doc_lengths):
    if X_by_label.shape[0] == 0:
        raise ValueError("Both label groups must contain at least one document.")

    inv_doc_lengths = np.divide(
        1.0,
        doc_lengths,
        out=np.zeros_like(doc_lengths, dtype=float),
        where=doc_lengths > 0,
    )
    normalized_values = X_by_label.multiply(inv_doc_lengths[:, None])
    return _sparse_column_sums(normalized_values) / X_by_label.shape[0]


def _log_normalized_frequency_matrix(X_count, document_word_counts):
    """非ゼロTFを `(1 + ln(freq)) / (1 + ln(doc_words))` へ変換する。"""
    document_word_counts = np.asarray(document_word_counts, dtype=float)
    denominators = np.ones_like(document_word_counts, dtype=float)
    positive_document_mask = document_word_counts > 0
    denominators[positive_document_mask] += np.log(
        document_word_counts[positive_document_mask]
    )
    inv_denominators = np.divide(
        1.0,
        denominators,
        out=np.zeros_like(denominators, dtype=float),
        where=positive_document_mask,
    )

    normalized_values = X_count.astype(float).copy()
    normalized_values.data = 1.0 + np.log(normalized_values.data)
    return normalized_values.multiply(inv_denominators[:, None]).tocsr()


def _log_frequency_numerator_matrix(X_count):
    """非ゼロTFを対数補正の分子 `1 + ln(freq)` へ変換する。"""
    numerator_values = X_count.astype(float).copy()
    numerator_values.data = 1.0 + np.log(numerator_values.data)
    return numerator_values.tocsr()


def _l2_normalize_sparse_rows(matrix):
    """全列を対象に、各文書行のL2ノルムが1になるよう正規化する。"""
    matrix = matrix.astype(float).tocsr()
    row_norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    inv_row_norms = np.divide(
        1.0,
        row_norms,
        out=np.zeros_like(row_norms, dtype=float),
        where=row_norms > 0,
    )
    return matrix.multiply(inv_row_norms[:, None]).tocsr()


def _build_metric_table(
    *,
    metric_id,
    metric_name,
    feature_names,
    label1_values,
    label0_values,
    label1_doc_count,
    label0_doc_count,
):
    signed_difference = label1_values - label0_values
    absolute_difference = np.abs(signed_difference)
    larger_label = np.where(label1_values >= label0_values, 1, 0)
    larger_value = np.maximum(label1_values, label0_values)

    return pd.DataFrame(
        {
            "metric_id": metric_id,
            "metric_name": metric_name,
            "feature_index": np.arange(len(feature_names), dtype=int),
            "term": feature_names,
            "label1_value": label1_values,
            "label0_value": label0_values,
            "signed_difference_label1_minus_label0": signed_difference,
            "absolute_difference": absolute_difference,
            "larger_label": larger_label,
            "larger_value": larger_value,
            "label1_doc_count": label1_doc_count,
            "label0_doc_count": label0_doc_count,
        }
    )


def build_frequency_difference_tables(X_count, feature_names, labels):
    label1_mask = labels == 1
    label0_mask = labels == 0
    label1_doc_count = int(np.sum(label1_mask))
    label0_doc_count = int(np.sum(label0_mask))
    if label1_doc_count == 0 or label0_doc_count == 0:
        raise ValueError("Both label 1 and label 0 documents are required.")

    label1_counts = X_count[label1_mask]
    label0_counts = X_count[label0_mask]
    label1_total_frequency = _sparse_column_sums(label1_counts)
    label0_total_frequency = _sparse_column_sums(label0_counts)

    metric1 = _build_metric_table(
        metric_id="total_frequency_difference",
        metric_name="abs(sum_freq_label1 - sum_freq_label0)",
        feature_names=feature_names,
        label1_values=label1_total_frequency,
        label0_values=label0_total_frequency,
        label1_doc_count=label1_doc_count,
        label0_doc_count=label0_doc_count,
    )

    metric2 = _build_metric_table(
        metric_id="mean_frequency_per_document_difference",
        metric_name="abs(sum_freq_label1 / n_label1 - sum_freq_label0 / n_label0)",
        feature_names=feature_names,
        label1_values=label1_total_frequency / label1_doc_count,
        label0_values=label0_total_frequency / label0_doc_count,
        label1_doc_count=label1_doc_count,
        label0_doc_count=label0_doc_count,
    )

    doc_lengths = _document_lengths_from_counts(X_count)
    metric3 = _build_metric_table(
        metric_id="mean_relative_frequency_per_document_difference",
        metric_name=(
            "abs(mean_i(freq_label1_i / doc_words_label1_i) "
            "- mean_i(freq_label0_i / doc_words_label0_i))"
        ),
        feature_names=feature_names,
        label1_values=_mean_document_length_normalized_values(
            label1_counts,
            doc_lengths[label1_mask],
        ),
        label0_values=_mean_document_length_normalized_values(
            label0_counts,
            doc_lengths[label0_mask],
        ),
        label1_doc_count=label1_doc_count,
        label0_doc_count=label0_doc_count,
    )

    X_log_normalized = _log_normalized_frequency_matrix(
        X_count,
        doc_lengths,
    )
    metric4 = _build_metric_table(
        metric_id="mean_log_normalized_frequency_per_document_difference",
        metric_name=(
            "abs(mean_i((1 + ln(freq_label1_i)) / "
            "(1 + ln(doc_words_label1_i))) "
            "- mean_i((1 + ln(freq_label0_i)) / "
            "(1 + ln(doc_words_label0_i))))"
        ),
        feature_names=feature_names,
        label1_values=(
            _sparse_column_sums(X_log_normalized[label1_mask])
            / label1_doc_count
        ),
        label0_values=(
            _sparse_column_sums(X_log_normalized[label0_mask])
            / label0_doc_count
        ),
        label1_doc_count=label1_doc_count,
        label0_doc_count=label0_doc_count,
    )

    X_tf_s = _l2_normalize_sparse_rows(X_count)
    metric5 = _build_metric_table(
        metric_id="mean_full_vocabulary_l2_frequency_difference",
        metric_name="abs(mean_i(TF_S_label1_i) - mean_i(TF_S_label0_i))",
        feature_names=feature_names,
        label1_values=(
            _sparse_column_sums(X_tf_s[label1_mask])
            / label1_doc_count
        ),
        label0_values=(
            _sparse_column_sums(X_tf_s[label0_mask])
            / label0_doc_count
        ),
        label1_doc_count=label1_doc_count,
        label0_doc_count=label0_doc_count,
    )

    X_log_tf_s = _l2_normalize_sparse_rows(
        _log_frequency_numerator_matrix(X_count)
    )
    metric6 = _build_metric_table(
        metric_id="mean_full_vocabulary_log_l2_frequency_difference",
        metric_name=(
            "abs(mean_i(log_TF_S_label1_i) "
            "- mean_i(log_TF_S_label0_i))"
        ),
        feature_names=feature_names,
        label1_values=(
            _sparse_column_sums(X_log_tf_s[label1_mask])
            / label1_doc_count
        ),
        label0_values=(
            _sparse_column_sums(X_log_tf_s[label0_mask])
            / label0_doc_count
        ),
        label1_doc_count=label1_doc_count,
        label0_doc_count=label0_doc_count,
    )

    return {
        "total_frequency_difference": metric1,
        "mean_frequency_per_document_difference": metric2,
        "mean_relative_frequency_per_document_difference": metric3,
        "mean_log_normalized_frequency_per_document_difference": metric4,
        "mean_full_vocabulary_l2_frequency_difference": metric5,
        "mean_full_vocabulary_log_l2_frequency_difference": metric6,
    }


def top_n_by_absolute_difference(metric_df, top_n):
    return (
        metric_df.sort_values(
            by=["absolute_difference", "term"],
            ascending=[False, True],
            kind="mergesort",
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def save_top_terms_histogram(top_df, output_path, title, ylabel):
    terms = top_df["term"].astype(str).to_numpy()
    label1_values = top_df["label1_value"].to_numpy(dtype=float)
    label0_values = top_df["label0_value"].to_numpy(dtype=float)
    x_positions = np.arange(len(terms))
    bar_width = 0.38
    fig_width = max(8.0, len(terms) * 0.65)

    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    ax.bar(
        x_positions - bar_width / 2,
        label1_values,
        bar_width,
        label="詐欺ラベル",
        color="#C04F15",
    )
    ax.bar(
        x_positions + bar_width / 2,
        label0_values,
        bar_width,
        label="詐欺でないラベル",
        color="#13501B",
    )
    ax.set_title(title)
    ax.set_xlabel("Term")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(terms, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_frequency_difference_outputs(metric_tables, output_dir, top_n):
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_metrics_df = pd.concat(metric_tables.values(), ignore_index=True)
    all_metrics_path = output_dir / "frequency_difference_metrics_all_terms.csv"
    combined_metrics_df.to_csv(all_metrics_path, index=False, encoding="utf-8-sig")

    output_paths = [all_metrics_path]
    top_tables = {}
    plot_settings = {
        "total_frequency_difference": (
            "term_frequency_1.csv",
            f"top_{top_n}_term_frequency_1.png",
            "Top Terms by Total Frequency Difference",
            "Total frequency",
        ),
        "mean_frequency_per_document_difference": (
            "term_frequency_document_count_2.csv",
            f"top_{top_n}_term_frequency_document_count_2.png",
            "Top Terms by Mean Frequency per Document Difference",
            "Mean frequency per document",
        ),
        "mean_relative_frequency_per_document_difference": (
            "term_frequency_sentence_length_3.csv",
            f"top_{top_n}_term_frequency_sentence_length_3.png",
            "Top Terms by Mean Relative Frequency per Document Difference",
            "Mean relative frequency per document",
        ),
        "mean_log_normalized_frequency_per_document_difference": (
            "term_frequency_log_normalized_4.csv",
            f"top_{top_n}_term_frequency_log_normalized_4.png",
            "Top Terms by Mean Log-Normalized Frequency Difference",
            "Mean log-normalized frequency per document",
        ),
        "mean_full_vocabulary_l2_frequency_difference": (
            "term_frequency_tf_s_5.csv",
            f"top_{top_n}_term_frequency_tf_s_5.png",
            "Top Terms by Mean Full-Vocabulary L2 Frequency Difference",
            "Mean TF_S per document",
        ),
        "mean_full_vocabulary_log_l2_frequency_difference": (
            "term_frequency_log_tf_s_6.csv",
            f"top_{top_n}_term_frequency_log_tf_s_6.png",
            "Top Terms by Mean Full-Vocabulary Log-L2 Frequency Difference",
            "Mean log-TF_S per document",
        ),
    }

    for metric_id, metric_df in metric_tables.items():
        top_df = top_n_by_absolute_difference(metric_df, top_n)
        top_tables[metric_id] = top_df

        top_csv_filename, plot_filename, title, ylabel = plot_settings[metric_id]
        top_csv_path = output_dir / top_csv_filename
        top_df.to_csv(top_csv_path, index=False, encoding="utf-8-sig")
        output_paths.append(top_csv_path)

        plot_path = output_dir / plot_filename
        save_top_terms_histogram(top_df, plot_path, title, ylabel)
        output_paths.append(plot_path)

    return output_paths, top_tables


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "term_analysis と同じデータ読み込み・前処理で全データをベクトル化し、"
            "全単語についてラベル間の出現頻度差を出力する。"
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SAVE_DIR,
        help=f"出力先の基点。直下に指標別フォルダを作成する。デフォルト: {SAVE_DIR}",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N,
        help=f"各指標で絶対差が大きい単語を出力・可視化する件数。デフォルト: {TOP_N}",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.top_n <= 0:
        raise ValueError("--top-n must be a positive integer.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frequency_output_dir = output_dir / "term_frequency"

    start_time = time.perf_counter()
    log_progress("[main] loading corpus")
    documents, labels, _ = load_corpus()
    log_progress(
        f"[main] documents={len(documents)}, label1={int(np.sum(labels == 1))}, label0={int(np.sum(labels == 0))}"
    )

    log_progress("[main] extracting all terms from count vectorizer")
    X_count, feature_names, _ = vectorize_all_documents(documents)
    log_progress(f"[main] total feature count: {len(feature_names)}")

    all_terms_df = build_all_terms_table(feature_names)
    all_terms_path = output_dir / "all_terms.csv"
    all_terms_df.to_csv(all_terms_path, index=False, encoding="utf-8-sig")

    log_progress("[main] calculating frequency-difference metrics")
    frequency_metric_tables = build_frequency_difference_tables(X_count, feature_names, labels)
    frequency_saved_paths, frequency_top_tables = save_frequency_difference_outputs(
        frequency_metric_tables,
        frequency_output_dir,
        args.top_n,
    )

    log_progress(f"\nsaved all terms: {all_terms_path}")
    for saved_path in frequency_saved_paths:
        log_progress(f"saved: {saved_path}")

    for metric_id, top_df in frequency_top_tables.items():
        log_progress(f"\n[{metric_id}] top {args.top_n}")
        display_columns = [
            "term",
            "label1_value",
            "label0_value",
            "signed_difference_label1_minus_label0",
            "absolute_difference",
            "larger_label",
        ]
        log_progress(top_df[display_columns].to_string(index=False))

    log_progress(f"elapsed: {format_elapsed_seconds(time.perf_counter() - start_time)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
