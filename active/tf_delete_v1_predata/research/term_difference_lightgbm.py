from pathlib import Path
import concurrent.futures as futures
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
import multiprocessing as mp
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def read_positive_float_env(env_name, default_value):
    """正の数値環境変数を読み取る。無効値はデフォルトへ戻す。"""
    env_value = os.environ.get(env_name, "").strip()
    if not env_value:
        return default_value
    try:
        value = float(env_value)
    except ValueError:
        return default_value
    if value <= 0:
        return default_value
    return value


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

from project_runtime import bootstrap_project_paths, redirect_relative_outputs

bootstrap_project_paths(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
OUTPUT_ROOT = Path(
    os.environ.get(
        "DIFFERENCES_BETWEEN_LABELS_OUTPUT_ROOT",
        r"D:\D_Student\HayatoTakehana",
    )
)
SAVE_DIR = (
    OUTPUT_ROOT
    / "data"
    / "outputs"
    / "tf_delete_v1_predata"
    / "term_difference_lightgbm"
)
SAVE_DIR.mkdir(parents=True, exist_ok=True)
redirect_relative_outputs(SAVE_DIR)

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from dataset_loader import (
    PRE_RESEARCH_DELETE_V1_DATASET_NAMES,
    load_document_filenames,
    load_documents,
)
from text_vectorizer import Tf_idf

FEATURE_COUNT_CANDIDATES = list(range(10, 3001, 10))
RELATIVE_TERM_FREQUENCY_MODE = "term_frequency"
LOG_TERM_FREQUENCY_MODE = "log_term_frequency"
TF_S_MODE = "tf_s"
LOG_TF_S_MODE = "log_tf_s"
SELECTION_MODES = (
    RELATIVE_TERM_FREQUENCY_MODE,
)
COLUMN_MIN_MAX_FEATURE_MODE = "column_min_max"
ROW_L2_FEATURE_MODE = "row_l2"
FULL_VOCAB_ROW_L2_FEATURE_MODE = "full_vocab_row_l2"
COLUMN_MIN_MAX_DF_GT_2PCT_FEATURE_MODE = "column_min_max_df_gt_2pct"
ROW_L2_DF_GT_2PCT_FEATURE_MODE = "row_l2_df_gt_2pct"
FULL_VOCAB_ROW_L2_DF_GT_2PCT_FEATURE_MODE = (
    "full_vocab_row_l2_df_gt_2pct"
)
KERNELS = ("lightgbm",)
BASE_FEATURE_CONDITIONS = (
    (RELATIVE_TERM_FREQUENCY_MODE, COLUMN_MIN_MAX_FEATURE_MODE),
)
FILTERED_FEATURE_MODE_BY_BASE = {
    COLUMN_MIN_MAX_FEATURE_MODE: COLUMN_MIN_MAX_DF_GT_2PCT_FEATURE_MODE,
    ROW_L2_FEATURE_MODE: ROW_L2_DF_GT_2PCT_FEATURE_MODE,
    FULL_VOCAB_ROW_L2_FEATURE_MODE: (
        FULL_VOCAB_ROW_L2_DF_GT_2PCT_FEATURE_MODE
    ),
}
BASE_FEATURE_MODE_BY_MODE = {
    COLUMN_MIN_MAX_FEATURE_MODE: COLUMN_MIN_MAX_FEATURE_MODE,
    ROW_L2_FEATURE_MODE: ROW_L2_FEATURE_MODE,
    FULL_VOCAB_ROW_L2_FEATURE_MODE: FULL_VOCAB_ROW_L2_FEATURE_MODE,
    COLUMN_MIN_MAX_DF_GT_2PCT_FEATURE_MODE: COLUMN_MIN_MAX_FEATURE_MODE,
    ROW_L2_DF_GT_2PCT_FEATURE_MODE: ROW_L2_FEATURE_MODE,
    FULL_VOCAB_ROW_L2_DF_GT_2PCT_FEATURE_MODE: (
        FULL_VOCAB_ROW_L2_FEATURE_MODE
    ),
}
FEATURE_CONDITIONS = BASE_FEATURE_CONDITIONS
EXPERIMENT_CONDITIONS = tuple(
    (selection_mode, feature_mode, kernel)
    for selection_mode, feature_mode in FEATURE_CONDITIONS
    for kernel in KERNELS
)
MIN_LEN = 0
MIN_DF = 0.0
DOCUMENT_FREQUENCY_EXCLUSION_THRESHOLD = 0.02
USE_STEMMING = True
KEEP_DIGITS = False
PERCENT_MODE = "drop"
GLOBAL_TOP_MARGIN_MISCLASSIFIED_N = 10
GLOBAL_TOP_MARGIN_MISCLASSIFIED_FILENAME = "global_top_margin_misclassified.csv"
TERM_FREQUENCY_IQR_MULTIPLIER = 1.5
TERM_FREQUENCY_IQR_PLOT_TOP_N = 10

N_ESTIMATORS_VALUES = [20, 30, 50, 100, 200, 300, 500]
NUM_LEAVES_VALUES = [7, 15, 31, 63, 127, 200, 500]
MAX_DEPTH_VALUES = [-1, 1,  3, 5, 7, 10]
LEARNING_RATE_VALUES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
MARGIN_THRESHOLD = 0.9990
CANDIDATE_TIMEOUT_SECONDS = read_positive_float_env(
    "TERM_DIFFERENCE_CANDIDATE_TIMEOUT_SECONDS",
    8 * 60 * 60,
)
N_SPLITS = 10
RANDOM_STATE = 42
CLASS_WEIGHT = "balanced"
TOL = 1e-10
CANDIDATE_PROGRESS_INTERVAL = 10
SPLIT_CONTEXT_WORKERS_ENV = "TERM_DIFFERENCE_SPLIT_WORKERS"
DEFAULT_SPLIT_CONTEXT_WORKER_LIMIT = 20
FEATURE_COUNT_WORKERS_ENV = "TERM_DIFFERENCE_FEATURE_COUNT_WORKERS"
DEFAULT_FEATURE_COUNT_WORKER_LIMIT = 20

FEATURE_MODE_DIR_NAMES = {
    COLUMN_MIN_MAX_FEATURE_MODE: "minmax",
    ROW_L2_FEATURE_MODE: "row_l2",
    FULL_VOCAB_ROW_L2_FEATURE_MODE: "full_vocab_row_l2",
    COLUMN_MIN_MAX_DF_GT_2PCT_FEATURE_MODE: "minmax_df_gt_2pct",
    ROW_L2_DF_GT_2PCT_FEATURE_MODE: "row_l2_df_gt_2pct",
    FULL_VOCAB_ROW_L2_DF_GT_2PCT_FEATURE_MODE: (
        "full_vocab_row_l2_df_gt_2pct"
    ),
}
SELECTION_MODE_DIR_NAMES = {
    RELATIVE_TERM_FREQUENCY_MODE: "tf",
    LOG_TERM_FREQUENCY_MODE: "log_tf",
    TF_S_MODE: "tf_s",
    LOG_TF_S_MODE: "log_tf_s",
}
KERNEL_DIR_NAMES = {
    "lightgbm": "lightgbm",
}
FEATURE_COUNT_DIR_NAME = "fcnt"
METRICS_DIR_NAME = "m"
FEATURE_OUTPUT_DIR_NAME = "feat"
MISCLASSIFIED_OUTPUT_DIR_NAME = "mis"

FEATURE_COUNT_SUMMARY_COLUMNS = [
    "feature_count",
    "selection_mode",
    "feature_mode",
    "kernel",
    "mean_selected_term_count",
    "selected_term_counts",
    "mean_accuracy",
    "mean_recall",
    "mean_precision",
    "mean_f1",
    "best_cs",
    "best_gammas",
    "valid_param_count",
    "dropped_param_count",
    "timeout_param_count",
]
FEATURE_COUNT_BEST_RESULT_COLUMNS = [
    "feature_count",
    "selection_mode",
    "feature_mode",
    "kernel",
    "mean_selected_term_count",
    "selected_term_counts",
    "mean_accuracy",
    "mean_recall",
    "mean_precision",
    "mean_f1",
    "best_cs",
    "best_gammas",
]


def load_corpus():
    """文書本文・ラベル・文書IDを読み込み、後段で扱いやすい配列へまとめる。"""
    documents_1, documents_0 = load_documents(
        real_scam=False,
        keep_digits=KEEP_DIGITS,
        percent_mode=PERCENT_MODE,
        dataset_names=PRE_RESEARCH_DELETE_V1_DATASET_NAMES,
    )
    # `load_documents` と同じフォルダ走査順で PDF 名も取得し、doc_id と対応づける。
    pdf_names_1, pdf_names_0 = load_document_filenames(
        real_scam=False,
        sort_files=False,
        dataset_names=PRE_RESEARCH_DELETE_V1_DATASET_NAMES,
    )
    documents = np.array(documents_1 + documents_0, dtype=object)
    labels = np.array([1] * len(documents_1) + [0] * len(documents_0), dtype=int)
    doc_ids = np.arange(len(documents), dtype=int)
    pdf_names = np.array(pdf_names_1 + pdf_names_0, dtype=object)

    if len(pdf_names) != len(documents):
        raise RuntimeError(
            "pdf filename count does not match document count: "
            f"pdf_names={len(pdf_names)}, documents={len(documents)}"
        )

    return documents, labels, doc_ids, pdf_names


def log_progress(message):
    """長時間処理の進捗を即時表示する。"""
    print(message, flush=True)


def format_elapsed_seconds(elapsed_seconds):
    """経過秒数を人が読みやすい `h/m/s` 形式へ整形する。"""
    elapsed_seconds = int(round(elapsed_seconds))
    minutes, seconds = divmod(elapsed_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def vectorize_train_test_documents(train_docs, test_docs):
    """train 側で語彙を学習し、その語彙で train/test の出現回数行列を作る。

    ここで必ず train 側だけで vectorizer を fit することで、test 側の語彙が
    先に混ざる情報リークを防ぐ。
    """
    train_vectorizer = Tf_idf(list(train_docs), True, MIN_LEN, use_stemming=USE_STEMMING)
    X_train_count, feature_names, fitted_count_vectorizer = train_vectorizer.term_frequency(
        MIN_DF,
        ngram_range=(1, 1),
    )

    # test 文書は train で確定した語彙に射影するだけで、新しい語は追加しない。
    test_vectorizer = Tf_idf(list(test_docs), True, MIN_LEN, use_stemming=USE_STEMMING)
    X_test_count = fitted_count_vectorizer.transform(test_vectorizer.processed_docs)

    train_word_counts = np.array(
        [len(doc.split()) for doc in train_vectorizer.processed_docs],
        dtype=float,
    )
    test_word_counts = np.array(
        [len(doc.split()) for doc in test_vectorizer.processed_docs],
        dtype=float,
    )

    return (
        X_train_count.tocsr(),
        X_test_count.tocsr(),
        np.array(feature_names),
        train_word_counts,
        test_word_counts,
    )


def filter_count_matrices_by_document_frequency(
    X_train_count,
    X_test_count,
    feature_names,
    exclusion_threshold=DOCUMENT_FREQUENCY_EXCLUSION_THRESHOLD,
):
    """train内の文書出現割合が閾値以下の単語列をtrain/testから除外する。"""
    if X_train_count.shape[0] == 0:
        raise ValueError("Training documents are required.")

    document_counts = np.asarray(
        (X_train_count != 0).sum(axis=0)
    ).ravel()
    document_frequency_ratios = document_counts / X_train_count.shape[0]
    keep_mask = document_frequency_ratios > exclusion_threshold
    if not np.any(keep_mask):
        raise ValueError(
            "No terms remain after document-frequency filtering: "
            f"threshold={exclusion_threshold}"
        )

    return (
        X_train_count[:, keep_mask].tocsr(),
        X_test_count[:, keep_mask].tocsr(),
        np.asarray(feature_names)[keep_mask],
        document_frequency_ratios[keep_mask],
    )


def _uses_document_frequency_filter(feature_mode):
    return feature_mode in FILTERED_FEATURE_MODE_BY_BASE.values()


def _base_feature_mode(feature_mode):
    try:
        return BASE_FEATURE_MODE_BY_MODE[feature_mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}") from exc


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


def _build_difference_table(metric_id, metric_name, feature_names, label1_values, label0_values):
    signed_difference = label1_values - label0_values
    absolute_difference = np.abs(signed_difference)
    dominant_label = np.where(
        signed_difference > 0,
        "label1",
        np.where(signed_difference < 0, "label0", "neutral"),
    )

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
            "dominant_label": dominant_label,
            # 誤分類分析の既存列と互換性を保つため、差分値の別名も保持する。
            "selected_score_source": metric_id,
            "selected_score": absolute_difference,
            "abs_score": absolute_difference,
            "label1_score": signed_difference,
            "label0_score": -signed_difference,
        }
    )


def build_term_frequency_difference_table(
    X_count,
    labels,
    feature_names,
    document_word_counts=None,
):
    """文書長とラベル文書数を考慮した平均相対出現頻度差を計算する。"""
    label1_mask = labels == 1
    label0_mask = labels == 0
    if not np.any(label1_mask) or not np.any(label0_mask):
        raise ValueError("Both label 1 and label 0 documents are required.")

    if document_word_counts is None:
        doc_lengths = _document_lengths_from_counts(X_count)
    else:
        doc_lengths = np.asarray(document_word_counts, dtype=float)
        if len(doc_lengths) != X_count.shape[0]:
            raise ValueError(
                "document_word_counts must contain one value per document."
            )
    label1_values = _mean_document_length_normalized_values(
        X_count[label1_mask],
        doc_lengths[label1_mask],
    )
    label0_values = _mean_document_length_normalized_values(
        X_count[label0_mask],
        doc_lengths[label0_mask],
    )
    return _build_difference_table(
        "mean_relative_frequency_per_document_difference",
        (
            "abs(mean_i(freq_label1_i / doc_words_label1_i) "
            "- mean_i(freq_label0_i / doc_words_label0_i))"
        ),
        feature_names,
        label1_values,
        label0_values,
    )


def build_log_term_frequency_difference_table(
    X_count,
    labels,
    feature_names,
    document_word_counts,
):
    """対数補正TFのラベル別平均差を計算する。"""
    label1_mask = labels == 1
    label0_mask = labels == 0
    label1_doc_count = int(np.sum(label1_mask))
    label0_doc_count = int(np.sum(label0_mask))
    if label1_doc_count == 0 or label0_doc_count == 0:
        raise ValueError("Both label 1 and label 0 documents are required.")

    X_log_normalized = _log_normalized_frequency_matrix(
        X_count,
        document_word_counts,
    )
    label1_values = (
        _sparse_column_sums(X_log_normalized[label1_mask])
        / label1_doc_count
    )
    label0_values = (
        _sparse_column_sums(X_log_normalized[label0_mask])
        / label0_doc_count
    )
    return _build_difference_table(
        "mean_log_normalized_frequency_per_document_difference",
        (
            "abs(mean_i((1 + ln(freq_label1_i)) / "
            "(1 + ln(doc_words_label1_i))) "
            "- mean_i((1 + ln(freq_label0_i)) / "
            "(1 + ln(doc_words_label0_i))))"
        ),
        feature_names,
        label1_values,
        label0_values,
    )


def build_tf_s_difference_table(X_count, labels, feature_names):
    """全語彙で行L2正規化したTF_Sのラベル別平均差を計算する。"""
    label1_mask = labels == 1
    label0_mask = labels == 0
    label1_doc_count = int(np.sum(label1_mask))
    label0_doc_count = int(np.sum(label0_mask))
    if label1_doc_count == 0 or label0_doc_count == 0:
        raise ValueError("Both label 1 and label 0 documents are required.")

    X_tf_s = _l2_normalize_sparse_rows(X_count)
    label1_values = _sparse_column_sums(X_tf_s[label1_mask]) / label1_doc_count
    label0_values = _sparse_column_sums(X_tf_s[label0_mask]) / label0_doc_count
    return _build_difference_table(
        "mean_full_vocabulary_l2_frequency_difference",
        "abs(mean_i(TF_S_label1_i) - mean_i(TF_S_label0_i))",
        feature_names,
        label1_values,
        label0_values,
    )


def build_log_tf_s_difference_table(X_count, labels, feature_names):
    """全語彙の対数補正TFを行L2正規化したlog-TF_Sの平均差を計算する。"""
    label1_mask = labels == 1
    label0_mask = labels == 0
    label1_doc_count = int(np.sum(label1_mask))
    label0_doc_count = int(np.sum(label0_mask))
    if label1_doc_count == 0 or label0_doc_count == 0:
        raise ValueError("Both label 1 and label 0 documents are required.")

    X_log_tf_s = _l2_normalize_sparse_rows(
        _log_frequency_numerator_matrix(X_count)
    )
    label1_values = (
        _sparse_column_sums(X_log_tf_s[label1_mask])
        / label1_doc_count
    )
    label0_values = (
        _sparse_column_sums(X_log_tf_s[label0_mask])
        / label0_doc_count
    )
    return _build_difference_table(
        "mean_full_vocabulary_log_l2_frequency_difference",
        "abs(mean_i(log_TF_S_label1_i) - mean_i(log_TF_S_label0_i))",
        feature_names,
        label1_values,
        label0_values,
    )


def sort_difference_table(difference_df):
    return difference_df.sort_values(
        by=["absolute_difference", "term"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _select_top_unique_terms(difference_df, feature_count, excluded_terms=None):
    excluded_terms = set() if excluded_terms is None else set(excluded_terms)
    available_df = difference_df.loc[~difference_df["term"].isin(excluded_terms)]
    if len(available_df) < feature_count:
        raise ValueError(
            f"Not enough unique terms: requested={feature_count}, available={len(available_df)}"
        )
    return available_df.head(feature_count).copy().reset_index(drop=True)


def _select_frequency_terms(difference_df, feature_count, selection_mode):
    if feature_count <= 0:
        raise ValueError("feature_count must be positive.")

    selected_terms_df = _select_top_unique_terms(
        difference_df,
        feature_count,
    )
    selected_terms_df["selection_source"] = selection_mode
    selected_terms_df["selection_rank_within_source"] = np.arange(
        1,
        feature_count + 1,
    )
    selected_terms_df.insert(0, "selection_mode", selection_mode)
    selected_terms_df.insert(1, "selection_rank", np.arange(1, len(selected_terms_df) + 1))
    return selected_terms_df


def select_term_frequency_terms(term_frequency_df, feature_count):
    """相対出現頻度差の絶対値が大きい上位語を採用する。"""
    return _select_frequency_terms(
        term_frequency_df,
        feature_count,
        RELATIVE_TERM_FREQUENCY_MODE,
    )


def select_log_term_frequency_terms(log_term_frequency_df, feature_count):
    """対数補正TF差の絶対値が大きい上位語を採用する。"""
    return _select_frequency_terms(
        log_term_frequency_df,
        feature_count,
        LOG_TERM_FREQUENCY_MODE,
    )


def select_tf_s_terms(tf_s_df, feature_count):
    """TF_Sのラベル平均差が大きい上位語を採用する。"""
    return _select_frequency_terms(
        tf_s_df,
        feature_count,
        TF_S_MODE,
    )


def select_log_tf_s_terms(log_tf_s_df, feature_count):
    """log-TF_Sのラベル平均差が大きい上位語を採用する。"""
    return _select_frequency_terms(
        log_tf_s_df,
        feature_count,
        LOG_TF_S_MODE,
    )


def _relative_frequency_matrix(X_count, document_word_counts):
    document_word_counts = np.asarray(document_word_counts, dtype=float)
    inv_word_counts = np.divide(
        1.0,
        document_word_counts,
        out=np.zeros_like(document_word_counts, dtype=float),
        where=document_word_counts > 0,
    )
    return X_count.multiply(inv_word_counts[:, None]).tocsr()


def calculate_iqr_outlier_statistics(values, multiplier=TERM_FREQUENCY_IQR_MULTIPLIER):
    """1次元の値について、TukeyのIQR法による外れ値統計を返す。"""
    values = np.asarray(values, dtype=float).ravel()
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return {
            "sample_count": 0,
            "nonzero_count": 0,
            "q1": np.nan,
            "q3": np.nan,
            "iqr": np.nan,
            "lower_bound": np.nan,
            "upper_bound": np.nan,
            "outlier_count": 0,
            "outlier_rate": np.nan,
            "has_outlier": False,
            "max_value": np.nan,
            "outlier_mask": np.zeros(values.shape, dtype=bool),
        }

    q1, q3 = np.percentile(finite_values, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    finite_mask = np.isfinite(values)
    outlier_mask = finite_mask & (
        (values < lower_bound) | (values > upper_bound)
    )
    outlier_count = int(np.sum(outlier_mask))

    return {
        "sample_count": int(finite_values.size),
        "nonzero_count": int(np.count_nonzero(finite_values)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "outlier_count": outlier_count,
        "outlier_rate": float(outlier_count / finite_values.size),
        "has_outlier": outlier_count > 0,
        "max_value": float(np.max(finite_values)),
        "outlier_mask": outlier_mask,
    }


def _term_frequency_iqr_column_names():
    columns = ["tf_iqr_multiplier"]
    statistic_names = (
        "sample_count",
        "nonzero_count",
        "q1",
        "q3",
        "iqr",
        "lower_bound",
        "upper_bound",
        "outlier_count",
        "outlier_rate",
        "has_outlier",
        "max_value",
    )
    for group_name in ("all", "label1", "label0"):
        columns.extend(
            f"tf_{group_name}_{statistic_name}"
            for statistic_name in statistic_names
        )
    return columns


def add_term_frequency_iqr_diagnostics(
    selected_terms_df,
    X_train_count,
    train_labels,
    feature_names,
    train_word_counts,
    multiplier=TERM_FREQUENCY_IQR_MULTIPLIER,
):
    """TF選択語へ、ゼロ出現も含む学習文書内の相対頻度IQR統計を付与する。"""
    diagnosed_df = selected_terms_df.copy()
    for column in _term_frequency_iqr_column_names():
        if column.endswith("_has_outlier"):
            diagnosed_df[column] = pd.Series(
                pd.NA,
                index=diagnosed_df.index,
                dtype="boolean",
            )
        else:
            diagnosed_df[column] = np.nan

    tf_row_mask = (
        diagnosed_df["selection_source"] == RELATIVE_TERM_FREQUENCY_MODE
    )
    if not tf_row_mask.any():
        return diagnosed_df

    feature_index = {str(term): index for index, term in enumerate(feature_names)}
    tf_terms = diagnosed_df.loc[tf_row_mask, "term"].astype(str).tolist()
    missing_terms = [term for term in tf_terms if term not in feature_index]
    if missing_terms:
        raise KeyError(
            "TF selected terms are missing from count vocabulary: "
            f"{missing_terms[:5]}"
        )

    term_indices = [feature_index[term] for term in tf_terms]
    relative_values = _relative_frequency_matrix(
        X_train_count,
        train_word_counts,
    )[:, term_indices].toarray()
    train_labels = np.asarray(train_labels, dtype=int)
    group_masks = {
        "all": np.ones(len(train_labels), dtype=bool),
        "label1": train_labels == 1,
        "label0": train_labels == 0,
    }

    for term_position, row_index in enumerate(diagnosed_df.index[tf_row_mask]):
        diagnosed_df.at[row_index, "tf_iqr_multiplier"] = multiplier
        term_values = relative_values[:, term_position]
        for group_name, group_mask in group_masks.items():
            statistics = calculate_iqr_outlier_statistics(
                term_values[group_mask],
                multiplier,
            )
            for statistic_name, statistic_value in statistics.items():
                if statistic_name == "outlier_mask":
                    continue
                diagnosed_df.at[
                    row_index,
                    f"tf_{group_name}_{statistic_name}",
                ] = statistic_value

    return diagnosed_df


def build_term_frequency_iqr_outlier_rows(
    selected_terms_df,
    X_train_count,
    train_labels,
    train_doc_ids,
    feature_names,
    train_word_counts,
    outer_fold,
    feature_count,
    multiplier=TERM_FREQUENCY_IQR_MULTIPLIER,
):
    """TF選択語のラベル別IQR外れ値を、文書単位の行へ展開する。"""
    tf_rows = selected_terms_df.loc[
        selected_terms_df["selection_source"] == RELATIVE_TERM_FREQUENCY_MODE
    ].drop_duplicates(subset=["term"])
    if tf_rows.empty:
        return []

    feature_index = {str(term): index for index, term in enumerate(feature_names)}
    train_labels = np.asarray(train_labels, dtype=int)
    train_doc_ids = np.asarray(train_doc_ids, dtype=int)
    tf_terms = tf_rows["term"].astype(str).tolist()
    missing_terms = [term for term in tf_terms if term not in feature_index]
    if missing_terms:
        raise KeyError(
            "TF selected terms are missing from count vocabulary: "
            f"{missing_terms[:5]}"
        )
    term_indices = [feature_index[term] for term in tf_terms]
    relative_values = _relative_frequency_matrix(
        X_train_count,
        train_word_counts,
    )[:, term_indices].toarray()
    outlier_rows = []

    for term_position, selected_row in enumerate(tf_rows.itertuples(index=False)):
        term = str(selected_row.term)
        term_values = relative_values[:, term_position]
        for label in (1, 0):
            group_indices = np.flatnonzero(train_labels == label)
            statistics = calculate_iqr_outlier_statistics(
                term_values[group_indices],
                multiplier,
            )
            outlier_group_indices = np.flatnonzero(statistics["outlier_mask"])
            for group_position in outlier_group_indices:
                document_position = group_indices[group_position]
                value = float(term_values[document_position])
                outlier_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "feature_count": feature_count,
                        "selection_mode": selected_row.selection_mode,
                        "selection_rank": selected_row.selection_rank,
                        "term": term,
                        "label": label,
                        "doc_id": int(train_doc_ids[document_position]),
                        "relative_frequency": value,
                        "q1": statistics["q1"],
                        "q3": statistics["q3"],
                        "iqr": statistics["iqr"],
                        "lower_bound": statistics["lower_bound"],
                        "upper_bound": statistics["upper_bound"],
                        "outlier_direction": (
                            "low" if value < statistics["lower_bound"] else "high"
                        ),
                    }
                )

    return outlier_rows


def save_term_frequency_iqr_boxplot(
    outer_context,
    selected_terms_df,
    output_path,
    outer_fold,
    top_n=TERM_FREQUENCY_IQR_PLOT_TOP_N,
):
    """TF上位語の文書別相対出現頻度を、ラベル別箱ひげ図で保存する。"""
    plot_terms_df = (
        selected_terms_df.loc[
            selected_terms_df["selection_source"]
            == RELATIVE_TERM_FREQUENCY_MODE
        ]
        .sort_values("selection_rank", kind="mergesort")
        .drop_duplicates(subset=["term"])
        .head(top_n)
    )
    if plot_terms_df.empty:
        return None

    feature_index = {
        str(term): index
        for index, term in enumerate(outer_context.feature_names)
    }
    terms = plot_terms_df["term"].astype(str).tolist()
    term_indices = [feature_index[term] for term in terms]
    relative_values = _relative_frequency_matrix(
        outer_context.X_train_count,
        outer_context.train_word_counts,
    )[:, term_indices].toarray()
    train_labels = np.asarray(outer_context.train_labels, dtype=int)

    box_values = []
    box_positions = []
    y_positions = np.arange(1, len(terms) + 1, dtype=float)
    for term_position, y_position in enumerate(y_positions):
        box_values.extend(
            [
                relative_values[train_labels == 1, term_position],
                relative_values[train_labels == 0, term_position],
            ]
        )
        box_positions.extend([y_position - 0.18, y_position + 0.18])

    fig_height = max(6.0, len(terms) * 0.65)
    fig, ax = plt.subplots(figsize=(11.0, fig_height))
    boxplot = ax.boxplot(
        box_values,
        positions=box_positions,
        widths=0.30,
        vert=False,
        patch_artist=True,
        whis=TERM_FREQUENCY_IQR_MULTIPLIER,
        showfliers=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "#333333",
            "markersize": 3,
        },
        flierprops={
            "marker": "o",
            "markerfacecolor": "#333333",
            "markeredgecolor": "none",
            "markersize": 2.5,
            "alpha": 0.45,
        },
        medianprops={"color": "#111111", "linewidth": 1.2},
    )
    label_colors = ("#C04F15", "#13501B")
    for box_index, box in enumerate(boxplot["boxes"]):
        box.set_facecolor(label_colors[box_index % 2])
        box.set_alpha(0.72)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [
            f"{int(rank)}. {term}"
            for rank, term in zip(plot_terms_df["selection_rank"], terms)
        ]
    )
    ax.invert_yaxis()
    ax.set_title(
        "TF relative-frequency boxplots "
        f"(outer fold {outer_fold}, whiskers={TERM_FREQUENCY_IQR_MULTIPLIER} x IQR)"
    )
    ax.set_xlabel("Relative term frequency in each training document")
    ax.set_ylabel("TF difference rank and term")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(
        handles=[
            Patch(facecolor=label_colors[0], alpha=0.72, label="Label 1"),
            Patch(facecolor=label_colors[1], alpha=0.72, label="Label 0"),
        ],
        loc="lower right",
    )
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def save_term_frequency_iqr_summary_plot(feature_count_summary_df, output_path):
    """特徴語数ごとのIQR外れ値率とIQR=0率を2段の折れ線図で保存する。"""
    summary_df = feature_count_summary_df.sort_values("feature_count")
    evaluated_counts = summary_df["evaluated_term_occurrences"].to_numpy(dtype=float)
    feature_counts = summary_df["feature_count"].to_numpy(dtype=int)

    label1_outlier_rate = (
        summary_df["terms_with_label1_outlier"].to_numpy(dtype=float)
        / evaluated_counts
    )
    label0_outlier_rate = (
        summary_df["terms_with_label0_outlier"].to_numpy(dtype=float)
        / evaluated_counts
    )
    either_outlier_rate = summary_df["any_label_outlier_rate"].to_numpy(dtype=float)
    label1_zero_iqr_rate = summary_df["label1_zero_iqr_rate"].to_numpy(dtype=float)
    label0_zero_iqr_rate = summary_df["label0_zero_iqr_rate"].to_numpy(dtype=float)
    both_zero_iqr_rate = (
        summary_df["both_labels_zero_iqr_term_occurrences"].to_numpy(dtype=float)
        / evaluated_counts
    )

    fig, (outlier_ax, zero_iqr_ax) = plt.subplots(
        2,
        1,
        figsize=(10.5, 8.0),
        sharex=True,
    )
    outlier_ax.plot(
        feature_counts,
        either_outlier_rate * 100,
        marker="o",
        label="Either label",
        color="#333333",
    )
    outlier_ax.plot(
        feature_counts,
        label1_outlier_rate * 100,
        marker="o",
        label="Label 1",
        color="#C04F15",
    )
    outlier_ax.plot(
        feature_counts,
        label0_outlier_rate * 100,
        marker="o",
        label="Label 0",
        color="#13501B",
    )
    outlier_ax.set_title("Terms containing 1.5 x IQR outliers across outer folds")
    outlier_ax.set_ylabel("Term occurrences with outliers (%)")
    outlier_ax.set_ylim(0, 105)
    outlier_ax.grid(alpha=0.25)
    outlier_ax.legend()

    zero_iqr_ax.plot(
        feature_counts,
        label1_zero_iqr_rate * 100,
        marker="o",
        label="Label 1",
        color="#C04F15",
    )
    zero_iqr_ax.plot(
        feature_counts,
        label0_zero_iqr_rate * 100,
        marker="o",
        label="Label 0",
        color="#13501B",
    )
    zero_iqr_ax.plot(
        feature_counts,
        both_zero_iqr_rate * 100,
        marker="o",
        label="Both labels",
        color="#6A3D9A",
    )
    zero_iqr_ax.set_title("Terms with IQR = 0")
    zero_iqr_ax.set_xlabel("Number of top TF terms")
    zero_iqr_ax.set_ylabel("Term occurrences with IQR = 0 (%)")
    zero_iqr_ax.set_xticks(feature_counts)
    zero_iqr_ax.grid(alpha=0.25)
    zero_iqr_ax.legend()

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _min_max_scale_columns_from_train(train_values, test_values):
    """train側だけで列ごとのmin/maxを決め、testは同じ値で変換する。"""
    train_values = np.asarray(train_values, dtype=float)
    test_values = np.asarray(test_values, dtype=float)
    train_min = np.min(train_values, axis=0)
    train_max = np.max(train_values, axis=0)
    ranges = train_max - train_min

    scaled_train = np.divide(
        train_values - train_min,
        ranges,
        out=np.zeros_like(train_values, dtype=float),
        where=ranges > 0,
    )
    scaled_test = np.divide(
        test_values - train_min,
        ranges,
        out=np.zeros_like(test_values, dtype=float),
        where=ranges > 0,
    )
    # train範囲を超えるtest値も0〜1範囲に収める。
    return (
        np.clip(scaled_train, 0.0, 1.0),
        np.clip(scaled_test, 0.0, 1.0),
        train_min,
        train_max,
    )


def _log_frequency_numerator_matrix(X_count):
    """非ゼロTFを対数補正の分子 `1 + ln(freq)` へ変換する。"""
    numerator_values = X_count.astype(float).copy()
    numerator_values.data = 1.0 + np.log(numerator_values.data)
    return numerator_values.tocsr()


def _l2_normalize_rows(values):
    """各文書行のL2ノルムが1になるよう正規化する。ゼロ行はゼロのまま残す。"""
    values = np.asarray(values, dtype=float)
    row_norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(
        values,
        row_norms,
        out=np.zeros_like(values, dtype=float),
        where=row_norms > 0,
    )


def build_selected_feature_matrices(
    X_train_count,
    X_test_count,
    count_feature_names,
    selected_terms_df,
    train_word_counts,
    test_word_counts,
):
    """選択語について、列Min-Max特徴と分子のみの行L2特徴を作る。"""
    selected_terms = selected_terms_df["term"].tolist()
    selection_modes = selected_terms_df["selection_mode"].unique()
    if len(selection_modes) != 1:
        raise ValueError("selected_terms_df must contain exactly one selection mode.")
    selection_mode = selection_modes[0]

    count_vocab_index = {term: idx for idx, term in enumerate(count_feature_names)}
    term_indices = [
        count_vocab_index[term]
        for term in selected_terms
    ]
    if selection_mode in (TF_S_MODE, LOG_TF_S_MODE):
        if selection_mode == TF_S_MODE:
            X_train_full_vocab_source = X_train_count
            X_test_full_vocab_source = X_test_count
            feature_suffix = "tf_s"
            transform_name = "full_vocabulary_raw_frequency_row_l2"
        else:
            X_train_full_vocab_source = _log_frequency_numerator_matrix(
                X_train_count
            )
            X_test_full_vocab_source = _log_frequency_numerator_matrix(
                X_test_count
            )
            feature_suffix = "log_tf_s"
            transform_name = "full_vocabulary_log_frequency_row_l2"

        X_train_tf_s = _l2_normalize_sparse_rows(
            X_train_full_vocab_source
        )
        X_test_tf_s = _l2_normalize_sparse_rows(
            X_test_full_vocab_source
        )
        selected_feature_names = [
            f"{term}__{feature_suffix}"
            for term in selected_terms
        ]
        return {
            "selected_terms": selected_terms,
            "feature_metadata": {
                FULL_VOCAB_ROW_L2_FEATURE_MODE: {
                    "selected_feature_names": selected_feature_names,
                    "feature_train_min": np.full(len(selected_terms), np.nan),
                    "feature_train_max": np.full(len(selected_terms), np.nan),
                    "feature_transform": np.full(
                        len(selected_terms),
                        transform_name,
                        dtype=object,
                    ),
                },
            },
            FULL_VOCAB_ROW_L2_FEATURE_MODE: {
                "train": X_train_tf_s[:, term_indices].toarray(),
                "test": X_test_tf_s[:, term_indices].toarray(),
            },
        }

    if selection_mode == RELATIVE_TERM_FREQUENCY_MODE:
        X_train_min_max_source = _relative_frequency_matrix(
            X_train_count,
            train_word_counts,
        )
        X_test_min_max_source = _relative_frequency_matrix(
            X_test_count,
            test_word_counts,
        )
        X_train_l2_source = X_train_count
        X_test_l2_source = X_test_count
        feature_suffix = "tf"
        min_max_transform_name = "relative_frequency_column_min_max"
        l2_transform_name = "raw_frequency_row_l2"
    elif selection_mode == LOG_TERM_FREQUENCY_MODE:
        X_train_min_max_source = _log_normalized_frequency_matrix(
            X_train_count,
            train_word_counts,
        )
        X_test_min_max_source = _log_normalized_frequency_matrix(
            X_test_count,
            test_word_counts,
        )
        X_train_l2_source = _log_frequency_numerator_matrix(X_train_count)
        X_test_l2_source = _log_frequency_numerator_matrix(X_test_count)
        feature_suffix = "log_tf"
        min_max_transform_name = "log_normalized_frequency_column_min_max"
        l2_transform_name = "log_frequency_numerator_row_l2"
    else:
        raise ValueError(f"Unsupported selection_mode: {selection_mode}")

    train_min_max_values = X_train_min_max_source[:, term_indices].toarray()
    test_min_max_values = X_test_min_max_source[:, term_indices].toarray()
    (
        X_train_min_max_features,
        X_test_min_max_features,
        train_mins,
        train_maxs,
    ) = (
        _min_max_scale_columns_from_train(
            train_min_max_values,
            test_min_max_values,
        )
    )
    X_train_l2_features = _l2_normalize_rows(
        X_train_l2_source[:, term_indices].toarray()
    )
    X_test_l2_features = _l2_normalize_rows(
        X_test_l2_source[:, term_indices].toarray()
    )

    selected_feature_names = [
        f"{term}__{feature_suffix}"
        for term in selected_terms
    ]

    return {
        "selected_terms": selected_terms,
        "feature_metadata": {
            COLUMN_MIN_MAX_FEATURE_MODE: {
                "selected_feature_names": selected_feature_names,
                "feature_train_min": train_mins,
                "feature_train_max": train_maxs,
                "feature_transform": np.full(
                    len(selected_terms),
                    min_max_transform_name,
                    dtype=object,
                ),
            },
            ROW_L2_FEATURE_MODE: {
                "selected_feature_names": selected_feature_names,
                "feature_train_min": np.full(len(selected_terms), np.nan),
                "feature_train_max": np.full(len(selected_terms), np.nan),
                "feature_transform": np.full(
                    len(selected_terms),
                    l2_transform_name,
                    dtype=object,
                ),
            },
        },
        COLUMN_MIN_MAX_FEATURE_MODE: {
            "train": X_train_min_max_features,
            "test": X_test_min_max_features,
        },
        ROW_L2_FEATURE_MODE: {
            "train": X_train_l2_features,
            "test": X_test_l2_features,
        },
    }


def _build_vocabulary_variant(
    X_train_count,
    X_test_count,
    feature_names,
    train_labels,
    train_word_counts,
    document_frequency_filter_applied,
):
    return {
        "X_train_count": X_train_count,
        "X_test_count": X_test_count,
        "feature_names": np.asarray(feature_names),
        "document_frequency_filter_applied": document_frequency_filter_applied,
        "term_frequency_df": sort_difference_table(
            build_term_frequency_difference_table(
                X_train_count,
                train_labels,
                feature_names,
                train_word_counts,
            )
        ),
        "log_term_frequency_df": sort_difference_table(
            build_log_term_frequency_difference_table(
                X_train_count,
                train_labels,
                feature_names,
                train_word_counts,
            )
        ),
        "tf_s_df": sort_difference_table(
            build_tf_s_difference_table(
                X_train_count,
                train_labels,
                feature_names,
            )
        ),
        "log_tf_s_df": sort_difference_table(
            build_log_tf_s_difference_table(
                X_train_count,
                train_labels,
                feature_names,
            )
        ),
    }


class SplitFeatureContext:
    """1 つの train/test split に必要な特徴量関連情報をまとめて持つコンテナ。"""

    def __init__(self, train_docs, train_labels, test_docs, test_labels, train_doc_ids, test_doc_ids):
        """split の学習データだけで語彙と2種類のTF差分表を作る。"""
        self.train_docs = np.array(train_docs, dtype=object)
        self.train_labels = np.array(train_labels, dtype=int)
        self.test_docs = np.array(test_docs, dtype=object)
        self.test_labels = np.array(test_labels, dtype=int)
        self.train_doc_ids = np.array(train_doc_ids, dtype=int)
        self.test_doc_ids = np.array(test_doc_ids, dtype=int)
        (
            self.X_train_count,
            self.X_test_count,
            self.feature_names,
            self.train_word_counts,
            self.test_word_counts,
        ) = vectorize_train_test_documents(self.train_docs, self.test_docs)
        unfiltered_variant = _build_vocabulary_variant(
            self.X_train_count,
            self.X_test_count,
            self.feature_names,
            self.train_labels,
            self.train_word_counts,
            document_frequency_filter_applied=False,
        )
        (
            filtered_X_train_count,
            filtered_X_test_count,
            filtered_feature_names,
            filtered_document_frequency_ratios,
        ) = filter_count_matrices_by_document_frequency(
            self.X_train_count,
            self.X_test_count,
            self.feature_names,
        )
        filtered_variant = _build_vocabulary_variant(
            filtered_X_train_count,
            filtered_X_test_count,
            filtered_feature_names,
            self.train_labels,
            self.train_word_counts,
            document_frequency_filter_applied=True,
        )
        filtered_variant["document_frequency_ratios"] = (
            filtered_document_frequency_ratios
        )
        self.vocabulary_variants = {
            False: unfiltered_variant,
            True: filtered_variant,
        }
        # 既存のTF監査処理は、従来どおりフィルタなし語彙を参照する。
        self.term_frequency_df = unfiltered_variant["term_frequency_df"]
        self.log_term_frequency_df = unfiltered_variant[
            "log_term_frequency_df"
        ]
        self.tf_s_df = unfiltered_variant["tf_s_df"]
        self.log_tf_s_df = unfiltered_variant["log_tf_s_df"]
        self.feature_cache = {}

    def get_feature_bundle(self, selection_mode, feature_count, feature_mode):
        """指定方式・単語数のTF上位語と特徴行列をキャッシュ付きで返す。"""
        filter_applied = _uses_document_frequency_filter(feature_mode)
        base_feature_mode = _base_feature_mode(feature_mode)
        cache_key = (selection_mode, feature_count, filter_applied)
        if cache_key not in self.feature_cache:
            vocabulary_variant = self.vocabulary_variants[filter_applied]
            if selection_mode == RELATIVE_TERM_FREQUENCY_MODE:
                selected_terms_df = select_term_frequency_terms(
                    vocabulary_variant["term_frequency_df"],
                    feature_count,
                )
            elif selection_mode == LOG_TERM_FREQUENCY_MODE:
                selected_terms_df = select_log_term_frequency_terms(
                    vocabulary_variant["log_term_frequency_df"],
                    feature_count,
                )
            elif selection_mode == TF_S_MODE:
                selected_terms_df = select_tf_s_terms(
                    vocabulary_variant["tf_s_df"],
                    feature_count,
                )
            elif selection_mode == LOG_TF_S_MODE:
                selected_terms_df = select_log_tf_s_terms(
                    vocabulary_variant["log_tf_s_df"],
                    feature_count,
                )
            else:
                raise ValueError(f"Unsupported selection_mode: {selection_mode}")

            feature_matrices = build_selected_feature_matrices(
                vocabulary_variant["X_train_count"],
                vocabulary_variant["X_test_count"],
                vocabulary_variant["feature_names"],
                selected_terms_df,
                self.train_word_counts,
                self.test_word_counts,
            )
            selected_terms_df = selected_terms_df.copy()
            self.feature_cache[cache_key] = {
                "selection_mode": selection_mode,
                "feature_count": feature_count,
                "selected_terms_df": selected_terms_df,
                "selected_term_count": len(feature_matrices["selected_terms"]),
                "feature_matrices": feature_matrices,
                "X_train_count": vocabulary_variant["X_train_count"],
                "X_test_count": vocabulary_variant["X_test_count"],
                "feature_names": vocabulary_variant["feature_names"],
                "document_frequency_filter_applied": filter_applied,
            }
        feature_bundle = self.feature_cache[cache_key]
        if base_feature_mode not in feature_bundle["feature_matrices"]:
            raise ValueError(
                "Feature mode is incompatible with selection mode: "
                f"selection_mode={selection_mode}, feature_mode={feature_mode}"
            )
        return feature_bundle


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
                f"[parallel warning] {env_name}={env_value!r} is not an integer; using 1 worker"
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


def build_outer_fold_context_from_indices(
    outer_fold,
    train_index,
    test_index,
    documents,
    labels,
    doc_ids,
):
    """1つのouter foldに必要なouter/innerコンテキストを構築する。"""
    outer_fold_start = time.perf_counter()
    log_progress(
        f"[split setup] outer fold {outer_fold}/{N_SPLITS}: "
        "preparing train/test feature contexts"
    )
    outer_train_docs = documents[train_index]
    outer_train_labels = labels[train_index]
    outer_test_docs = documents[test_index]
    outer_test_labels = labels[test_index]
    outer_train_doc_ids = doc_ids[train_index]
    outer_test_doc_ids = doc_ids[test_index]

    outer_context = SplitFeatureContext(
        outer_train_docs,
        outer_train_labels,
        outer_test_docs,
        outer_test_labels,
        outer_train_doc_ids,
        outer_test_doc_ids,
    )

    # outer trainの中だけでinner CVを作り、inner側でもリークを防ぐ。
    inner_cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    inner_contexts = []
    for inner_train_index, inner_test_index in inner_cv.split(
        outer_train_docs,
        outer_train_labels,
    ):
        inner_contexts.append(
            SplitFeatureContext(
                outer_train_docs[inner_train_index],
                outer_train_labels[inner_train_index],
                outer_train_docs[inner_test_index],
                outer_train_labels[inner_test_index],
                outer_train_doc_ids[inner_train_index],
                outer_train_doc_ids[inner_test_index],
            )
        )

    log_progress(
        f"[split setup] outer fold {outer_fold}/{N_SPLITS} completed "
        f"({len(inner_contexts)} inner contexts, "
        f"elapsed {format_elapsed_seconds(time.perf_counter() - outer_fold_start)})"
    )
    return {
        "outer_fold": outer_fold,
        "outer_context": outer_context,
        "inner_contexts": inner_contexts,
    }


_SPLIT_WORKER_DOCUMENTS = None
_SPLIT_WORKER_LABELS = None
_SPLIT_WORKER_DOC_IDS = None


def _init_split_context_worker(documents, labels, doc_ids):
    """各ワーカーへcorpusを1回だけ渡す。"""
    global _SPLIT_WORKER_DOCUMENTS
    global _SPLIT_WORKER_LABELS
    global _SPLIT_WORKER_DOC_IDS

    _SPLIT_WORKER_DOCUMENTS = documents
    _SPLIT_WORKER_LABELS = labels
    _SPLIT_WORKER_DOC_IDS = doc_ids


def _build_outer_fold_context_worker(fold_payload):
    """ProcessPoolExecutorから呼ぶouter fold構築ジョブ。"""
    if _SPLIT_WORKER_DOCUMENTS is None:
        raise RuntimeError("split context worker was not initialized")
    outer_fold, train_index, test_index = fold_payload
    return build_outer_fold_context_from_indices(
        outer_fold,
        train_index,
        test_index,
        _SPLIT_WORKER_DOCUMENTS,
        _SPLIT_WORKER_LABELS,
        _SPLIT_WORKER_DOC_IDS,
    )


def build_outer_fold_contexts(documents, labels, doc_ids):
    """outer/inner の全 split に対する `SplitFeatureContext` を先に構築する。

    候補パラメータごとに毎回ベクトル化し直すと非常に遅いため、split ごとの
    前処理・語彙化・スコア表をここでまとめて用意する。
    """
    outer_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    outer_splits = list(enumerate(outer_cv.split(documents, labels), start=1))
    build_start = time.perf_counter()

    log_progress(f"[split setup] building outer/inner feature contexts for {N_SPLITS}-fold nested CV")

    split_worker_count = resolve_worker_count(
        SPLIT_CONTEXT_WORKERS_ENV,
        DEFAULT_SPLIT_CONTEXT_WORKER_LIMIT,
        len(outer_splits),
    )
    log_progress(
        f"[split setup] workers: {split_worker_count} "
        f"(override with {SPLIT_CONTEXT_WORKERS_ENV})"
    )

    if split_worker_count == 1:
        outer_fold_contexts = [
            build_outer_fold_context_from_indices(
                outer_fold,
                train_index,
                test_index,
                documents,
                labels,
                doc_ids,
            )
            for outer_fold, (train_index, test_index) in outer_splits
        ]
    else:
        outer_fold_contexts = []
        executor = futures.ProcessPoolExecutor(
            max_workers=split_worker_count,
            mp_context=mp.get_context("spawn"),
            initializer=_init_split_context_worker,
            initargs=(documents, labels, doc_ids),
        )
        try:
            future_to_fold = {
                executor.submit(
                    _build_outer_fold_context_worker,
                    (outer_fold, train_index, test_index),
                ): outer_fold
                for outer_fold, (train_index, test_index) in outer_splits
            }
            completed_folds = 0
            for future in futures.as_completed(future_to_fold):
                outer_fold = future_to_fold[future]
                outer_fold_contexts.append(future.result())
                completed_folds += 1
                log_progress(
                    f"[split setup] outer fold {outer_fold}/{N_SPLITS} collected "
                    f"({completed_folds}/{len(outer_splits)})"
                )
        finally:
            executor.shutdown(cancel_futures=True)

        outer_fold_contexts.sort(key=lambda fold_bundle: fold_bundle["outer_fold"])

    log_progress(
        f"[split setup] all feature contexts completed "
        f"(elapsed {format_elapsed_seconds(time.perf_counter() - build_start)})"
    )
    return outer_fold_contexts


def _build_lightgbm_model(n_estimators, num_leaves, max_depth, learning_rate):
    """LightGBM分類器を、学習データ全件正分類の候補として生成する。"""
    return LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=int(n_estimators),
        num_leaves=int(num_leaves),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        min_child_samples=1,
        min_child_weight=0.0,
        min_split_gain=0.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        subsample=1.0,
        subsample_freq=0,
        colsample_bytree=1.0,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=-1,
        force_col_wise=True,
    )


def _compute_margins(model, X_train, y_train):
    """学習データ上の raw-score signed margin を計算する。"""
    decision_values = model.predict(X_train, raw_score=True)
    y_signed = np.where(y_train == 0, -1, 1)
    return y_signed * decision_values


def format_candidate_label(selection_mode, feature_count, c, gamma=None):
    """単語選択方式・単語数・LightGBMパラメータをログ用文字列にする。"""
    num_leaves, max_depth, learning_rate = gamma
    return (
        f"selection={selection_mode}, feature_count={feature_count}, "
        f"n_estimators={c}, num_leaves={num_leaves}, "
        f"max_depth={max_depth}, learning_rate={learning_rate}"
    )


def format_label_name(label):
    """0/1 ラベルを人が読める名前へ変換する。"""
    return "label1" if label == 1 else "label0"


def fit_predict_with_margin_check(kernel, c, gamma, X_train, y_train, X_test):
    """1回分のLightGBM学習・raw margin判定・予測を同一プロセス内で行う。"""
    try:
        num_leaves, max_depth, learning_rate = gamma
        model = _build_lightgbm_model(
            c,
            num_leaves,
            max_depth,
            learning_rate,
        )
        model.fit(X_train, y_train)

        margins = _compute_margins(model, X_train, y_train)
        if not np.all(margins >= MARGIN_THRESHOLD):
            return {"status": "invalid_margin"}

        y_pred = model.predict(X_test)
        decision_values = np.asarray(model.predict(X_test, raw_score=True)).ravel()
        return {
            "status": "ok",
            "y_pred": np.asarray(y_pred, dtype=int).tolist(),
            "decision_values": decision_values.astype(float).tolist(),
        }
    except BaseException as exc:
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def fit_and_evaluate_split_context(
    split_context,
    selection_mode,
    feature_count,
    feature_mode,
    kernel,
    c,
    gamma,
    include_misclassified=False,
):
    """1 つの split で特徴選択から学習・評価までを実行する。

    train 側でのみ語を採用し、その語で train/test を特徴化した後に LightGBM を学習する。
    margin 条件を満たさない候補はここで `None` として落とす。
    """
    feature_bundle = split_context.get_feature_bundle(
        selection_mode,
        feature_count,
        feature_mode,
    )
    selected_term_count = feature_bundle["selected_term_count"]
    if selected_term_count == 0:
        return None

    base_feature_mode = _base_feature_mode(feature_mode)
    X_train = feature_bundle["feature_matrices"][base_feature_mode]["train"]
    X_test = feature_bundle["feature_matrices"][base_feature_mode]["test"]

    fit_result = fit_predict_with_margin_check(
        kernel,
        c,
        gamma,
        X_train,
        split_context.train_labels,
        X_test,
    )
    if fit_result["status"] == "invalid_margin":
        return None
    if fit_result["status"] == "error":
        raise RuntimeError(f"{fit_result['error_type']}: {fit_result['error_message']}")

    y_pred = np.asarray(fit_result["y_pred"], dtype=int)
    result = {
        "selected_term_count": selected_term_count,
        "accuracy": float(accuracy_score(split_context.test_labels, y_pred)),
        "recall": float(recall_score(split_context.test_labels, y_pred, zero_division=0)),
        "precision": float(precision_score(split_context.test_labels, y_pred, zero_division=0)),
        "f1_score": float(f1_score(split_context.test_labels, y_pred, zero_division=0)),
    }

    if include_misclassified:
        misclassified_rows = []
        for row_idx, (actual_label, predicted_label) in enumerate(zip(split_context.test_labels, y_pred)):
            if actual_label == predicted_label:
                continue
            # outer test で間違えた文書を、後で見直せるよう本文つきで残す。
            misclassified_rows.append(
                {
                    "document_id": int(split_context.test_doc_ids[row_idx]),
                    "actual_label": int(actual_label),
                    "actual_label_name": format_label_name(int(actual_label)),
                    "predicted_label": int(predicted_label),
                    "predicted_label_name": format_label_name(int(predicted_label)),
                    "document_text": str(split_context.test_docs[row_idx]),
                }
            )
        result["misclassified_count"] = len(misclassified_rows)
        result["misclassified_rows"] = misclassified_rows

    return result


def evaluate_candidate_across_outer_folds(
    outer_fold_contexts,
    selection_mode,
    feature_count,
    feature_mode,
    kernel,
    c,
    gamma,
):
    """1 つの候補設定を outer 全 fold で評価する。

    どこか 1 fold でも inner/outer の margin 条件を満たせなければ、その候補は
    実験全体から除外する。
    """
    candidate_label = format_candidate_label(selection_mode, feature_count, c, gamma)
    fold_records = []

    for fold_bundle in outer_fold_contexts:
        inner_scores = []
        for inner_fold, inner_context in enumerate(fold_bundle["inner_contexts"], start=1):
            inner_result = fit_and_evaluate_split_context(
                inner_context,
                selection_mode,
                feature_count,
                feature_mode,
                kernel,
                c,
                gamma,
            )
            if inner_result is None:
                return None
            inner_scores.append(inner_result["accuracy"])

        # inner で生き残った候補だけを outer test で評価する。
        outer_result = fit_and_evaluate_split_context(
            fold_bundle["outer_context"],
            selection_mode,
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
                "selection_mode": selection_mode,
                "feature_count": feature_count,
                "selected_term_count": outer_result["selected_term_count"],
                "c": c,
                "gamma": gamma,
                "param_label": candidate_label,
                # この inner 平均精度を、fold ごとの候補選択基準として使う。
                "inner_score": float(sum(inner_scores) / len(inner_scores)),
                "accuracy": outer_result["accuracy"],
                "recall": outer_result["recall"],
                "precision": outer_result["precision"],
                "f1_score": outer_result["f1_score"],
            }
        )

    return fold_records


def _persistent_candidate_worker(
    connection,
    outer_fold_contexts,
    selection_mode,
    feature_count,
    feature_mode,
    kernel,
):
    """同一条件のLightGBM候補を順次評価する常駐ワーカー。"""
    try:
        connection.send({"message_type": "ready"})
        while True:
            command = connection.recv()
            if command["message_type"] == "shutdown":
                return

            task_id = command["task_id"]
            c = command["c"]
            gamma = command["gamma"]
            try:
                candidate_records = evaluate_candidate_across_outer_folds(
                    outer_fold_contexts,
                    selection_mode,
                    feature_count,
                    feature_mode,
                    kernel,
                    c,
                    gamma,
                )
                result = {
                    "status": "ok",
                    "candidate_records": candidate_records,
                }
            except BaseException as exc:
                result = {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            connection.send(
                {
                    "message_type": "result",
                    "task_id": task_id,
                    "result": result,
                }
            )
    except (EOFError, BrokenPipeError, OSError):
        return
    finally:
        connection.close()


class PersistentCandidateWorker:
    """条件単位で常駐し、タイムアウト時だけ再起動する候補評価ワーカー。"""

    def __init__(
        self,
        outer_fold_contexts,
        selection_mode,
        feature_count,
        feature_mode,
        kernel,
    ):
        self.outer_fold_contexts = outer_fold_contexts
        self.selection_mode = selection_mode
        self.feature_count = feature_count
        self.feature_mode = feature_mode
        self.kernel = kernel
        self.ctx = mp.get_context("spawn")
        self.connection = None
        self.process = None
        self.next_task_id = 0

    def _stop(self, force=False):
        connection = self.connection
        process = self.process
        self.connection = None
        self.process = None

        if process is None:
            return

        if process.is_alive() and not force and connection is not None:
            try:
                connection.send({"message_type": "shutdown"})
                process.join(timeout=5)
            except (BrokenPipeError, EOFError, OSError):
                pass

        if process.is_alive():
            process.terminate()
            process.join()

        if connection is not None:
            connection.close()
        process.close()

    def close(self):
        self._stop()

    def _worker_exit_result(self):
        exitcode = self.process.exitcode if self.process is not None else None
        return {
            "status": "error",
            "error_type": "WorkerExit",
            "error_message": f"candidate worker exited with code {exitcode}",
        }

    def _wait_for_message(self, deadline):
        while True:
            remaining_seconds = deadline - time.perf_counter()
            if remaining_seconds <= 0:
                return None

            try:
                message_available = self.connection.poll(min(0.1, remaining_seconds))
            except (EOFError, OSError):
                return self._worker_exit_result()

            if message_available:
                try:
                    return self.connection.recv()
                except (EOFError, OSError):
                    return self._worker_exit_result()

            if not self.process.is_alive():
                self.process.join()
                return self._worker_exit_result()

    def _start(self):
        parent_connection, child_connection = self.ctx.Pipe(duplex=True)
        process = self.ctx.Process(
            target=_persistent_candidate_worker,
            args=(
                child_connection,
                self.outer_fold_contexts,
                self.selection_mode,
                self.feature_count,
                self.feature_mode,
                self.kernel,
            ),
        )
        process.daemon = True
        try:
            process.start()
        except BaseException as exc:
            parent_connection.close()
            child_connection.close()
            process.close()
            return (
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
                None,
            )

        child_connection.close()
        self.connection = parent_connection
        self.process = process

        # 旧実装と同様、Process.start()完了後から候補タイムアウトを計測する。
        deadline = time.perf_counter() + CANDIDATE_TIMEOUT_SECONDS
        ready_message = self._wait_for_message(deadline)
        if ready_message is None:
            self._stop(force=True)
            return {"status": "timeout", "candidate_records": None}, None
        if ready_message.get("status") == "error":
            self._stop(force=True)
            return ready_message, None
        if ready_message.get("message_type") != "ready":
            self._stop(force=True)
            return (
                {
                    "status": "error",
                    "error_type": "WorkerProtocolError",
                    "error_message": "candidate worker did not send a ready message",
                },
                None,
            )
        return None, deadline

    def evaluate(self, c, gamma):
        """1候補を評価し、制限時間超過時はワーカーを終了する。"""
        candidate_label = format_candidate_label(
            self.selection_mode,
            self.feature_count,
            c,
            gamma,
        )
        if self.process is None:
            start_result, deadline = self._start()
            if start_result is not None:
                if start_result["status"] == "timeout":
                    log_progress(
                        f"[candidate timeout] {candidate_label}: exceeded "
                        f"{format_elapsed_seconds(CANDIDATE_TIMEOUT_SECONDS)}"
                    )
                return start_result
        else:
            deadline = time.perf_counter() + CANDIDATE_TIMEOUT_SECONDS

        self.next_task_id += 1
        task_id = self.next_task_id
        try:
            self.connection.send(
                {
                    "message_type": "evaluate",
                    "task_id": task_id,
                    "c": c,
                    "gamma": gamma,
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            result = self._worker_exit_result()
            self._stop(force=True)
            return result

        response = self._wait_for_message(deadline)
        if response is None:
            self._stop(force=True)
            log_progress(
                f"[candidate timeout] {candidate_label}: exceeded "
                f"{format_elapsed_seconds(CANDIDATE_TIMEOUT_SECONDS)}"
            )
            return {"status": "timeout", "candidate_records": None}
        if response.get("status") == "error":
            self._stop(force=True)
            return response
        if (
            response.get("message_type") != "result"
            or response.get("task_id") != task_id
        ):
            self._stop(force=True)
            return {
                "status": "error",
                "error_type": "WorkerProtocolError",
                "error_message": "candidate worker returned an unexpected response",
            }
        return response["result"]


def round_half_up(value, digits=3):
    """CSV 用に四捨五入したい数値を丸める。空値はそのまま返す。"""
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


def build_empty_result(selection_mode, feature_count, feature_mode, kernel):
    """候補が全滅した場合にも同じ形式で扱える空の結果辞書を作る。"""
    result = {
        "selection_mode": selection_mode,
        "feature_count": feature_count,
        "feature_mode": feature_mode,
        "kernel": kernel,
        "margin_threshold": MARGIN_THRESHOLD,
        "valid_param_labels": [],
        "dropped_param_labels": [],
        "timeout_param_labels": [],
        "selected_records": [],
        "best_cs": [],
        "selected_term_counts": [],
        "misclassified_counts": [],
    }
    if kernel == "lightgbm":
        result["best_gammas"] = []
    return result


def run_nested_cv_for_condition(
    outer_fold_contexts,
    selection_mode,
    feature_mode,
    kernel,
    feature_count,
):
    """固定した特徴数で nested CV を回し、fold ごとの最良LightGBMパラメータを選ぶ。"""
    c_values = N_ESTIMATORS_VALUES
    gamma_candidates = [
        (num_leaves, max_depth, learning_rate)
        for num_leaves in NUM_LEAVES_VALUES
        for max_depth in MAX_DEPTH_VALUES
        for learning_rate in LEARNING_RATE_VALUES
        if max_depth <= 0 or num_leaves <= 2**max_depth
    ]
    total_candidates = len(c_values) * len(gamma_candidates)
    condition_label = format_condition_label(selection_mode, feature_mode, kernel)
    condition_start = time.perf_counter()

    valid_candidate_results = {}
    dropped_param_labels = []
    timeout_param_labels = []
    processed_candidates = 0

    log_progress(
        f"[condition start] {condition_label}: "
        f"feature_count={feature_count}, {total_candidates} LightGBM parameter settings"
    )

    candidate_worker = PersistentCandidateWorker(
        outer_fold_contexts,
        selection_mode,
        feature_count,
        feature_mode,
        kernel,
    )
    try:
        for c in c_values:
            for gamma in gamma_candidates:
                candidate_label = format_candidate_label(selection_mode, feature_count, c, gamma)
                # 1 候補について outer 全 fold を走らせ、全 fold で通るか確認する。
                candidate_result = candidate_worker.evaluate(c, gamma)
                if candidate_result["status"] == "error":
                    raise RuntimeError(
                        f"{candidate_result['error_type']}: {candidate_result['error_message']}"
                    )
                candidate_records = candidate_result["candidate_records"]
                processed_candidates += 1
                if candidate_result["status"] == "timeout":
                    timeout_param_labels.append(candidate_label)
                elif candidate_records is None:
                    dropped_param_labels.append(candidate_label)
                else:
                    valid_candidate_results[(c, gamma)] = candidate_records

                if (
                    processed_candidates == 1
                    or processed_candidates == total_candidates
                    or processed_candidates % CANDIDATE_PROGRESS_INTERVAL == 0
                ):
                    log_progress(
                        f"[condition progress] {condition_label}: "
                        f"{processed_candidates}/{total_candidates} candidates processed "
                        f"(valid={len(valid_candidate_results)}, dropped={len(dropped_param_labels)}, "
                        f"timeout={len(timeout_param_labels)}, "
                        f"elapsed {format_elapsed_seconds(time.perf_counter() - condition_start)})"
                    )
    finally:
        candidate_worker.close()

    result = build_empty_result(selection_mode, feature_count, feature_mode, kernel)
    result["valid_param_labels"] = [
        format_candidate_label(selection_mode, feature_count, c, gamma)
        for c, gamma in valid_candidate_results.keys()
    ]
    result["dropped_param_labels"] = dropped_param_labels
    result["timeout_param_labels"] = timeout_param_labels

    if not valid_candidate_results:
        log_progress(
            f"[condition done] {condition_label}: "
            f"no valid candidates after {format_elapsed_seconds(time.perf_counter() - condition_start)}"
        )
        return result

    selected_records = []
    for fold_idx in range(len(outer_fold_contexts)):
        best_record = None
        for candidate_records in valid_candidate_results.values():
            candidate = candidate_records[fold_idx]
            if best_record is None:
                best_record = candidate
                continue
            if candidate["inner_score"] > best_record["inner_score"]:
                best_record = candidate
                continue
            if candidate["inner_score"] == best_record["inner_score"] and candidate["param_label"] < best_record["param_label"]:
                best_record = candidate
        selected_records.append(best_record)

    result["selected_records"] = selected_records
    result["best_cs"] = [record["c"] for record in selected_records]
    result["selected_term_counts"] = [record["selected_term_count"] for record in selected_records]
    if kernel == "lightgbm":
        result["best_gammas"] = [record["gamma"] for record in selected_records]

    # fold ごとに最終採用された候補だけを使って、誤分類文書を保存用に再取得する。
    for fold_idx, selected_record in enumerate(result["selected_records"]):
        outer_detail = fit_and_evaluate_split_context(
            outer_fold_contexts[fold_idx]["outer_context"],
            selection_mode,
            selected_record["feature_count"],
            feature_mode,
            kernel,
            selected_record["c"],
            selected_record["gamma"],
            include_misclassified=True,
        )
        if outer_detail is None:
            selected_record["misclassified_count"] = ""
            selected_record["misclassified_rows"] = []
            continue
        selected_record["misclassified_count"] = outer_detail["misclassified_count"]
        selected_record["misclassified_rows"] = outer_detail["misclassified_rows"]

    result["misclassified_counts"] = [record["misclassified_count"] for record in selected_records]

    result["mean_accuracy"] = float(np.mean([record["accuracy"] for record in selected_records]))
    result["mean_recall"] = float(np.mean([record["recall"] for record in selected_records]))
    result["mean_precision"] = float(np.mean([record["precision"] for record in selected_records]))
    result["mean_f1"] = float(np.mean([record["f1_score"] for record in selected_records]))
    result["mean_selected_term_count"] = float(np.mean(result["selected_term_counts"]))
    log_progress(
        f"[condition done] {condition_label}: "
        f"completed in {format_elapsed_seconds(time.perf_counter() - condition_start)}"
    )
    return result


def fit_selected_outer_model_detail(outer_context, selected_record, feature_mode, kernel):
    """outer foldで最終採用された設定を再学習し、誤分類集計用の詳細を返す。"""
    feature_bundle = outer_context.get_feature_bundle(
        selected_record["selection_mode"],
        selected_record["feature_count"],
        feature_mode,
    )
    base_feature_mode = _base_feature_mode(feature_mode)
    X_train = feature_bundle["feature_matrices"][base_feature_mode]["train"]
    X_test = feature_bundle["feature_matrices"][base_feature_mode]["test"]

    fit_result = fit_predict_with_margin_check(
        kernel,
        selected_record["c"],
        selected_record["gamma"],
        X_train,
        outer_context.train_labels,
        X_test,
    )
    if fit_result["status"] == "invalid_margin":
        return None
    if fit_result["status"] == "error":
        raise RuntimeError(f"{fit_result['error_type']}: {fit_result['error_message']}")

    y_pred = np.asarray(fit_result["y_pred"], dtype=int)
    decision_values = np.asarray(fit_result["decision_values"], dtype=float)
    misclassified_indices = np.flatnonzero(y_pred != outer_context.test_labels)

    return {
        "feature_bundle": feature_bundle,
        "X_train": X_train,
        "X_test": X_test,
        "y_pred": y_pred,
        "decision_values": decision_values,
        "misclassified_indices": misclassified_indices,
        "outer_context": outer_context,
    }


def compute_margin_distance(actual_label, decision_value):
    """真のラベル基準の signed margin から、誤分類点の離れ具合を正値で返す。"""
    signed_label = 1 if int(actual_label) == 1 else -1
    signed_margin = signed_label * float(decision_value)
    return float(-signed_margin)


def run_optional_output_step(result, step_label, callback, *args):
    """補助出力で失敗しても主結果を落とさず、警告として保持する。"""
    try:
        callback(*args)
    except Exception as exc:
        log_progress(f"[output warning] {step_label} failed: {type(exc).__name__}: {exc}")
        result.setdefault("output_warnings", []).append(
            f"{step_label}: {type(exc).__name__}: {exc}"
        )


def format_feature_count_slug(feature_count):
    """単語数をディレクトリ名に使いやすい文字列へ変換する。"""
    return f"n{feature_count}"


def get_condition_metrics_dir(metrics_root_dir, selection_mode, feature_mode, kernel):
    """単語選択方式・特徴値・カーネルごとの出力ディレクトリを返す。"""
    condition_dir = (
        Path(metrics_root_dir)
        / SELECTION_MODE_DIR_NAMES[selection_mode]
        / FEATURE_MODE_DIR_NAMES[feature_mode]
        / KERNEL_DIR_NAMES[kernel]
    )
    condition_dir.mkdir(parents=True, exist_ok=True)
    return condition_dir


def save_nested_cv_result_csvs(result, output_dir, output_prefix):
    """nested CV の主要結果を 3 種類の CSV に保存する。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"{output_prefix}_metrics.csv"
    selected_path = output_dir / f"{output_prefix}_folds.csv"
    params_path = output_dir / f"{output_prefix}_params.csv"

    # 条件全体の平均指標を 1 行で見られるよう、summary 用 CSV を先に作る。
    metrics_row = {
        "selection_mode": result["selection_mode"],
        "feature_count": result["feature_count"],
        "feature_mode": result["feature_mode"],
        "kernel": result["kernel"],
        "margin_threshold": result["margin_threshold"],
        "mean_selected_term_count": round_half_up(result.get("mean_selected_term_count", "")),
        "selected_term_counts": ",".join(map(str, result.get("selected_term_counts", []))),
        "mean_accuracy": round_half_up(result.get("mean_accuracy", "")),
        "mean_recall": round_half_up(result.get("mean_recall", "")),
        "mean_precision": round_half_up(result.get("mean_precision", "")),
        "mean_f1": round_half_up(result.get("mean_f1", "")),
        "best_cs": ",".join(map(str, result.get("best_cs", []))),
        "best_gammas": ",".join(map(str, result.get("best_gammas", []))) if "best_gammas" in result else "",
    }
    pd.DataFrame([metrics_row]).to_csv(metrics_path, index=False, encoding="utf-8-sig")

    # fold ごとに実際に採用された候補と指標も別 CSV へ保存する。
    selected_records_df = pd.DataFrame(
        result.get("selected_records", []),
        columns=[
            "outer_fold",
            "selection_mode",
            "feature_count",
            "selected_term_count",
            "misclassified_count",
            "c",
            "gamma",
            "param_label",
            "inner_score",
            "accuracy",
            "recall",
            "precision",
            "f1_score",
        ],
    )
    if not selected_records_df.empty:
        selected_records_df = selected_records_df.copy()
        for column in ["inner_score", "accuracy", "recall", "precision", "f1_score"]:
            selected_records_df[column] = selected_records_df[column].map(round_half_up)
    selected_records_df.to_csv(selected_path, index=False, encoding="utf-8-sig")

    # 全候補のうち margin 条件を通ったもの / 落ちたものも後で確認できるようにする。
    param_status_rows = [{"status": "valid", "param_label": label} for label in result.get("valid_param_labels", [])]
    param_status_rows.extend(
        {"status": "dropped", "param_label": label} for label in result.get("dropped_param_labels", [])
    )
    param_status_rows.extend(
        {"status": "timeout", "param_label": label} for label in result.get("timeout_param_labels", [])
    )
    pd.DataFrame(param_status_rows, columns=["status", "param_label"]).to_csv(
        params_path,
        index=False,
        encoding="utf-8-sig",
    )

    result["saved_csv_paths"] = {
        "metrics": str(metrics_path),
        "selected_records": str(selected_path),
        "param_status": str(params_path),
    }


def save_outer_fold_selected_terms(result, outer_fold_contexts, output_dir):
    """outer fold ごとに、最終採用された語一覧を保存する。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for record in result.get("selected_records", []):
        outer_fold = record["outer_fold"]
        outer_context = outer_fold_contexts[outer_fold - 1]["outer_context"]
        feature_bundle = outer_context.get_feature_bundle(
            record["selection_mode"],
            record["feature_count"],
            result["feature_mode"],
        )
        if record["selection_mode"] == RELATIVE_TERM_FREQUENCY_MODE:
            selected_terms_df = add_term_frequency_iqr_diagnostics(
                feature_bundle["selected_terms_df"],
                feature_bundle["X_train_count"],
                outer_context.train_labels,
                feature_bundle["feature_names"],
                outer_context.train_word_counts,
            )
        else:
            selected_terms_df = feature_bundle["selected_terms_df"].copy()
        base_feature_mode = _base_feature_mode(result["feature_mode"])
        feature_metadata = feature_bundle["feature_matrices"][
            "feature_metadata"
        ][base_feature_mode]
        selected_terms_df["feature_mode"] = result["feature_mode"]
        selected_terms_df["document_frequency_filter_applied"] = (
            feature_bundle["document_frequency_filter_applied"]
        )
        selected_terms_df["feature_name"] = feature_metadata[
            "selected_feature_names"
        ]
        selected_terms_df["feature_transform"] = feature_metadata[
            "feature_transform"
        ]
        selected_terms_df["train_min"] = feature_metadata["feature_train_min"]
        selected_terms_df["train_max"] = feature_metadata["feature_train_max"]
        selected_terms_df.insert(0, "outer_fold", outer_fold)
        selected_terms_df.insert(1, "feature_count", record["feature_count"])
        file_path = output_dir / f"f{outer_fold:02d}_terms.csv"
        selected_terms_df.to_csv(file_path, index=False, encoding="utf-8-sig")

        iqr_outlier_rows = []
        iqr_outlier_path = ""
        tf_rows = selected_terms_df.iloc[0:0]
        label1_has_outlier = pd.Series(dtype=bool)
        label0_has_outlier = pd.Series(dtype=bool)
        if record["selection_mode"] == RELATIVE_TERM_FREQUENCY_MODE:
            iqr_outlier_rows = build_term_frequency_iqr_outlier_rows(
                selected_terms_df,
                feature_bundle["X_train_count"],
                outer_context.train_labels,
                outer_context.train_doc_ids,
                feature_bundle["feature_names"],
                outer_context.train_word_counts,
                outer_fold,
                record["feature_count"],
            )
            iqr_outlier_path = (
                output_dir / f"f{outer_fold:02d}_tf_iqr_outliers.csv"
            )
            pd.DataFrame(
                iqr_outlier_rows,
                columns=[
                    "outer_fold",
                    "feature_count",
                    "selection_mode",
                    "selection_rank",
                    "term",
                    "label",
                    "doc_id",
                    "relative_frequency",
                    "q1",
                    "q3",
                    "iqr",
                    "lower_bound",
                    "upper_bound",
                    "outlier_direction",
                ],
            ).to_csv(iqr_outlier_path, index=False, encoding="utf-8-sig")

            tf_rows = selected_terms_df.loc[
                selected_terms_df["selection_source"]
                == RELATIVE_TERM_FREQUENCY_MODE
            ]
            label1_has_outlier = (
                tf_rows["tf_label1_has_outlier"].fillna(False).astype(bool)
            )
            label0_has_outlier = (
                tf_rows["tf_label0_has_outlier"].fillna(False).astype(bool)
            )
        summary_rows.append(
            {
                "outer_fold": outer_fold,
                "selection_mode": record["selection_mode"],
                "feature_count": record["feature_count"],
                "selected_term_count": record["selected_term_count"],
                "tf_iqr_evaluated_term_count": len(tf_rows),
                "tf_iqr_term_count_with_label1_outlier": int(label1_has_outlier.sum()),
                "tf_iqr_term_count_with_label0_outlier": int(label0_has_outlier.sum()),
                "tf_iqr_term_count_with_any_label_outlier": int(
                    (label1_has_outlier | label0_has_outlier).sum()
                ),
                "tf_iqr_outlier_document_term_count": len(iqr_outlier_rows),
                "file_path": str(file_path),
                "tf_iqr_outlier_file_path": str(iqr_outlier_path),
            }
        )

    pd.DataFrame(
        summary_rows,
        columns=[
            "outer_fold",
            "selection_mode",
            "feature_count",
            "selected_term_count",
            "tf_iqr_evaluated_term_count",
            "tf_iqr_term_count_with_label1_outlier",
            "tf_iqr_term_count_with_label0_outlier",
            "tf_iqr_term_count_with_any_label_outlier",
            "tf_iqr_outlier_document_term_count",
            "file_path",
            "tf_iqr_outlier_file_path",
        ],
    ).to_csv(
        output_dir / "feat_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


def save_outer_fold_misclassified_documents(result, output_dir):
    """outer test で誤分類した文書を fold 別・全 fold 結合の両方で保存する。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_rows = []
    summary_rows = []

    for record in result.get("selected_records", []):
        outer_fold = record["outer_fold"]
        fold_rows = []
        for row in record.get("misclassified_rows", []):
            fold_rows.append(
                {
                    "outer_fold": outer_fold,
                    "selection_mode": record["selection_mode"],
                    "feature_count": record["feature_count"],
                    "selected_term_count": record["selected_term_count"],
                    "c": record["c"],
                    "gamma": record["gamma"],
                    "param_label": record["param_label"],
                    **row,
                }
            )

        # fold 単位の誤分類 CSV と、全 fold 結合 CSV の両方を作る。
        file_path = output_dir / f"f{outer_fold:02d}_mis.csv"
        pd.DataFrame(
            fold_rows,
            columns=[
                "outer_fold",
                "selection_mode",
                "feature_count",
                "selected_term_count",
                "c",
                "gamma",
                "param_label",
                "document_id",
                "actual_label",
                "actual_label_name",
                "predicted_label",
                "predicted_label_name",
                "document_text",
            ],
        ).to_csv(file_path, index=False, encoding="utf-8-sig")

        combined_rows.extend(fold_rows)
        summary_rows.append(
            {
                "outer_fold": outer_fold,
                "selection_mode": record["selection_mode"],
                "feature_count": record["feature_count"],
                "selected_term_count": record["selected_term_count"],
                "misclassified_count": record.get("misclassified_count", 0),
                "file_path": str(file_path),
            }
        )

    combined_path = output_dir / "mis_all.csv"
    pd.DataFrame(
        combined_rows,
        columns=[
            "outer_fold",
            "selection_mode",
            "feature_count",
            "selected_term_count",
            "c",
            "gamma",
            "param_label",
            "document_id",
            "actual_label",
            "actual_label_name",
            "predicted_label",
            "predicted_label_name",
            "document_text",
        ],
    ).to_csv(combined_path, index=False, encoding="utf-8-sig")

    summary_path = output_dir / "mis_summary.csv"
    pd.DataFrame(
        summary_rows,
        columns=[
            "outer_fold",
            "selection_mode",
            "feature_count",
            "selected_term_count",
            "misclassified_count",
            "file_path",
        ],
    ).to_csv(summary_path, index=False, encoding="utf-8-sig")

    result.setdefault("saved_csv_paths", {})
    result["saved_csv_paths"]["misclassified_all"] = str(combined_path)
    result["saved_csv_paths"]["misclassified_summary"] = str(summary_path)


def save_global_top_margin_misclassified_documents(results, outer_fold_contexts, pdf_names, output_path, top_n):
    """全単語数を横断して、margin から最も遠い誤分類文書 top-N を 1 CSV にまとめる。"""
    if top_n <= 0:
        raise ValueError(f"top_n must be positive, got {top_n}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_names = np.asarray(pdf_names, dtype=object)
    active_conditions = {
        (result["selection_mode"], result["feature_mode"], result["kernel"])
        for result in results
        if result.get("selected_records")
    }
    include_condition_columns = len(active_conditions) > 1

    all_rows = []
    for result in results:
        selection_mode = result["selection_mode"]
        feature_mode = result["feature_mode"]
        kernel = result["kernel"]

        for selected_record in result.get("selected_records", []):
            outer_fold = int(selected_record["outer_fold"])
            outer_context = outer_fold_contexts[outer_fold - 1]["outer_context"]
            model_detail = fit_selected_outer_model_detail(
                outer_context,
                selected_record,
                feature_mode,
                kernel,
            )
            if model_detail is None:
                log_progress(
                    "[output warning] global top-margin model detail skipped: "
                    f"selection={selection_mode}, feature_count={selected_record['feature_count']}, "
                    f"outer_fold={outer_fold}, kernel={kernel}"
                )
                continue

            for misclassified_idx in model_detail["misclassified_indices"]:
                actual_label = int(model_detail["outer_context"].test_labels[misclassified_idx])
                predicted_label = int(model_detail["y_pred"][misclassified_idx])
                decision_value = float(model_detail["decision_values"][misclassified_idx])
                document_id = int(model_detail["outer_context"].test_doc_ids[misclassified_idx])

                if not 0 <= document_id < len(pdf_names):
                    raise RuntimeError(
                        "document_id is out of range for pdf_names: "
                        f"document_id={document_id}, pdf_name_count={len(pdf_names)}"
                    )

                row = {
                    "selection_mode": selection_mode,
                    "feature_count": int(selected_record["feature_count"]),
                    "rank": 0,
                    "pdf_name": str(pdf_names[document_id]),
                    "actual_label": actual_label,
                    "predicted_label": predicted_label,
                    "margin_distance": compute_margin_distance(actual_label, decision_value),
                }
                if include_condition_columns:
                    row["feature_mode"] = feature_mode
                    row["kernel"] = kernel
                all_rows.append(row)

    all_rows.sort(
        key=lambda row: (
            -row["margin_distance"],
            row["selection_mode"],
            row["feature_count"],
            row.get("feature_mode", ""),
            row.get("kernel", ""),
            row["pdf_name"],
            row["actual_label"],
            row["predicted_label"],
        )
    )

    top_rows = all_rows[:top_n]
    for rank, row in enumerate(top_rows, start=1):
        row["rank"] = rank

    columns = [
        "selection_mode",
        "feature_count",
        "rank",
        "pdf_name",
        "actual_label",
        "predicted_label",
        "margin_distance",
    ]
    if include_condition_columns:
        columns.extend(["feature_mode", "kernel"])

    pd.DataFrame(top_rows, columns=columns).to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def save_term_frequency_iqr_audit(
    outer_fold_contexts,
    feature_count_candidates,
    output_dir,
):
    """最大候補数までのTF上位語を、outer foldごとにIQR監査して保存する。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    maximum_feature_count = max(feature_count_candidates)
    fold_term_tables = []
    all_outlier_rows = []
    fold_summary_rows = []
    plots_dir = output_dir / "plots"
    fold_boxplot_paths = []

    for fold_bundle in outer_fold_contexts:
        outer_fold = fold_bundle["outer_fold"]
        outer_context = fold_bundle["outer_context"]
        feature_bundle = outer_context.get_feature_bundle(
            RELATIVE_TERM_FREQUENCY_MODE,
            maximum_feature_count,
            COLUMN_MIN_MAX_FEATURE_MODE,
        )
        fold_terms_df = add_term_frequency_iqr_diagnostics(
            feature_bundle["selected_terms_df"],
            outer_context.X_train_count,
            outer_context.train_labels,
            outer_context.feature_names,
            outer_context.train_word_counts,
        )
        fold_terms_df.insert(0, "outer_fold", outer_fold)
        fold_terms_df.insert(1, "audit_feature_count", maximum_feature_count)
        fold_term_tables.append(fold_terms_df)
        fold_boxplot_path = save_term_frequency_iqr_boxplot(
            outer_context,
            fold_terms_df,
            plots_dir / f"f{outer_fold:02d}_tf_top10_boxplot.png",
            outer_fold,
        )
        if fold_boxplot_path is not None:
            fold_boxplot_paths.append(str(fold_boxplot_path))

        all_outlier_rows.extend(
            build_term_frequency_iqr_outlier_rows(
                fold_terms_df,
                outer_context.X_train_count,
                outer_context.train_labels,
                outer_context.train_doc_ids,
                outer_context.feature_names,
                outer_context.train_word_counts,
                outer_fold,
                maximum_feature_count,
            )
        )

        for feature_count in feature_count_candidates:
            candidate_rows = fold_terms_df.loc[
                fold_terms_df["selection_rank"] <= feature_count
            ]
            label1_has_outlier = (
                candidate_rows["tf_label1_has_outlier"].fillna(False).astype(bool)
            )
            label0_has_outlier = (
                candidate_rows["tf_label0_has_outlier"].fillna(False).astype(bool)
            )
            fold_summary_rows.append(
                {
                    "outer_fold": outer_fold,
                    "feature_count": feature_count,
                    "evaluated_term_count": len(candidate_rows),
                    "term_count_with_label1_outlier": int(label1_has_outlier.sum()),
                    "term_count_with_label0_outlier": int(label0_has_outlier.sum()),
                    "term_count_with_any_label_outlier": int(
                        (label1_has_outlier | label0_has_outlier).sum()
                    ),
                    "outlier_document_term_count": int(
                        candidate_rows["tf_label1_outlier_count"].sum()
                        + candidate_rows["tf_label0_outlier_count"].sum()
                    ),
                    "label1_zero_iqr_term_count": int(
                        np.isclose(candidate_rows["tf_label1_iqr"], 0.0).sum()
                    ),
                    "label0_zero_iqr_term_count": int(
                        np.isclose(candidate_rows["tf_label0_iqr"], 0.0).sum()
                    ),
                    "both_labels_zero_iqr_term_count": int(
                        (
                            np.isclose(candidate_rows["tf_label1_iqr"], 0.0)
                            & np.isclose(candidate_rows["tf_label0_iqr"], 0.0)
                        ).sum()
                    ),
                }
            )

    terms_path = output_dir / "tf_iqr_terms.csv"
    outliers_path = output_dir / "tf_iqr_outliers.csv"
    fold_summary_path = output_dir / "tf_iqr_fold_summary.csv"
    feature_count_summary_path = output_dir / "tf_iqr_feature_count_summary.csv"
    pd.concat(fold_term_tables, ignore_index=True).to_csv(
        terms_path,
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        all_outlier_rows,
        columns=[
            "outer_fold",
            "feature_count",
            "selection_mode",
            "selection_rank",
            "term",
            "label",
            "doc_id",
            "relative_frequency",
            "q1",
            "q3",
            "iqr",
            "lower_bound",
            "upper_bound",
            "outlier_direction",
        ],
    ).to_csv(outliers_path, index=False, encoding="utf-8-sig")
    fold_summary_df = pd.DataFrame(fold_summary_rows)
    fold_summary_df.to_csv(
        fold_summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    feature_count_summary_df = fold_summary_df.groupby(
        "feature_count",
        as_index=False,
    ).agg(
        evaluated_term_occurrences=("evaluated_term_count", "sum"),
        terms_with_label1_outlier=("term_count_with_label1_outlier", "sum"),
        terms_with_label0_outlier=("term_count_with_label0_outlier", "sum"),
        terms_with_any_label_outlier=("term_count_with_any_label_outlier", "sum"),
        outlier_document_term_count=("outlier_document_term_count", "sum"),
        label1_zero_iqr_term_occurrences=("label1_zero_iqr_term_count", "sum"),
        label0_zero_iqr_term_occurrences=("label0_zero_iqr_term_count", "sum"),
        both_labels_zero_iqr_term_occurrences=(
            "both_labels_zero_iqr_term_count",
            "sum",
        ),
    )
    evaluated_counts = feature_count_summary_df["evaluated_term_occurrences"]
    feature_count_summary_df["any_label_outlier_rate"] = (
        feature_count_summary_df["terms_with_any_label_outlier"] / evaluated_counts
    )
    feature_count_summary_df["label1_zero_iqr_rate"] = (
        feature_count_summary_df["label1_zero_iqr_term_occurrences"]
        / evaluated_counts
    )
    feature_count_summary_df["label0_zero_iqr_rate"] = (
        feature_count_summary_df["label0_zero_iqr_term_occurrences"]
        / evaluated_counts
    )
    feature_count_summary_df.to_csv(
        feature_count_summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    summary_plot_path = save_term_frequency_iqr_summary_plot(
        feature_count_summary_df,
        plots_dir / "tf_iqr_feature_count_summary.png",
    )

    return {
        "terms": str(terms_path),
        "outliers": str(outliers_path),
        "fold_summary": str(fold_summary_path),
        "feature_count_summary": str(feature_count_summary_path),
        "summary_plot": str(summary_plot_path),
        "fold_boxplots": fold_boxplot_paths,
    }


def save_feature_count_term_outputs(
    outer_fold_contexts,
    feature_count_candidates,
    output_dir,
):
    """各TF方式の単語数ごとの実採用語数をouter fold単位と要約で保存する。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    for fold_bundle in outer_fold_contexts:
        outer_fold = fold_bundle["outer_fold"]
        outer_context = fold_bundle["outer_context"]
        for selection_mode, feature_mode in FEATURE_CONDITIONS:
            for feature_count in feature_count_candidates:
                feature_bundle = outer_context.get_feature_bundle(
                    selection_mode,
                    feature_count,
                    feature_mode,
                )
                fold_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "selection_mode": selection_mode,
                        "feature_mode": feature_mode,
                        "feature_count": feature_count,
                        "selected_term_count": feature_bundle["selected_term_count"],
                    }
                )

    fold_counts_path = output_dir / "fold_feature_counts.csv"
    pd.DataFrame(
        fold_rows,
        columns=[
            "outer_fold",
            "selection_mode",
            "feature_mode",
            "feature_count",
            "selected_term_count",
        ],
    ).to_csv(fold_counts_path, index=False, encoding="utf-8-sig")

    # 単語数ごとの平均件数も別 CSV にして、単語数比較しやすくする。
    summary_rows = []
    for selection_mode, feature_mode in FEATURE_CONDITIONS:
        for feature_count in feature_count_candidates:
            count_rows = [
                row
                for row in fold_rows
                if row["selection_mode"] == selection_mode
                and row["feature_mode"] == feature_mode
                and row["feature_count"] == feature_count
            ]
            summary_rows.append(
                {
                    "selection_mode": selection_mode,
                    "feature_mode": feature_mode,
                    "feature_count": feature_count,
                    "mean_selected_term_count": round_half_up(
                        np.mean([row["selected_term_count"] for row in count_rows])
                    ),
                    "selected_term_counts": ",".join(
                        str(row["selected_term_count"]) for row in count_rows
                    ),
                }
            )

    summary_path = output_dir / "feature_count_summary.csv"
    pd.DataFrame(
        summary_rows,
        columns=[
            "selection_mode",
            "feature_mode",
            "feature_count",
            "mean_selected_term_count",
            "selected_term_counts",
        ],
    ).to_csv(summary_path, index=False, encoding="utf-8-sig")

    return {
        "outer_fold_feature_counts": str(fold_counts_path),
        "feature_count_summary": str(summary_path),
    }


def format_condition_label(selection_mode, feature_mode, kernel):
    """条件名をログ表示向けの日本語へ整形する。"""
    selection_mode_label = {
        RELATIVE_TERM_FREQUENCY_MODE: "相対出現頻度",
        LOG_TERM_FREQUENCY_MODE: "対数補正出現頻度",
        TF_S_MODE: "全語彙行L2正規化TF_S",
        LOG_TF_S_MODE: "全語彙行L2正規化log-TF_S",
    }[selection_mode]
    feature_mode_label = {
        COLUMN_MIN_MAX_FEATURE_MODE: "分母込み・列Min-Max",
        ROW_L2_FEATURE_MODE: "分子のみ・行L2",
        FULL_VOCAB_ROW_L2_FEATURE_MODE: "全語彙行L2後の選択列",
        COLUMN_MIN_MAX_DF_GT_2PCT_FEATURE_MODE: (
            "DF>2%語彙・分母込み・列Min-Max"
        ),
        ROW_L2_DF_GT_2PCT_FEATURE_MODE: "DF>2%語彙・分子のみ・行L2",
        FULL_VOCAB_ROW_L2_DF_GT_2PCT_FEATURE_MODE: (
            "DF>2%語彙・全語彙行L2後の選択列"
        ),
    }[feature_mode]
    kernel_label = "LightGBM"
    return f"{selection_mode_label} / {feature_mode_label} / {kernel_label}"


def build_feature_count_summary_row(feature_count, result):
    """1特徴数ぶんの実験結果を summary の1行へまとめる。"""
    row = {
        "feature_count": feature_count,
        "selection_mode": result["selection_mode"],
        "feature_mode": result["feature_mode"],
        "kernel": result["kernel"],
        "mean_selected_term_count": "",
        "selected_term_counts": "",
        "mean_accuracy": "",
        "mean_recall": "",
        "mean_precision": "",
        "mean_f1": "",
        "best_cs": "",
        "best_gammas": "",
        "valid_param_count": len(result.get("valid_param_labels", [])),
        "dropped_param_count": len(result.get("dropped_param_labels", [])),
        "timeout_param_count": len(result.get("timeout_param_labels", [])),
    }

    if result.get("selected_records"):
        row.update(
            {
                "mean_selected_term_count": round_half_up(result["mean_selected_term_count"]),
                "selected_term_counts": ",".join(map(str, result.get("selected_term_counts", []))),
                "mean_accuracy": round_half_up(result["mean_accuracy"]),
                "mean_recall": round_half_up(result["mean_recall"]),
                "mean_precision": round_half_up(result["mean_precision"]),
                "mean_f1": round_half_up(result["mean_f1"]),
                "best_cs": ",".join(map(str, result.get("best_cs", []))),
                "best_gammas": ",".join(map(str, result.get("best_gammas", []))) if "best_gammas" in result else "",
            }
        )

    return row


def save_summary_csv(rows, columns, output_path):
    """列順を固定した summary CSV を保存する。"""
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False, encoding="utf-8-sig")


def save_best_result_csv(best_row, columns, output_path):
    """best result があれば 1 行、なければヘッダーだけの CSV を保存する。"""
    if best_row is None:
        pd.DataFrame(columns=columns).to_csv(output_path, index=False, encoding="utf-8-sig")
        return
    pd.DataFrame([{column: best_row.get(column, "") for column in columns}]).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


def save_feature_count_global_outputs(summary_rows_by_condition, best_results_by_condition, output_root):
    """全特徴数の summary と best result を条件別に保存する。"""
    metrics_root = Path(output_root) / METRICS_DIR_NAME
    metrics_root.mkdir(parents=True, exist_ok=True)

    for selection_mode, feature_mode, kernel in summary_rows_by_condition.keys():
        condition_dir = get_condition_metrics_dir(
            metrics_root,
            selection_mode,
            feature_mode,
            kernel,
        )
        save_summary_csv(
            summary_rows_by_condition[(selection_mode, feature_mode, kernel)],
            FEATURE_COUNT_SUMMARY_COLUMNS,
            condition_dir / "summary.csv",
        )
        save_best_result_csv(
            best_results_by_condition[(selection_mode, feature_mode, kernel)],
            FEATURE_COUNT_BEST_RESULT_COLUMNS,
            condition_dir / "best.csv",
        )


def clear_feature_caches(outer_fold_contexts):
    """特徴数ごとの行列キャッシュを解放し、ワーカーのメモリ増加を抑える。"""
    for fold_bundle in outer_fold_contexts:
        fold_bundle["outer_context"].feature_cache.clear()
        for inner_context in fold_bundle["inner_contexts"]:
            inner_context.feature_cache.clear()


_FEATURE_COUNT_WORKER_OUTER_FOLD_CONTEXTS = None
_FEATURE_COUNT_WORKER_EXPERIMENT_CONDITIONS = None


def _init_feature_count_worker(outer_fold_contexts, experiment_conditions):
    """各ワーカーへ大きなsplit contextを1回だけ渡す。"""
    global _FEATURE_COUNT_WORKER_OUTER_FOLD_CONTEXTS
    global _FEATURE_COUNT_WORKER_EXPERIMENT_CONDITIONS

    _FEATURE_COUNT_WORKER_OUTER_FOLD_CONTEXTS = outer_fold_contexts
    _FEATURE_COUNT_WORKER_EXPERIMENT_CONDITIONS = tuple(experiment_conditions)


def run_feature_count_job(feature_count, outer_fold_contexts, experiment_conditions):
    """1つの特徴数について12特徴条件×2カーネルを評価する。"""
    log_progress(
        f"\n[feature count start] feature_count={feature_count} pid={os.getpid()}"
    )
    condition_results = []

    try:
        for selection_mode, feature_mode, kernel in experiment_conditions:
            result = run_nested_cv_for_condition(
                outer_fold_contexts,
                selection_mode,
                feature_mode,
                kernel,
                feature_count,
            )
            condition_results.append(
                {
                    "selection_mode": selection_mode,
                    "feature_mode": feature_mode,
                    "kernel": kernel,
                    "result": result,
                }
            )
    finally:
        clear_feature_caches(outer_fold_contexts)

    return {
        "feature_count": feature_count,
        "condition_results": condition_results,
    }


def _run_feature_count_worker(feature_count):
    """ProcessPoolExecutorから呼ぶ特徴数単位のジョブ。"""
    if _FEATURE_COUNT_WORKER_OUTER_FOLD_CONTEXTS is None:
        raise RuntimeError("feature count worker was not initialized")
    return run_feature_count_job(
        feature_count,
        _FEATURE_COUNT_WORKER_OUTER_FOLD_CONTEXTS,
        _FEATURE_COUNT_WORKER_EXPERIMENT_CONDITIONS,
    )


def record_feature_count_job_result(
    feature_count_job_result,
    outer_fold_contexts,
    summary_rows_by_condition,
    best_results_by_condition,
    global_margin_source_results,
):
    """特徴数ジョブの結果を親プロセスで保存・集計する。"""
    feature_count = feature_count_job_result["feature_count"]
    feature_count_root = SAVE_DIR / format_feature_count_slug(feature_count)

    for condition_result in feature_count_job_result["condition_results"]:
        selection_mode = condition_result["selection_mode"]
        feature_mode = condition_result["feature_mode"]
        kernel = condition_result["kernel"]
        result = condition_result["result"]

        condition_dir = get_condition_metrics_dir(
            feature_count_root / METRICS_DIR_NAME,
            selection_mode,
            feature_mode,
            kernel,
        )
        save_nested_cv_result_csvs(result, condition_dir, "lightgbm")
        run_optional_output_step(
            result,
            "selected term export",
            save_outer_fold_selected_terms,
            result,
            outer_fold_contexts,
            condition_dir / FEATURE_OUTPUT_DIR_NAME,
        )
        run_optional_output_step(
            result,
            "misclassified document export",
            save_outer_fold_misclassified_documents,
            result,
            condition_dir / MISCLASSIFIED_OUTPUT_DIR_NAME,
        )
        print_result(
            result,
            (
                f"{format_condition_label(selection_mode, feature_mode, kernel)} "
                f"/ 特徴数={feature_count}"
            ),
        )
        global_margin_source_results.append(result)

        summary_row = build_feature_count_summary_row(feature_count, result)
        condition_key = (selection_mode, feature_mode, kernel)
        summary_rows_by_condition[condition_key].append(summary_row)

        if result.get("selected_records"):
            current_best = best_results_by_condition[condition_key]
            if (
                current_best is None
                or result["mean_accuracy"] > current_best["raw_accuracy"]
                or (
                    result["mean_accuracy"] == current_best["raw_accuracy"]
                    and feature_count < current_best["feature_count"]
                )
            ):
                best_results_by_condition[condition_key] = {
                    **summary_row,
                    "raw_accuracy": result["mean_accuracy"],
                }


def sort_feature_count_summary_rows(summary_rows_by_condition):
    """並列完了順で集まったsummaryを特徴数順へ戻す。"""
    for rows in summary_rows_by_condition.values():
        rows.sort(key=lambda row: int(row["feature_count"]))


def print_result(result, heading):
    """1 条件ぶんの結果をコンソールに見やすく表示する。"""
    print(f"\n{heading}")
    if not result.get("selected_records"):
        print(f"margin >= {result['margin_threshold']} を全foldで満たす組み合わせがありませんでした。")
        if "saved_csv_paths" in result:
            print(f"saved metrics csv: {result['saved_csv_paths']['metrics']}")
            print(f"saved selected-records csv: {result['saved_csv_paths']['selected_records']}")
            print(f"saved param-status csv: {result['saved_csv_paths']['param_status']}")
            if "misclassified_all" in result["saved_csv_paths"]:
                print(f"saved misclassified csv: {result['saved_csv_paths']['misclassified_all']}")
                print(f"saved misclassified summary csv: {result['saved_csv_paths']['misclassified_summary']}")
        for warning_message in result.get("output_warnings", []):
            print(f"warning: {warning_message}")
        return

    print(f"feature count: {result['feature_count']}")
    print(f"selected term counts: {result['selected_term_counts']}")
    print(f"accuracy : {result['mean_accuracy']:.4f}")
    print(f"recall   : {result['mean_recall']:.4f}")
    print(f"precision: {result['mean_precision']:.4f}")
    print(f"f1       : {result['mean_f1']:.4f}")
    print(f"selected n_estimators values: {result['best_cs']}")
    if "best_gammas" in result:
        print(
            "selected (num_leaves, max_depth, learning_rate) values: "
            f"{result['best_gammas']}"
        )
    print(f"globally valid params: {result['valid_param_labels']}")
    if "saved_csv_paths" in result:
        print(f"saved metrics csv: {result['saved_csv_paths']['metrics']}")
        print(f"saved selected-records csv: {result['saved_csv_paths']['selected_records']}")
        print(f"saved param-status csv: {result['saved_csv_paths']['param_status']}")
        if "misclassified_all" in result["saved_csv_paths"]:
            print(f"saved misclassified csv: {result['saved_csv_paths']['misclassified_all']}")
            print(f"saved misclassified summary csv: {result['saved_csv_paths']['misclassified_summary']}")
    for warning_message in result.get("output_warnings", []):
        print(f"warning: {warning_message}")


if __name__ == "__main__":
    mp.freeze_support()
    documents, labels, doc_ids, pdf_names = load_corpus()
    global_margin_source_results = []
    log_progress("[main] corpus loaded; building split contexts")
    outer_fold_contexts = build_outer_fold_contexts(documents, labels, doc_ids)

    summary_rows_by_condition = {condition: [] for condition in EXPERIMENT_CONDITIONS}
    best_results_by_condition = {condition: None for condition in EXPERIMENT_CONDITIONS}

    log_progress("\n[main] running fixed-feature-count 10-fold nested CV")
    log_progress(f"[main] feature counts: {FEATURE_COUNT_CANDIDATES}")
    feature_count_worker_count = resolve_worker_count(
        FEATURE_COUNT_WORKERS_ENV,
        DEFAULT_FEATURE_COUNT_WORKER_LIMIT,
        len(FEATURE_COUNT_CANDIDATES),
    )
    log_progress(
        f"[main] feature-count workers: {feature_count_worker_count} "
        f"(override with {FEATURE_COUNT_WORKERS_ENV})"
    )

    if feature_count_worker_count == 1:
        for feature_count in FEATURE_COUNT_CANDIDATES:
            feature_count_job_result = run_feature_count_job(
                feature_count,
                outer_fold_contexts,
                EXPERIMENT_CONDITIONS,
            )
            record_feature_count_job_result(
                feature_count_job_result,
                outer_fold_contexts,
                summary_rows_by_condition,
                best_results_by_condition,
                global_margin_source_results,
            )
    else:
        executor = futures.ProcessPoolExecutor(
            max_workers=feature_count_worker_count,
            mp_context=mp.get_context("spawn"),
            initializer=_init_feature_count_worker,
            initargs=(outer_fold_contexts, EXPERIMENT_CONDITIONS),
        )
        try:
            future_to_feature_count = {
                executor.submit(
                    _run_feature_count_worker,
                    feature_count,
                ): feature_count
                for feature_count in FEATURE_COUNT_CANDIDATES
            }
            completed_feature_counts = 0
            for future in futures.as_completed(future_to_feature_count):
                feature_count = future_to_feature_count[future]
                feature_count_job_result = future.result()
                completed_feature_counts += 1
                log_progress(
                    f"[main] feature count {feature_count} completed "
                    f"({completed_feature_counts}/{len(FEATURE_COUNT_CANDIDATES)}); "
                    "saving outputs"
                )
                record_feature_count_job_result(
                    feature_count_job_result,
                    outer_fold_contexts,
                    summary_rows_by_condition,
                    best_results_by_condition,
                    global_margin_source_results,
                )
        finally:
            executor.shutdown(cancel_futures=True)

    sort_feature_count_summary_rows(summary_rows_by_condition)
    save_feature_count_global_outputs(
        summary_rows_by_condition,
        best_results_by_condition,
        SAVE_DIR,
    )

    feature_count_paths = save_feature_count_term_outputs(
        outer_fold_contexts,
        FEATURE_COUNT_CANDIDATES,
        SAVE_DIR / FEATURE_COUNT_DIR_NAME,
    )
    log_progress(
        f"saved fold feature-count csv: "
        f"{feature_count_paths['outer_fold_feature_counts']}"
    )
    log_progress(
        f"saved feature-count summary csv: "
        f"{feature_count_paths['feature_count_summary']}"
    )
    tf_iqr_paths = save_term_frequency_iqr_audit(
        outer_fold_contexts,
        FEATURE_COUNT_CANDIDATES,
        SAVE_DIR / FEATURE_COUNT_DIR_NAME / "tf_iqr",
    )
    log_progress(f"saved TF IQR term csv: {tf_iqr_paths['terms']}")
    log_progress(f"saved TF IQR outlier csv: {tf_iqr_paths['outliers']}")
    log_progress(f"saved TF IQR fold summary csv: {tf_iqr_paths['fold_summary']}")
    log_progress(
        "saved TF IQR feature-count summary csv: "
        f"{tf_iqr_paths['feature_count_summary']}"
    )
    log_progress(f"saved TF IQR summary plot: {tf_iqr_paths['summary_plot']}")
    log_progress(
        f"saved TF IQR fold boxplots: {len(tf_iqr_paths['fold_boxplots'])} files"
    )

    try:
        global_margin_csv_path = save_global_top_margin_misclassified_documents(
            global_margin_source_results,
            outer_fold_contexts,
            pdf_names,
            SAVE_DIR / GLOBAL_TOP_MARGIN_MISCLASSIFIED_FILENAME,
            GLOBAL_TOP_MARGIN_MISCLASSIFIED_N,
        )
        log_progress(
            f"saved global top-margin misclassified csv: {global_margin_csv_path}"
        )
    except Exception as exc:
        log_progress(
            "[output warning] global top-margin misclassified export failed: "
            f"{type(exc).__name__}: {exc}"
        )
