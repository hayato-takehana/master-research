from pathlib import Path
import concurrent.futures as futures
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
import multiprocessing as mp
import os
import sys
import time

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
    / "differences_between_labels"
    / "term_difference_topn_margin_nested_cv_experiment"
)
SAVE_DIR.mkdir(parents=True, exist_ok=True)
redirect_relative_outputs(SAVE_DIR)

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from dataset_loader import load_document_filenames, load_documents
from text_vectorizer import Tf_idf

FEATURE_COUNT_CANDIDATES = [990, 1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070, 1080, 1090, 1100, 1110, 1120, 1130, 1140, 1150, 1160, 1170, 1180, 1190, 1200, 1210, 1220, 1230, 1240, 1250, 1260, 1270, 1280, 1290, 1300, 1310, 1320, 1330, 1340, 1350, 1360, 1370, 1380, 1390, 1400, 1410, 1420, 1430, 1440, 1450, 1460, 1470, 1480, 1490, 1500, 1510, 1520, 1530, 1540, 1550, 1560, 1570, 1580, 1590, 1600, 1610, 1620, 1630, 1640, 1650, 1660, 1670, 1680, 1690, 1700, 1710, 1720, 1730, 1740, 1750, 1760, 1770, 1780, 1790, 1800, 1810, 1820, 1830, 1840, 1850, 1860, 1870, 1880, 1890, 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100, 2110, 2120, 2130, 2140, 2150, 2160, 2170, 2180, 2190, 2200, 2210, 2220, 2230, 2240, 2250, 2260, 2270, 2280, 2290, 2300, 2310, 2320, 2330, 2340, 2350, 2360, 2370, 2380, 2390, 2400, 2410, 2420, 2430, 2440, 2450, 2460, 2470, 2480, 2490, 2500, 2510, 2520, 2530, 2540, 2550, 2560, 2570, 2580, 2590, 2600, 2610, 2620, 2630, 2640, 2650, 2660, 2670, 2680, 2690, 2700, 2710, 2720, 2730, 2740, 2750, 2760, 2770, 2780, 2790, 2800, 2810, 2820, 2830, 2840, 2850, 2860, 2870, 2880, 2890, 2900, 2910, 2920, 2930, 2940, 2950, 2960, 2970, 2980, 2990, 3000]
SELECTION_MODES = (
    "document_frequency",
    "term_frequency",
    "tfidf",
    "combined_unique",
    "combined_duplicate",
    "document_tfidf_unique",
    "document_tfidf_duplicate",
)
SOURCE_SPECIFIC_FEATURE_MODE = "source_specific"
KERNELS = ("linear", "rbf")
EXPERIMENT_CONDITIONS = tuple(
    (selection_mode, SOURCE_SPECIFIC_FEATURE_MODE, kernel)
    for selection_mode in SELECTION_MODES
    for kernel in KERNELS
)
MIN_LEN = 0
MIN_DF = 0.0
TFIDF_MIN_LEN = 3
TFIDF_MIN_DF = 0.02
USE_STEMMING = True
KEEP_DIGITS = False
PERCENT_MODE = "drop"
GLOBAL_TOP_MARGIN_MISCLASSIFIED_N = 10
GLOBAL_TOP_MARGIN_MISCLASSIFIED_FILENAME = "global_top_margin_misclassified.csv"

LINEAR_C_VALUES = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
RBF_C_VALUES = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
RBF_GAMMA_VALUES = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
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
    SOURCE_SPECIFIC_FEATURE_MODE: "src",
}
SELECTION_MODE_DIR_NAMES = {
    "term_frequency": "tf",
    "document_frequency": "df",
    "tfidf": "tfidf",
    "combined_unique": "mix_unique",
    "combined_duplicate": "mix_duplicate",
    "document_tfidf_unique": "df_tfidf_unique",
    "document_tfidf_duplicate": "df_tfidf_duplicate",
}
KERNEL_DIR_NAMES = {
    "linear": "lin",
    "rbf": "rbf",
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
    )
    # `load_documents` と同じフォルダ走査順で PDF 名も取得し、doc_id と対応づける。
    pdf_names_1, pdf_names_0 = load_document_filenames(real_scam=False, sort_files=False)
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


def vectorize_train_test_tfidf_documents(train_docs, test_docs):
    """TF-IDF専用設定でtrain側だけ語彙とIDFを学習し、train/testを変換する。"""
    train_vectorizer = Tf_idf(list(train_docs), True, TFIDF_MIN_LEN, use_stemming=USE_STEMMING)
    X_train_tfidf, feature_names, fitted_tfidf_vectorizer = train_vectorizer.tf_idf(
        TFIDF_MIN_DF,
        ngram_range=(1, 1),
    )

    test_vectorizer = Tf_idf(list(test_docs), True, TFIDF_MIN_LEN, use_stemming=USE_STEMMING)
    X_test_tfidf = fitted_tfidf_vectorizer.transform(test_vectorizer.processed_docs)

    return (
        X_train_tfidf.tocsr(),
        X_test_tfidf.tocsr(),
        np.array(feature_names),
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


def _binary_presence_matrix(X_count):
    X_presence = X_count.copy()
    X_presence.data = np.ones_like(X_presence.data, dtype=float)
    return X_presence


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


def build_term_frequency_difference_table(X_count, labels, feature_names):
    """文書長とラベル文書数を考慮した平均相対出現頻度差を計算する。"""
    label1_mask = labels == 1
    label0_mask = labels == 0
    if not np.any(label1_mask) or not np.any(label0_mask):
        raise ValueError("Both label 1 and label 0 documents are required.")

    doc_lengths = _document_lengths_from_counts(X_count)
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


def build_document_frequency_difference_table(X_count, labels, feature_names):
    """ラベル文書数を考慮した文書出現率差を計算する。"""
    label1_mask = labels == 1
    label0_mask = labels == 0
    label1_doc_count = int(np.sum(label1_mask))
    label0_doc_count = int(np.sum(label0_mask))
    if label1_doc_count == 0 or label0_doc_count == 0:
        raise ValueError("Both label 1 and label 0 documents are required.")

    X_presence = _binary_presence_matrix(X_count)
    label1_values = _sparse_column_sums(X_presence[label1_mask]) / label1_doc_count
    label0_values = _sparse_column_sums(X_presence[label0_mask]) / label0_doc_count
    return _build_difference_table(
        "mean_document_frequency_per_document_difference",
        "abs(sum_presence_label1 / n_label1 - sum_presence_label0 / n_label0)",
        feature_names,
        label1_values,
        label0_values,
    )


def build_tfidf_difference_table(X_tfidf, labels, feature_names):
    """ラベルごとの平均TF-IDF値の差を計算する。"""
    label1_mask = labels == 1
    label0_mask = labels == 0
    label1_doc_count = int(np.sum(label1_mask))
    label0_doc_count = int(np.sum(label0_mask))
    if label1_doc_count == 0 or label0_doc_count == 0:
        raise ValueError("Both label 1 and label 0 documents are required.")

    label1_values = _sparse_column_sums(X_tfidf[label1_mask]) / label1_doc_count
    label0_values = _sparse_column_sums(X_tfidf[label0_mask]) / label0_doc_count
    return _build_difference_table(
        "tfidf_average_difference",
        "abs(mean_tfidf_label1 - mean_tfidf_label0)",
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


def _prepare_selected_source_rows(difference_df, selection_source, feature_count):
    selected_df = _select_top_unique_terms(difference_df, feature_count)
    selected_df["selection_source"] = selection_source
    selected_df["selection_rank_within_source"] = np.arange(1, feature_count + 1)
    return selected_df


def _select_combined_unique_terms(
    first_source_df,
    first_source,
    second_source_df,
    second_source,
    feature_count,
    equal_rank_preferred_source="document_frequency",
):
    if feature_count % 2 != 0:
        raise ValueError("combined_unique selection requires an even feature_count.")

    per_source_count = feature_count // 2
    sources = (first_source, second_source)
    ranked_tables = {
        first_source: first_source_df.reset_index(drop=True),
        second_source: second_source_df.reset_index(drop=True),
    }
    rank_lookup = {
        source: {
            term: rank
            for rank, term in enumerate(table["term"].astype(str), start=1)
        }
        for source, table in ranked_tables.items()
    }
    selected_by_source = {
        first_source: [],
        second_source: [],
    }
    owner_by_term = {}
    next_index = {
        first_source: 0,
        second_source: 0,
    }
    overlap_resolution_by_term = {}

    while any(len(rows) < per_source_count for rows in selected_by_source.values()):
        made_progress = False
        for source in sources:
            table = ranked_tables[source]
            while len(selected_by_source[source]) < per_source_count:
                if next_index[source] >= len(table):
                    raise ValueError(
                        f"Not enough unique terms for combined_unique: source={source}, "
                        f"requested_per_source={per_source_count}"
                    )

                row = table.iloc[next_index[source]].copy()
                next_index[source] += 1
                term = str(row["term"])
                source_rank = rank_lookup[source][term]
                current_owner = owner_by_term.get(term)

                if current_owner is None:
                    row["selection_source"] = source
                    row["selection_rank_within_source"] = source_rank
                    selected_by_source[source].append(row)
                    owner_by_term[term] = source
                    made_progress = True
                    continue

                if current_owner == source:
                    continue

                other_source = current_owner
                other_rank = rank_lookup[other_source][term]
                source_wins = source_rank < other_rank
                equal_rank = source_rank == other_rank
                if equal_rank:
                    if equal_rank_preferred_source in sources:
                        source_wins = source == equal_rank_preferred_source
                    else:
                        source_wins = source == first_source

                winning_source = source if source_wins else other_source
                if equal_rank:
                    resolution = f"{winning_source}_preferred_on_equal_rank"
                else:
                    resolution = f"{winning_source}_kept_by_higher_rank"
                overlap_resolution = {
                    f"overlap_{first_source}_rank": rank_lookup[first_source].get(term, np.nan),
                    f"overlap_{second_source}_rank": rank_lookup[second_source].get(term, np.nan),
                    "overlap_resolution": resolution,
                    "equal_rank_document_frequency_priority": (
                        equal_rank and winning_source == "document_frequency"
                    ),
                }
                overlap_resolution_by_term[term] = overlap_resolution

                if not source_wins:
                    continue

                selected_by_source[other_source] = [
                    selected_row
                    for selected_row in selected_by_source[other_source]
                    if str(selected_row["term"]) != term
                ]
                row["selection_source"] = source
                row["selection_rank_within_source"] = source_rank
                selected_by_source[source].append(row)
                owner_by_term[term] = source
                made_progress = True

        if not made_progress:
            raise RuntimeError("combined_unique selection could not fill both source quotas.")

    selected_terms_df = pd.DataFrame(
        selected_by_source[first_source] + selected_by_source[second_source]
    ).reset_index(drop=True)
    for source in ("term_frequency", "document_frequency", "tfidf"):
        selected_terms_df[f"overlap_{source}_rank"] = np.nan
    selected_terms_df["overlap_resolution"] = ""
    selected_terms_df["equal_rank_document_frequency_priority"] = False
    for row_index, term in selected_terms_df["term"].astype(str).items():
        resolution = overlap_resolution_by_term.get(term)
        if resolution is None:
            continue
        for column, value in resolution.items():
            selected_terms_df.at[row_index, column] = value

    return selected_terms_df


def select_terms_by_mode(
    term_frequency_df,
    document_frequency_df,
    tfidf_df,
    selection_mode,
    feature_count,
):
    """差分絶対値の上位語を、指定した方式で採用する。"""
    if feature_count <= 0:
        raise ValueError("feature_count must be positive.")

    if selection_mode == "term_frequency":
        selected_terms_df = _prepare_selected_source_rows(
            term_frequency_df,
            "term_frequency",
            feature_count,
        )
    elif selection_mode == "document_frequency":
        selected_terms_df = _prepare_selected_source_rows(
            document_frequency_df,
            "document_frequency",
            feature_count,
        )
    elif selection_mode == "tfidf":
        selected_terms_df = _prepare_selected_source_rows(
            tfidf_df,
            "tfidf",
            feature_count,
        )
    elif selection_mode == "combined_unique":
        selected_terms_df = _select_combined_unique_terms(
            term_frequency_df,
            "term_frequency",
            document_frequency_df,
            "document_frequency",
            feature_count,
        )
    elif selection_mode == "combined_duplicate":
        if feature_count % 2 != 0:
            raise ValueError("combined_duplicate selection requires an even feature_count.")
        per_source_count = feature_count // 2
        frequency_terms_df = _prepare_selected_source_rows(
            term_frequency_df,
            "term_frequency",
            per_source_count,
        )
        document_terms_df = _prepare_selected_source_rows(
            document_frequency_df,
            "document_frequency",
            per_source_count,
        )
        selected_terms_df = pd.concat(
            [frequency_terms_df, document_terms_df],
            ignore_index=True,
        )
    elif selection_mode == "document_tfidf_unique":
        selected_terms_df = _select_combined_unique_terms(
            document_frequency_df,
            "document_frequency",
            tfidf_df,
            "tfidf",
            feature_count,
        )
    elif selection_mode == "document_tfidf_duplicate":
        if feature_count % 2 != 0:
            raise ValueError("document_tfidf_duplicate selection requires an even feature_count.")
        per_source_count = feature_count // 2
        document_terms_df = _prepare_selected_source_rows(
            document_frequency_df,
            "document_frequency",
            per_source_count,
        )
        tfidf_terms_df = _prepare_selected_source_rows(
            tfidf_df,
            "tfidf",
            per_source_count,
        )
        selected_terms_df = pd.concat(
            [document_terms_df, tfidf_terms_df],
            ignore_index=True,
        )
    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")

    selected_terms_df.insert(0, "selection_mode", selection_mode)
    selected_terms_df.insert(1, "selection_rank", np.arange(1, len(selected_terms_df) + 1))
    return selected_terms_df


def _relative_frequency_matrix(X_count, document_word_counts):
    document_word_counts = np.asarray(document_word_counts, dtype=float)
    inv_word_counts = np.divide(
        1.0,
        document_word_counts,
        out=np.zeros_like(document_word_counts, dtype=float),
        where=document_word_counts > 0,
    )
    return X_count.multiply(inv_word_counts[:, None]).tocsr()


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
    # train範囲を超えるtest値も、文書頻度と同じ0〜1範囲にそろえる。
    return (
        np.clip(scaled_train, 0.0, 1.0),
        np.clip(scaled_test, 0.0, 1.0),
        train_min,
        train_max,
    )


def build_selected_feature_matrices(
    X_train_count,
    X_test_count,
    count_feature_names,
    X_train_tfidf,
    X_test_tfidf,
    tfidf_feature_names,
    selected_terms_df,
    train_word_counts,
    test_word_counts,
):
    """選択元に応じ、相対出現頻度・文書出現有無・TF-IDF特徴列を作る。"""
    selected_terms = selected_terms_df["term"].tolist()
    count_vocab_index = {term: idx for idx, term in enumerate(count_feature_names)}
    tfidf_vocab_index = {term: idx for idx, term in enumerate(tfidf_feature_names)}
    selection_sources = selected_terms_df["selection_source"].tolist()
    feature_count = len(selected_terms)
    X_train_features = np.zeros((X_train_count.shape[0], feature_count), dtype=float)
    X_test_features = np.zeros((X_test_count.shape[0], feature_count), dtype=float)
    train_mins = np.zeros(feature_count, dtype=float)
    train_maxs = np.ones(feature_count, dtype=float)
    transform_names = np.empty(feature_count, dtype=object)

    term_frequency_positions = [
        position
        for position, source in enumerate(selection_sources)
        if source == "term_frequency"
    ]
    if term_frequency_positions:
        term_frequency_indices = [
            count_vocab_index[selected_terms[position]]
            for position in term_frequency_positions
        ]
        X_train_relative = _relative_frequency_matrix(X_train_count, train_word_counts)
        X_test_relative = _relative_frequency_matrix(X_test_count, test_word_counts)
        train_relative_values = X_train_relative[:, term_frequency_indices].toarray()
        test_relative_values = X_test_relative[:, term_frequency_indices].toarray()
        scaled_train, scaled_test, source_mins, source_maxs = _min_max_scale_columns_from_train(
            train_relative_values,
            test_relative_values,
        )
        X_train_features[:, term_frequency_positions] = scaled_train
        X_test_features[:, term_frequency_positions] = scaled_test
        train_mins[term_frequency_positions] = source_mins
        train_maxs[term_frequency_positions] = source_maxs
        transform_names[term_frequency_positions] = "relative_frequency_min_max"

    document_frequency_positions = [
        position
        for position, source in enumerate(selection_sources)
        if source == "document_frequency"
    ]
    if document_frequency_positions:
        document_frequency_indices = [
            count_vocab_index[selected_terms[position]]
            for position in document_frequency_positions
        ]
        X_train_features[:, document_frequency_positions] = (
            X_train_count[:, document_frequency_indices] != 0
        ).astype(np.int8).toarray()
        X_test_features[:, document_frequency_positions] = (
            X_test_count[:, document_frequency_indices] != 0
        ).astype(np.int8).toarray()
        transform_names[document_frequency_positions] = "document_presence_binary"

    tfidf_positions = [
        position
        for position, source in enumerate(selection_sources)
        if source == "tfidf"
    ]
    if tfidf_positions:
        tfidf_indices = [
            tfidf_vocab_index[selected_terms[position]]
            for position in tfidf_positions
        ]
        train_tfidf_values = X_train_tfidf[:, tfidf_indices].toarray()
        test_tfidf_values = X_test_tfidf[:, tfidf_indices].toarray()
        X_train_features[:, tfidf_positions] = train_tfidf_values
        X_test_features[:, tfidf_positions] = test_tfidf_values
        train_mins[tfidf_positions] = np.min(train_tfidf_values, axis=0)
        train_maxs[tfidf_positions] = np.max(train_tfidf_values, axis=0)
        transform_names[tfidf_positions] = "tfidf"

    source_suffixes = {
        "term_frequency": "tf",
        "document_frequency": "df",
        "tfidf": "tfidf",
    }
    feature_names_with_source = [
        f"{term}__{source_suffixes[source]}"
        for term, source in zip(selected_terms, selection_sources)
    ]

    return {
        "selected_terms": selected_terms,
        "selected_feature_names": feature_names_with_source,
        "feature_train_min": train_mins,
        "feature_train_max": train_maxs,
        "feature_transform": transform_names,
        SOURCE_SPECIFIC_FEATURE_MODE: {
            "train": X_train_features,
            "test": X_test_features,
        },
    }


class SplitFeatureContext:
    """1 つの train/test split に必要な特徴量関連情報をまとめて持つコンテナ。"""

    def __init__(self, train_docs, train_labels, test_docs, test_labels, train_doc_ids, test_doc_ids):
        """split の学習データだけで語彙と2種類の差分表を作る。"""
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
        (
            self.X_train_tfidf,
            self.X_test_tfidf,
            self.tfidf_feature_names,
        ) = vectorize_train_test_tfidf_documents(self.train_docs, self.test_docs)
        self.term_frequency_df = sort_difference_table(
            build_term_frequency_difference_table(
                self.X_train_count,
                self.train_labels,
                self.feature_names,
            )
        )
        self.document_frequency_df = sort_difference_table(
            build_document_frequency_difference_table(
                self.X_train_count,
                self.train_labels,
                self.feature_names,
            )
        )
        self.tfidf_df = sort_difference_table(
            build_tfidf_difference_table(
                self.X_train_tfidf,
                self.train_labels,
                self.tfidf_feature_names,
            )
        )
        self.feature_cache = {}

    def get_feature_bundle(self, selection_mode, feature_count):
        """指定方式・単語数の採用語と特徴行列をキャッシュ付きで返す。"""
        cache_key = (selection_mode, feature_count)
        if cache_key not in self.feature_cache:
            selected_terms_df = select_terms_by_mode(
                self.term_frequency_df,
                self.document_frequency_df,
                self.tfidf_df,
                selection_mode,
                feature_count,
            )
            feature_matrices = build_selected_feature_matrices(
                self.X_train_count,
                self.X_test_count,
                self.feature_names,
                self.X_train_tfidf,
                self.X_test_tfidf,
                self.tfidf_feature_names,
                selected_terms_df,
                self.train_word_counts,
                self.test_word_counts,
            )
            selected_terms_df = selected_terms_df.copy()
            selected_terms_df["feature_name"] = feature_matrices["selected_feature_names"]
            selected_terms_df["feature_transform"] = feature_matrices["feature_transform"]
            selected_terms_df["train_min"] = feature_matrices["feature_train_min"]
            selected_terms_df["train_max"] = feature_matrices["feature_train_max"]
            source_counts = selected_terms_df["selection_source"].value_counts()
            unique_term_count = int(selected_terms_df["term"].nunique())
            self.feature_cache[cache_key] = {
                "selection_mode": selection_mode,
                "feature_count": feature_count,
                "selected_terms_df": selected_terms_df,
                "selected_term_count": len(feature_matrices["selected_terms"]),
                "unique_term_count": unique_term_count,
                "duplicate_feature_count": len(feature_matrices["selected_terms"]) - unique_term_count,
                "term_frequency_selected_count": int(source_counts.get("term_frequency", 0)),
                "document_frequency_selected_count": int(source_counts.get("document_frequency", 0)),
                "tfidf_selected_count": int(source_counts.get("tfidf", 0)),
                "equal_rank_document_frequency_priority_count": int(
                    selected_terms_df.get(
                        "equal_rank_document_frequency_priority",
                        pd.Series(False, index=selected_terms_df.index),
                    ).sum()
                ),
                "feature_matrices": feature_matrices,
            }
        return self.feature_cache[cache_key]


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


def _build_svc(kernel, c, gamma=None):
    """指定カーネルとハイパーパラメータから SVC を生成する。"""
    kwargs = {
        "kernel": kernel,
        "C": c,
        "class_weight": CLASS_WEIGHT,
        "tol": TOL,
    }
    if gamma is not None:
        kwargs["gamma"] = gamma
    return svm.SVC(**kwargs)


def _compute_margins(model, X_train, y_train):
    """学習データ上の signed margin を計算する。"""
    decision_values = model.decision_function(X_train)
    y_signed = np.where(y_train == 0, -1, 1)
    return y_signed * decision_values


def format_candidate_label(selection_mode, feature_count, c, gamma=None):
    """単語選択方式・単語数・SVMパラメータをログ用文字列にする。"""
    if gamma is None:
        return f"selection={selection_mode}, feature_count={feature_count}, C={c}"
    return f"selection={selection_mode}, feature_count={feature_count}, C={c}, gamma={gamma}"


def format_label_name(label):
    """0/1 ラベルを人が読める名前へ変換する。"""
    return "label1" if label == 1 else "label0"


def fit_predict_with_margin_check(kernel, c, gamma, X_train, y_train, X_test):
    """1回分のSVM学習・margin判定・予測を同一プロセス内で行う。"""
    try:
        model = _build_svc(kernel, c, gamma=gamma)
        model.fit(X_train, y_train)

        margins = _compute_margins(model, X_train, y_train)
        if not np.all(margins >= MARGIN_THRESHOLD):
            return {"status": "invalid_margin"}

        y_pred = model.predict(X_test)
        decision_values = np.asarray(model.decision_function(X_test)).ravel()
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

    train 側でのみ語を採用し、その語で train/test を特徴化した後に SVM を学習する。
    margin 条件を満たさない候補はここで `None` として落とす。
    """
    feature_bundle = split_context.get_feature_bundle(selection_mode, feature_count)
    selected_term_count = feature_bundle["selected_term_count"]
    if selected_term_count == 0:
        return None

    X_train = feature_bundle["feature_matrices"][feature_mode]["train"]
    X_test = feature_bundle["feature_matrices"][feature_mode]["test"]

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
        "unique_term_count": feature_bundle["unique_term_count"],
        "duplicate_feature_count": feature_bundle["duplicate_feature_count"],
        "term_frequency_selected_count": feature_bundle["term_frequency_selected_count"],
        "document_frequency_selected_count": feature_bundle["document_frequency_selected_count"],
        "tfidf_selected_count": feature_bundle["tfidf_selected_count"],
        "equal_rank_document_frequency_priority_count": feature_bundle[
            "equal_rank_document_frequency_priority_count"
        ],
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
                "unique_term_count": outer_result["unique_term_count"],
                "duplicate_feature_count": outer_result["duplicate_feature_count"],
                "term_frequency_selected_count": outer_result[
                    "term_frequency_selected_count"
                ],
                "document_frequency_selected_count": outer_result[
                    "document_frequency_selected_count"
                ],
                "tfidf_selected_count": outer_result["tfidf_selected_count"],
                "equal_rank_document_frequency_priority_count": outer_result[
                    "equal_rank_document_frequency_priority_count"
                ],
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
    """同一条件のSVM候補を順次評価する常駐ワーカー。"""
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
        "unique_term_counts": [],
        "duplicate_feature_counts": [],
        "misclassified_counts": [],
    }
    if kernel == "rbf":
        result["best_gammas"] = []
    return result


def run_nested_cv_for_condition(
    outer_fold_contexts,
    selection_mode,
    feature_mode,
    kernel,
    feature_count,
):
    """固定した特徴数で nested CV を回し、fold ごとの最良SVMパラメータを選ぶ。"""
    c_values = LINEAR_C_VALUES if kernel == "linear" else RBF_C_VALUES
    gamma_candidates = [None] if kernel == "linear" else RBF_GAMMA_VALUES
    total_candidates = len(c_values) * len(gamma_candidates)
    condition_label = format_condition_label(selection_mode, feature_mode, kernel)
    condition_start = time.perf_counter()

    valid_candidate_results = {}
    dropped_param_labels = []
    timeout_param_labels = []
    processed_candidates = 0

    log_progress(
        f"[condition start] {condition_label}: "
        f"feature_count={feature_count}, {total_candidates} SVM parameter settings"
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
    result["unique_term_counts"] = [record["unique_term_count"] for record in selected_records]
    result["duplicate_feature_counts"] = [
        record["duplicate_feature_count"] for record in selected_records
    ]
    if kernel == "rbf":
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
    )
    X_train = feature_bundle["feature_matrices"][feature_mode]["train"]
    X_test = feature_bundle["feature_matrices"][feature_mode]["test"]

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
        "unique_term_counts": ",".join(map(str, result.get("unique_term_counts", []))),
        "duplicate_feature_counts": ",".join(
            map(str, result.get("duplicate_feature_counts", []))
        ),
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
            "unique_term_count",
            "duplicate_feature_count",
            "term_frequency_selected_count",
            "document_frequency_selected_count",
            "tfidf_selected_count",
            "equal_rank_document_frequency_priority_count",
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
        feature_bundle = outer_fold_contexts[outer_fold - 1]["outer_context"].get_feature_bundle(
            record["selection_mode"],
            record["feature_count"],
        )
        selected_terms_df = feature_bundle["selected_terms_df"].copy()
        selected_terms_df.insert(0, "outer_fold", outer_fold)
        selected_terms_df.insert(1, "feature_count", record["feature_count"])
        file_path = output_dir / f"f{outer_fold:02d}_terms.csv"
        selected_terms_df.to_csv(file_path, index=False, encoding="utf-8-sig")
        # どの fold でどの単語数・何語採用されたかだけを一覧でも持っておく。
        summary_rows.append(
            {
                "outer_fold": outer_fold,
                "selection_mode": record["selection_mode"],
                "feature_count": record["feature_count"],
                "selected_term_count": record["selected_term_count"],
                "unique_term_count": feature_bundle["unique_term_count"],
                "duplicate_feature_count": feature_bundle["duplicate_feature_count"],
                "term_frequency_selected_count": feature_bundle["term_frequency_selected_count"],
                "document_frequency_selected_count": feature_bundle["document_frequency_selected_count"],
                "tfidf_selected_count": feature_bundle["tfidf_selected_count"],
                "equal_rank_document_frequency_priority_count": feature_bundle[
                    "equal_rank_document_frequency_priority_count"
                ],
                "file_path": str(file_path),
            }
        )

    pd.DataFrame(
        summary_rows,
        columns=[
            "outer_fold",
            "selection_mode",
            "feature_count",
            "selected_term_count",
            "unique_term_count",
            "duplicate_feature_count",
            "term_frequency_selected_count",
            "document_frequency_selected_count",
            "tfidf_selected_count",
            "equal_rank_document_frequency_priority_count",
            "file_path",
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


def save_feature_count_term_outputs(
    outer_fold_contexts,
    selection_modes,
    feature_count_candidates,
    output_dir,
):
    """方式・単語数ごとの実採用語数を outer fold 単位と要約で保存する。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    for fold_bundle in outer_fold_contexts:
        outer_fold = fold_bundle["outer_fold"]
        outer_context = fold_bundle["outer_context"]
        for selection_mode in selection_modes:
            for feature_count in feature_count_candidates:
                feature_bundle = outer_context.get_feature_bundle(selection_mode, feature_count)
                fold_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "selection_mode": selection_mode,
                        "feature_count": feature_count,
                        "selected_term_count": feature_bundle["selected_term_count"],
                        "unique_term_count": feature_bundle["unique_term_count"],
                        "duplicate_feature_count": feature_bundle["duplicate_feature_count"],
                        "term_frequency_selected_count": feature_bundle["term_frequency_selected_count"],
                        "document_frequency_selected_count": feature_bundle[
                            "document_frequency_selected_count"
                        ],
                        "tfidf_selected_count": feature_bundle["tfidf_selected_count"],
                        "equal_rank_document_frequency_priority_count": feature_bundle[
                            "equal_rank_document_frequency_priority_count"
                        ],
                    }
                )

    fold_counts_path = output_dir / "fold_feature_counts.csv"
    if any(
        row["equal_rank_document_frequency_priority_count"] > 0
        for row in fold_rows
    ):
        log_progress(
            "[selection note] 重複なし方式で差分順位が同じ語は、"
            "文書頻度特徴を優先しました。"
        )
    pd.DataFrame(
        fold_rows,
        columns=[
            "outer_fold",
            "selection_mode",
            "feature_count",
            "selected_term_count",
            "unique_term_count",
            "duplicate_feature_count",
            "term_frequency_selected_count",
            "document_frequency_selected_count",
            "tfidf_selected_count",
            "equal_rank_document_frequency_priority_count",
        ],
    ).to_csv(fold_counts_path, index=False, encoding="utf-8-sig")

    # 単語数ごとの平均件数も別 CSV にして、単語数比較しやすくする。
    summary_rows = []
    for selection_mode in selection_modes:
        for feature_count in feature_count_candidates:
            count_rows = [
                row
                for row in fold_rows
                if row["selection_mode"] == selection_mode
                and row["feature_count"] == feature_count
            ]
            summary_rows.append(
                {
                    "selection_mode": selection_mode,
                    "feature_count": feature_count,
                    "mean_selected_term_count": round_half_up(
                        np.mean([row["selected_term_count"] for row in count_rows])
                    ),
                    "mean_unique_term_count": round_half_up(
                        np.mean([row["unique_term_count"] for row in count_rows])
                    ),
                    "mean_duplicate_feature_count": round_half_up(
                        np.mean([row["duplicate_feature_count"] for row in count_rows])
                    ),
                    "mean_term_frequency_selected_count": round_half_up(
                        np.mean([row["term_frequency_selected_count"] for row in count_rows])
                    ),
                    "mean_document_frequency_selected_count": round_half_up(
                        np.mean([row["document_frequency_selected_count"] for row in count_rows])
                    ),
                    "mean_tfidf_selected_count": round_half_up(
                        np.mean([row["tfidf_selected_count"] for row in count_rows])
                    ),
                    "equal_rank_document_frequency_priority_counts": ",".join(
                        str(row["equal_rank_document_frequency_priority_count"])
                        for row in count_rows
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
            "feature_count",
            "mean_selected_term_count",
            "mean_unique_term_count",
            "mean_duplicate_feature_count",
            "mean_term_frequency_selected_count",
            "mean_document_frequency_selected_count",
            "mean_tfidf_selected_count",
            "equal_rank_document_frequency_priority_counts",
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
        "term_frequency": "出現頻度のみ",
        "document_frequency": "文書頻度のみ",
        "tfidf": "TF-IDFのみ",
        "combined_unique": "出現頻度+文書頻度（重複なし）",
        "combined_duplicate": "出現頻度+文書頻度（重複あり）",
        "document_tfidf_unique": "文書頻度+TF-IDF（重複なし）",
        "document_tfidf_duplicate": "文書頻度+TF-IDF（重複あり）",
    }[selection_mode]
    feature_mode_label = "選択元別の特徴変換"
    kernel_label = "線形SVM" if kernel == "linear" else "非線形SVM"
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
    """1つの特徴数について4方式×2カーネルを評価する。"""
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
        save_nested_cv_result_csvs(result, condition_dir, "svm")
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
    print(f"selected C values: {result['best_cs']}")
    if "best_gammas" in result:
        print(f"selected gamma values: {result['best_gammas']}")
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
        SELECTION_MODES,
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
