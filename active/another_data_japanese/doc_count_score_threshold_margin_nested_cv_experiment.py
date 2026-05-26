from __future__ import annotations

from pathlib import Path
import importlib
import json
import os
import re
import sys


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
SAVE_DIR = PROJECT_ROOT / "data" / "outputs" / "another_data_japanese" / "dcstmnce_edinet_text_only"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
redirect_relative_outputs(SAVE_DIR, override=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402


DATASET_NAME = "SakanaAI/EDINET-Bench"
CONFIG_NAME = "fraud_detection"
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
ID_COLUMNS = ("doc_id", "ammended_doc_id", "edinet_code")
ENGINE_MODULE_NAME = "active.another_data.doc_count_score_threshold_margin_nested_cv_experiment"
JAPANESE_STOP_WORDS = {
    "これ",
    "それ",
    "ため",
    "もの",
    "こと",
    "よう",
    "及び",
    "また",
    "なお",
    "当社",
    "当該",
    "これら",
    "について",
    "により",
    "による",
    "として",
    "おり",
    "ます",
    "です",
}
TOKEN_PATTERN = re.compile(
    r"[一-龯々〆ヵヶ]+|[ァ-ヴー]+|[ぁ-ん]+|[A-Za-z][A-Za-z0-9_]+|[0-9]+(?:\.[0-9]+)?"
)


def load_edinet_dataset():
    """EDINET-Bench fraud_detection を読み込む。

    特徴量として使う列は `text` のみ。`label` は教師ラベルとしてだけ使う。
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required to load SakanaAI/EDINET-Bench. "
            "Install it with `pip install datasets` in this environment."
        ) from exc

    dataset_dict = load_dataset(DATASET_NAME, CONFIG_NAME)
    frames = []
    for split_name, dataset_split in dataset_dict.items():
        required_columns = [TEXT_COLUMN, LABEL_COLUMN]
        missing_columns = [column for column in required_columns if column not in dataset_split.column_names]
        if missing_columns:
            raise ValueError(f"{split_name} split is missing required columns: {missing_columns}")

        available_id_columns = [column for column in ID_COLUMNS if column in dataset_split.column_names]
        columns = [TEXT_COLUMN, LABEL_COLUMN, *available_id_columns]
        frame = dataset_split.select_columns(columns).to_pandas()
        frame.insert(0, "source_split", split_name)
        frames.append(frame)

    if not frames:
        raise ValueError(f"{DATASET_NAME}/{CONFIG_NAME} did not contain any splits.")

    return pd.concat(frames, ignore_index=True)


def iter_leaf_text_values(value):
    """EDINET `text` の JSON から、キー名ではなく本文値だけを再帰的に取り出す。"""
    if isinstance(value, str):
        if value.strip():
            yield value
        return
    if isinstance(value, dict):
        for child in value.values():
            yield from iter_leaf_text_values(child)
        return
    if isinstance(value, list):
        for child in value:
            yield from iter_leaf_text_values(child)


def extract_text_only(raw_text):
    """`text` 列の JSON 文字列から本文だけを結合し、JSON キーを特徴に混ぜない。"""
    text = str(raw_text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    extracted = " ".join(iter_leaf_text_values(parsed))
    return extracted if extracted else text


def japanese_tokenize(text):
    """日本語文を TF-IDF/出現頻度で扱えるよう、表記種別ごとの粗い語に分割する。"""
    normalized_text = re.sub(r"[0-9０-９]+(?:年|月|日)?", " ", str(text).lower())
    tokens = []
    for match in TOKEN_PATTERN.finditer(normalized_text):
        token = match.group(0)
        if token in JAPANESE_STOP_WORDS:
            continue
        if re.fullmatch(r"[ぁ-ん]+", token):
            continue
        if token.isdigit():
            continue
        if len(token) < 2:
            continue
        tokens.append(token)
    return tokens


class JapaneseTfIdfVectorizer:
    """既存 `Tf_idf` と同じ API を持つ日本語向けベクトル化クラス。"""

    def __init__(self, documents, deacc, min_len, use_stemming=False):
        self.documents = documents
        self.deacc = deacc
        self.min_len = min_len
        self.use_stemming = use_stemming
        self.processed_docs = self.preprocess()

    def labels(self, document_sagi):
        return [1] * len(document_sagi) + [0] * (len(self.documents) - len(document_sagi))

    def _preprocess_doc(self, doc):
        return " ".join(japanese_tokenize(doc))

    def preprocess(self):
        return [self._preprocess_doc(doc) for doc in self.documents]

    def _Vectorizer(self, min_df, ngram_range=(1, 1), use_idf=True, norm="l2"):
        return TfidfVectorizer(
            stop_words=None,
            min_df=min_df,
            ngram_range=ngram_range,
            token_pattern=r"(?u)\b\w+\b",
            use_idf=use_idf,
            norm=norm,
        )

    def tf_idf(self, min_df, ngram_range=(1, 1)):
        vectorizer = self._Vectorizer(min_df, ngram_range=ngram_range)
        X = vectorizer.fit_transform(self.processed_docs)
        feature_names = vectorizer.get_feature_names_out()
        return X, feature_names, vectorizer

    def term_frequency(self, min_df=1, ngram_range=(1, 5)):
        vectorizer = self._Vectorizer(
            min_df,
            ngram_range=ngram_range,
            use_idf=False,
            norm=None,
        )
        X = vectorizer.fit_transform(self.processed_docs)
        feature_names = vectorizer.get_feature_names_out()
        return X, feature_names, vectorizer


def load_corpus():
    """another_data と同じ戻り値形式で EDINET-Bench の本文・ラベル・IDを返す。"""
    dataset = load_edinet_dataset()
    dataset = dataset.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).reset_index(drop=True)

    documents = dataset[TEXT_COLUMN].map(extract_text_only).astype(str).to_numpy(dtype=object)
    labels = pd.to_numeric(dataset[LABEL_COLUMN], errors="raise").astype(int).to_numpy()
    unknown_labels = sorted(set(labels) - {0, 1})
    if unknown_labels:
        raise ValueError(f"{LABEL_COLUMN} column contains unsupported labels: {unknown_labels}. Expected 0 and 1.")

    doc_ids = np.arange(len(documents), dtype=int)
    pdf_names = build_document_names(dataset, doc_ids)

    if len(pdf_names) != len(documents):
        raise RuntimeError(
            "document id count does not match document count: "
            f"pdf_names={len(pdf_names)}, documents={len(documents)}"
        )

    return documents, labels, doc_ids, pdf_names


def build_document_names(dataset, doc_ids):
    """誤分類出力用に、既存データセット内の識別子から文書名を作る。"""
    names = []
    for row_index, row in dataset.iterrows():
        parts = [str(row.get("source_split", "unknown"))]
        for column in ID_COLUMNS:
            if column in dataset.columns and pd.notna(row[column]) and str(row[column]).strip():
                parts.append(f"{column}={row[column]}")
        if len(parts) == 1:
            parts.append(f"row={doc_ids[row_index]}")
        names.append(" | ".join(parts))
    return np.array(names, dtype=object)


def configure_engine():
    """another_data の実験本体を EDINET-Bench 用に差し替える。"""
    engine = importlib.import_module(ENGINE_MODULE_NAME)
    engine.SAVE_DIR = SAVE_DIR
    engine.load_corpus = load_corpus
    engine.Tf_idf = JapaneseTfIdfVectorizer
    engine.USE_STEMMING = False
    return engine


def main():
    engine = configure_engine()
    runtime_config = engine.prompt_experiment_configuration()
    engine.print_experiment_configuration(runtime_config)

    documents, labels, doc_ids, pdf_names = load_corpus()
    active_conditions = engine.get_active_conditions(
        runtime_config["kernel_mode"],
        runtime_config["feature_mode"],
    )
    global_margin_source_results = []
    engine.log_progress("[main] EDINET-Bench corpus loaded; building split contexts")
    engine.log_progress(
        f"[main] documents={len(documents)}, label1={int(np.sum(labels == 1))}, label0={int(np.sum(labels == 0))}"
    )
    outer_fold_contexts = engine.build_outer_fold_contexts(documents, labels, doc_ids)

    fixed_root = SAVE_DIR / engine.FIXED_DIR_NAME
    tuned_root = SAVE_DIR / engine.TUNED_DIR_NAME
    threshold_count_paths = engine.save_threshold_term_count_outputs(
        outer_fold_contexts,
        engine.ABS_SCORE_THRESHOLDS,
        SAVE_DIR / engine.THRESHOLD_COUNT_DIR_NAME,
    )

    engine.log_progress(f"saved threshold term-count csv: {threshold_count_paths['outer_fold_threshold_term_counts']}")
    engine.log_progress(f"saved threshold term-count summary csv: {threshold_count_paths['threshold_term_count_summary']}")

    if runtime_config["threshold_mode"] in ("fixed", "both"):
        fixed_summary_rows_by_condition = {condition: [] for condition in active_conditions}
        fixed_best_results_by_condition = {condition: None for condition in active_conditions}

        engine.log_progress("\n[main] running fixed-threshold nested CV")
        engine.log_progress(f"[main] threshold candidates: {engine.ABS_SCORE_THRESHOLDS}")
        for score_threshold in engine.ABS_SCORE_THRESHOLDS:
            engine.log_progress(f"\n[main] fixed threshold = {score_threshold}")
            threshold_root = fixed_root / engine.format_threshold_slug(score_threshold)

            for feature_mode, kernel in active_conditions:
                result = engine.run_nested_cv_for_condition(
                    outer_fold_contexts,
                    feature_mode,
                    kernel,
                    [score_threshold],
                )

                condition_dir = engine.get_condition_metrics_dir(
                    threshold_root / engine.METRICS_DIR_NAME,
                    feature_mode,
                    kernel,
                )
                engine.save_nested_cv_result_csvs(result, condition_dir, "svm")
                engine.run_optional_output_step(
                    result,
                    "selected term export",
                    engine.save_outer_fold_selected_terms,
                    result,
                    outer_fold_contexts,
                    condition_dir / engine.FEATURE_OUTPUT_DIR_NAME,
                )
                engine.run_optional_output_step(
                    result,
                    "misclassified document export",
                    engine.save_outer_fold_misclassified_documents,
                    result,
                    condition_dir / engine.MISCLASSIFIED_OUTPUT_DIR_NAME,
                )
                engine.run_optional_output_step(
                    result,
                    "misclassification feature analysis",
                    engine.save_outer_fold_misclassification_feature_analysis,
                    result,
                    outer_fold_contexts,
                    feature_mode,
                    kernel,
                    condition_dir / engine.ERROR_ANALYSIS_OUTPUT_DIR_NAME,
                )
                engine.print_result(result, f"{engine.format_condition_label(feature_mode, kernel)} / 固定閾値={score_threshold}")
                global_margin_source_results.append(result)

                summary_row = engine.build_fixed_threshold_summary_row(score_threshold, result)
                fixed_summary_rows_by_condition[(feature_mode, kernel)].append(summary_row)

                if result.get("selected_records"):
                    current_best = fixed_best_results_by_condition[(feature_mode, kernel)]
                    if current_best is None or result["mean_accuracy"] > current_best["raw_accuracy"]:
                        fixed_best_results_by_condition[(feature_mode, kernel)] = {
                            **summary_row,
                            "raw_accuracy": result["mean_accuracy"],
                        }

        engine.save_fixed_threshold_global_outputs(
            fixed_summary_rows_by_condition,
            fixed_best_results_by_condition,
            fixed_root,
        )

    if runtime_config["threshold_mode"] in ("tuned", "both"):
        tuned_summary_rows = []
        engine.log_progress("\n[main] running tuned-threshold nested CV")
        engine.log_progress(f"[main] threshold candidates: {engine.ABS_SCORE_THRESHOLDS}")
        for feature_mode, kernel in active_conditions:
            result = engine.run_nested_cv_for_condition(
                outer_fold_contexts,
                feature_mode,
                kernel,
                engine.ABS_SCORE_THRESHOLDS,
            )

            condition_dir = engine.get_condition_metrics_dir(tuned_root / engine.METRICS_DIR_NAME, feature_mode, kernel)
            engine.save_nested_cv_result_csvs(result, condition_dir, "svm")
            engine.run_optional_output_step(
                result,
                "selected term export",
                engine.save_outer_fold_selected_terms,
                result,
                outer_fold_contexts,
                condition_dir / engine.FEATURE_OUTPUT_DIR_NAME,
            )
            engine.run_optional_output_step(
                result,
                "misclassified document export",
                engine.save_outer_fold_misclassified_documents,
                result,
                condition_dir / engine.MISCLASSIFIED_OUTPUT_DIR_NAME,
            )
            engine.run_optional_output_step(
                result,
                "misclassification feature analysis",
                engine.save_outer_fold_misclassification_feature_analysis,
                result,
                outer_fold_contexts,
                feature_mode,
                kernel,
                condition_dir / engine.ERROR_ANALYSIS_OUTPUT_DIR_NAME,
            )
            engine.print_result(result, f"{engine.format_condition_label(feature_mode, kernel)} / 閾値チューニング")
            tuned_summary_rows.append(engine.build_tuned_threshold_summary_row(result))

        engine.save_tuned_threshold_global_outputs(tuned_summary_rows, tuned_root)

    if not global_margin_source_results:
        engine.log_progress("\n[main] running fixed-threshold evaluations for global top-margin misclassification export")
        for score_threshold in engine.ABS_SCORE_THRESHOLDS:
            engine.log_progress(f"[main] export-only fixed threshold = {score_threshold}")
            for feature_mode, kernel in active_conditions:
                global_margin_source_results.append(
                    engine.run_nested_cv_for_condition(
                        outer_fold_contexts,
                        feature_mode,
                        kernel,
                        [score_threshold],
                    )
                )

    try:
        global_margin_csv_path = engine.save_global_top_margin_misclassified_documents(
            global_margin_source_results,
            outer_fold_contexts,
            pdf_names,
            SAVE_DIR / engine.GLOBAL_TOP_MARGIN_MISCLASSIFIED_FILENAME,
            engine.GLOBAL_TOP_MARGIN_MISCLASSIFIED_N,
        )
        engine.log_progress(f"saved global top-margin misclassified csv: {global_margin_csv_path}")
    except Exception as exc:
        engine.log_progress(
            "[output warning] global top-margin misclassified export failed: "
            f"{type(exc).__name__}: {exc}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
