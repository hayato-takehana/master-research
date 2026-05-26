from __future__ import annotations

from pathlib import Path
import importlib
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

from project_runtime import bootstrap_project_paths, get_output_dir, redirect_relative_outputs  # noqa: E402

bootstrap_project_paths(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
SAVE_DIR = get_output_dir(__file__, PROJECT_ROOT)
redirect_relative_outputs(SAVE_DIR, override=True)

import pandas as pd  # noqa: E402

from active.another_data_japanese.doc_count_score_threshold_margin_nested_cv_experiment import (  # noqa: E402
    JapaneseTfIdfVectorizer,
    extract_text_only,
)

redirect_relative_outputs(SAVE_DIR, override=True)


DATASET_NAME = "SakanaAI/EDINET-Bench"
CONFIG_NAME = "fraud_detection"
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
ENGINE_MODULE_NAME = "active.another_data.pre_research_reproduction"


def load_corpus() -> tuple[list[str], list[int]]:
    """EDINET-Bench の `text` だけを文書特徴として読み込む。"""
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
        missing_columns = [
            column for column in (TEXT_COLUMN, LABEL_COLUMN) if column not in dataset_split.column_names
        ]
        if missing_columns:
            raise ValueError(f"{split_name} split is missing required columns: {missing_columns}")
        frames.append(dataset_split.select_columns([TEXT_COLUMN, LABEL_COLUMN]).to_pandas())

    dataset = pd.concat(frames, ignore_index=True).dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).reset_index(drop=True)
    labels = pd.to_numeric(dataset[LABEL_COLUMN], errors="raise").astype(int)
    unknown_labels = sorted(set(labels) - {0, 1})
    if unknown_labels:
        raise ValueError(f"{LABEL_COLUMN} column contains unsupported labels: {unknown_labels}. Expected 0 and 1.")

    print(int((labels == 1).sum()))
    print(int((labels == 0).sum()))
    return dataset[TEXT_COLUMN].map(extract_text_only).astype(str).tolist(), labels.astype(int).tolist()


def configure_engine():
    engine = importlib.import_module(ENGINE_MODULE_NAME)
    engine.SAVE_DIR = SAVE_DIR
    engine.RESULT_CSV = SAVE_DIR / "pre_research_svm_reproduction_results.csv"
    engine.load_corpus = load_corpus
    engine.Tf_idf = JapaneseTfIdfVectorizer
    engine.USE_STEMMING = False
    engine.MIN_LEN = 1
    return engine


def main() -> int:
    return configure_engine().main()


if __name__ == "__main__":
    raise SystemExit(main())
