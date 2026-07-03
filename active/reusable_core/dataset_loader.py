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

from project_runtime import bootstrap_project_paths

bootstrap_project_paths(PROJECT_ROOT)

from pdf_text_loader import Text_road_and_dell
from text_vectorizer import Tf_idf
import pandas as pd
from project_runtime import find_project_root


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(MODULE_DIR)

PRE_RESEARCH_DATASET_NAMES = (
    "詐欺_先行研究",
    "詐欺じゃない_先行研究",
)
PRE_RESEARCH_DELETE_V1_DATASET_NAMES = (
    "詐欺_先行研究_delete_v1",
    "詐欺じゃない_先行研究_delete_v1",
)
PRE_RESEARCH_NEW_LABEL_DATASET_NAMES = (
    "詐欺_先行研究_新ラベル",
    "詐欺じゃない_先行研究_新ラベル",
)
REAL_SCAM_DATASET_NAMES = (
    "詐欺_実際の詐欺",
    "詐欺じゃない_実際の詐欺",
)


def _cache_path(filename, keep_digits=False, percent_mode="drop"):
    suffix_parts = []
    if keep_digits:
        suffix_parts.append("digits")
    if percent_mode == "word":
        suffix_parts.append("percent_word")

    if not suffix_parts:
        return PROJECT_ROOT / filename

    path = Path(filename)
    suffix = "_" + "_".join(suffix_parts)
    return PROJECT_ROOT / f"{path.stem}{suffix}{path.suffix}"


def _resolve_dataset_names(real_scam=False, dataset_names=None):
    if dataset_names is None:
        return (
            REAL_SCAM_DATASET_NAMES
            if real_scam
            else PRE_RESEARCH_DATASET_NAMES
        )

    if isinstance(dataset_names, (str, bytes)):
        raise ValueError(
            "dataset_names must contain scam and non-scam directory names."
        )
    try:
        resolved_names = tuple(dataset_names)
    except TypeError as exc:
        raise ValueError(
            "dataset_names must contain scam and non-scam directory names."
        ) from exc
    if len(resolved_names) != 2:
        raise ValueError(
            "dataset_names must contain exactly two directory names."
        )

    normalized_names = tuple(str(name).strip() for name in resolved_names)
    if not all(normalized_names):
        raise ValueError("Dataset directory names must not be empty.")
    return normalized_names


def _dataset_paths(real_scam=False, dataset_names=None):
    scam_dataset_name, non_scam_dataset_name = _resolve_dataset_names(
        real_scam=real_scam,
        dataset_names=dataset_names,
    )
    return (
        PROJECT_ROOT / scam_dataset_name,
        PROJECT_ROOT / non_scam_dataset_name,
    )


def load_document_filenames(
    real_scam=False,
    sort_files=False,
    dataset_names=None,
):
    scam_folder, non_scam_folder = _dataset_paths(
        real_scam=real_scam,
        dataset_names=dataset_names,
    )

    def _display_document_name(filename):
        path = Path(filename)
        if path.suffix.lower() != ".txt":
            return filename
        stem = path.stem
        for suffix in ("_text_cleaned", "_ocr_text_cleaned", "_cleaned"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        return f"{stem}.pdf"

    def _list_document_names(folder):
        files = [
            f
            for f in os.listdir(folder)
            if f.lower().endswith(".pdf") or f.lower().endswith(".txt")
        ]
        if sort_files:
            files.sort()
        return [_display_document_name(f) for f in files]

    return _list_document_names(scam_folder), _list_document_names(non_scam_folder)


def load_documents(
    real_scam=False,
    keep_digits=False,
    percent_mode="drop",
    dataset_names=None,
):
    scam_dataset_name, non_scam_dataset_name = _resolve_dataset_names(
        real_scam=real_scam,
        dataset_names=dataset_names,
    )
    scam_cache = _cache_path(
        f"document_{scam_dataset_name}.pkl",
        keep_digits,
        percent_mode,
    )
    non_scam_cache = _cache_path(
        f"document_{non_scam_dataset_name}.pkl",
        keep_digits,
        percent_mode,
    )

    scam_folder, non_scam_folder = _dataset_paths(
        real_scam=real_scam,
        dataset_names=(scam_dataset_name, non_scam_dataset_name),
    )

    text_sagi = Text_road_and_dell(
        str(scam_cache),
        str(scam_folder),
        keep_digits=keep_digits,
        percent_mode=percent_mode,
    )
    document_sagi = text_sagi.read_PDF()

    text_no_sagi = Text_road_and_dell(
        str(non_scam_cache),
        str(non_scam_folder),
        keep_digits=keep_digits,
        percent_mode=percent_mode,
    )
    document_no_sagi = text_no_sagi.read_PDF()

    return document_sagi, document_no_sagi


# ファイルの読み込みと、TF-IDF real_scam:trueで実際の詐欺、tf_idf_true:TrueでTF-IDF,Falseで頻度
def common(
    real_scam,
    tf_idf_true,
    keep_digits=False,
    percent_mode="drop",
    use_stemming=False,
    dataset_names=None,
):
    document_sagi, document_no_sagi = load_documents(
        real_scam=real_scam,
        keep_digits=keep_digits,
        percent_mode=percent_mode,
        dataset_names=dataset_names,
    )

    # 詐欺のと詐欺でないドキュメントの結合
    documents = document_sagi + document_no_sagi

    # tf_idfを行う
    tf_idf = Tf_idf(documents, 'True', 1, use_stemming=use_stemming)
    labels = tf_idf.labels(document_sagi)
    tf_idf.preprocess()

    X, feature_names, vectorizer = tf_idf.tf_idf(0.0)
    X_freq, feature_names_freq, vectorizer = tf_idf.term_frequency(0.0)


    df = pd.DataFrame(X.toarray(), columns=feature_names)
    # print(X)
    # print(feature_names)
    # print(len(feature_names))
    if tf_idf_true:
        return X, labels,feature_names, vectorizer
    else:
        return X_freq, labels, feature_names_freq, vectorizer


