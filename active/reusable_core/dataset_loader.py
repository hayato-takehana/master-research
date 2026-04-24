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


def _dataset_paths(real_scam=False):
    if real_scam:
        return (
            PROJECT_ROOT / "詐欺_実際の詐欺",
            PROJECT_ROOT / "詐欺じゃない_実際の詐欺",
        )
    return (
        PROJECT_ROOT / "詐欺_先行研究",
        PROJECT_ROOT / "詐欺じゃない_先行研究",
    )


def load_document_filenames(real_scam=False, sort_files=False):
    scam_folder, non_scam_folder = _dataset_paths(real_scam=real_scam)

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


def load_documents(real_scam=False, keep_digits=False, percent_mode="drop"):
    if real_scam:
        scam_cache = _cache_path("document_詐欺_実際の詐欺.pkl", keep_digits, percent_mode)
        non_scam_cache = _cache_path("document_詐欺じゃない_実際の詐欺.pkl", keep_digits, percent_mode)
    else:
        scam_cache = _cache_path("document_詐欺_先行研究.pkl", keep_digits, percent_mode)
        non_scam_cache = _cache_path("document_詐欺じゃない_先行研究.pkl", keep_digits, percent_mode)

    scam_folder, non_scam_folder = _dataset_paths(real_scam=real_scam)

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
def common(real_scam, tf_idf_true, keep_digits=False, percent_mode="drop", use_stemming=False):
    document_sagi, document_no_sagi = load_documents(
        real_scam=real_scam,
        keep_digits=keep_digits,
        percent_mode=percent_mode,
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


