from pathlib import Path
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
redirect_relative_outputs(get_output_dir(__file__, PROJECT_ROOT))

from word_bias_explorer import word_chose
from pdf_text_loader import Text_road_and_dell
from text_vectorizer import Tf_idf


text_sagi = Text_road_and_dell("document_詐欺_先行研究.pkl", "詐欺_先行研究")
document_sagi = text_sagi.read_PDF()

# 詐欺ではないテキストの読み込み
text_no_sagi = Text_road_and_dell("document_詐欺じゃない_先行研究.pkl", "詐欺じゃない_先行研究")
document_no_sagi = text_no_sagi.read_PDF()

documents= document_sagi+document_no_sagi

count= Tf_idf(documents, True, 3)
labels = count.labels(document_sagi)

word = word_chose(documents, labels)

word._print_top_word(False)

tokens_target = ["liveness"]

word._token_plot(tokens_target)
