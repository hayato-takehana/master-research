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

import pandas as pd
import time
from PyPDF2 import PdfReader
import re
import os
import pickle
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.parsing.preprocessing import stem_text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.tree import export_graphviz
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.svm import OneClassSVM
from sklearn import svm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from collections import defaultdict, Counter
from sklearn.metrics import classification_report


def text_dell(text):
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^a-zA-X\s]', '', text)
    text = re.sub(r'\b\w{1}\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# キャッシュファイルのパス
cache_file_1 = "document_詐欺_先行研究.pkl"

# PDFファイルが保存されているフォルダのパス
folder_path_1 = "詐欺_先行研究"

# キャッシュファイルが存在するか確認
if os.path.exists(cache_file_1):
    with open(cache_file_1, "rb") as f:
        documents_1 = pickle.load(f)
    print("キャッシュファイルからdocumentsを読み込みました。")
else:
    documents_1 = []
    a=0
    # フォルダ内のすべてのPDFファイルを取得
    for filename in os.listdir(folder_path_1):

        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path_1, filename)
            pdf_document = PdfReader(file_path)
            extracted_text = ""

            for i in range(len(pdf_document.pages)):
                page = pdf_document.pages[i]
                text = page.extract_text()
                extracted_text += text

            extracted_text = text_dell(extracted_text)
            documents_1.append(extracted_text)
            print(a)
        a+=1

    # 抽出したデータをキャッシュファイルに保存
    with open(cache_file_1, "wb") as f:
        pickle.dump(documents_1, f)
    print("新たにdocumentsを生成し、キャッシュファイルに保存しました。")

# 以後、documentsに対して処理を行う


# キャッシュファイルのパス
cache_file_0 = "document_詐欺じゃない_先行研究.pkl"

# PDFファイルが保存されているフォルダのパス
folder_path_0 = "詐欺じゃない_先行研究"

# キャッシュファイルが存在するか確認
if os.path.exists(cache_file_0):
    with open(cache_file_0, "rb") as f:
        documents_0 = pickle.load(f)
    print("キャッシュファイルからdocuments_0を読み込みました。")
else:
    documents_0 = []
    a=0
    # フォルダ内のすべてのPDFファイルを取得
    for filename in os.listdir(folder_path_0):

        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path_0, filename)
            pdf_document = PdfReader(file_path)
            extracted_text = ""

            for i in range(len(pdf_document.pages)):
                page = pdf_document.pages[i]
                text = page.extract_text()
                extracted_text += text

            extracted_text = text_dell(extracted_text)
            documents_0.append(extracted_text)
            print(a)
        a+=1

    # 抽出したデータをキャッシュファイルに保存
    with open(cache_file_0, "wb") as f:
        pickle.dump(documents_0, f)
    print("新たにdocuments_0を生成し、キャッシュファイルに保存しました。")


documents = documents_1 + documents_0
labels = [1] * len(documents_1) + [0] * (len(documents) - len(documents_1))

## ----------------------------------------------------------------
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 修正版コード ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# ----------------------------------------------------------------
from collections import Counter
import pandas as pd
from gensim.parsing.preprocessing import STOPWORDS # STOPWORDSをインポート

def create_global_word_frequency_ranking_without_stopwords(doc_list_1, label_1, doc_list_2, label_2):
    """
    2つの文書グループからストップワードを除外し、
    全ての「単語、文書名、出現回数」の組み合わせリストを作成し、
    出現回数の多い順にソートしてランキング表示する関数。

    Args:
        doc_list_1 (list): 1つ目の文書リスト (例: documents_1)。
        label_1 (str): 1つ目の文書リストのラベル (例: "詐欺_先行研究")。
        doc_list_2 (list): 2つ目の文書リスト (例: documents_0)。
        label_2 (str): 2つ目の文書リストのラベル (例: "詐欺じゃない_先行研究")。
    """
    # 全ての「単語、文書名、出現回数」の組み合わせを格納するリスト
    all_word_occurrences = []

    # 処理をまとめるために、文書リストとラベルをタプルのリストにする
    document_groups = [(doc_list_1, label_1), (doc_list_2, label_2)]

    # 各文書グループ（「詐欺」と「詐欺じゃない」）でループ
    for doc_list, label in document_groups:
        # 各文書でループ（enumerateでインデックスも取得）
        for i, doc in enumerate(doc_list):
            # 文書名を生成 (例: "詐欺_先行研究_1")
            doc_name = f"{label}_{i+1}"

            # ▼▼▼▼▼▼▼▼▼ 変更点 ▼▼▼▼▼▼▼▼▼
            # 1. 文書を単語に分割
            words = doc.split()

            # 2. ストップワードリストに含まれない単語のみを抽出
            #    (単語を小文字に変換してストップワードと比較)
            filtered_words = [word for word in words if word.lower() not in STOPWORDS]

            # 3. ストップワード除外後の単語リストで出現回数をカウント
            word_counts = Counter(filtered_words)
            # ▲▲▲▲▲▲▲▲▲ 変更点 ▲▲▲▲▲▲▲▲▲

            # カウントした単語と回数をループ
            for word, count in word_counts.items():
                # リストに辞書形式で追加
                all_word_occurrences.append({
                    "Word": word,
                    "Document": doc_name,
                    "Frequency": count
                })

    # pandas DataFrameに変換
    df = pd.DataFrame(all_word_occurrences)

    # "Frequency"（出現回数）の列を基準に、降順（多い順）でソート
    df_sorted = df.sort_values(by="Frequency", ascending=False)

    # ランキングが見やすいようにインデックスをリセット
    df_sorted.reset_index(drop=True, inplace=True)
    df_sorted.index += 1 # インデックスを1から始める
    df_sorted.index.name = "Rank"

    print("\n--- 全文書・全単語横断：出現回数ランキング (上位100, ストップワード除外) ---")

    # DataFrameの先頭から100行を表示する
    print(df_sorted.head(100))


# 関数を実行してランキングを表示
create_global_word_frequency_ranking_without_stopwords(
    documents_1, "詐欺_先行研究",
    documents_0, "詐欺じゃない_先行研究"
)
