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
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\b\w{1}\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# --- ラベル1 (詐欺_先行研究) のPDF読み込み ---
# (キャッシュファイルが存在する場合、このブロックは高速に実行されます)
cache_file_1 = "document_詐欺_先行研究.pkl"
folder_path_1 = "詐欺_先行研究"
if os.path.exists(cache_file_1):
    with open(cache_file_1, "rb") as f:
        documents_1 = pickle.load(f)
    print("キャッシュファイルからdocuments_1を読み込みました。")
else:
    documents_1 = []
    a = 0
    for filename in os.listdir(folder_path_1):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path_1, filename)
            try:
                pdf_document = PdfReader(file_path)
                extracted_text = ""
                for i in range(len(pdf_document.pages)):
                    page = pdf_document.pages[i]
                    text = page.extract_text()
                    if text:  # テキストが None でないことを確認
                        extracted_text += text
                extracted_text = text_dell(extracted_text)
                documents_1.append(extracted_text)
                print(f"File 1-{a}: {filename} 読み込み完了")
            except Exception as e:
                print(f"File 1-{a}: {filename} の読み込み中にエラーが発生しました: {e}")
        a += 1
    with open(cache_file_1, "wb") as f:
        pickle.dump(documents_1, f)
    print("新たにdocuments_1を生成し、キャッシュファイルに保存しました。")

# --- ラベル0 (詐欺じゃない_先行研究) のPDF読み込み ---
# (キャッシュファイルが存在する場合、このブロックは高速に実行されます)
cache_file_0 = "document_詐欺じゃない_先行研究.pkl"
folder_path_0 = "詐欺じゃない_先行研究"
if os.path.exists(cache_file_0):
    with open(cache_file_0, "rb") as f:
        documents_0 = pickle.load(f)
    print("キャッシュファイルからdocuments_0を読み込みました。")
else:
    documents_0 = []
    a = 0
    for filename in os.listdir(folder_path_0):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path_0, filename)
            try:
                pdf_document = PdfReader(file_path)
                extracted_text = ""
                for i in range(len(pdf_document.pages)):
                    page = pdf_document.pages[i]
                    text = page.extract_text()
                    if text:  # テキストが None でないことを確認
                        extracted_text += text
                extracted_text = text_dell(extracted_text)
                documents_0.append(extracted_text)
                print(f"File 0-{a}: {filename} 読み込み完了")
            except Exception as e:
                print(f"File 0-{a}: {filename} の読み込み中にエラーが発生しました: {e}")
        a += 1
    with open(catche_0, "wb") as f:
        pickle.dump(documents_0, f)
    print("新たにdocuments_0を生成し、キャッシュファイルに保存しました。")


# --- 前処理 (ここまでは変更なし) ---
def preprocess_with_stopwords(doc):
    tokens = gensim.utils.simple_preprocess(doc, deacc=True, min_len=3)
    stopped_tokens = [token for token in tokens if token not in STOPWORDS]
    stemmed_tokens = [stem_text(token) for token in stopped_tokens]
    return ' '.join(stemmed_tokens)


print("\n--- ラベル1の前処理を開始 ---")
processed_docs_1 = [preprocess_with_stopwords(doc) for doc in documents_1]
print("--- ラベル1の前処理が完了 ---")

print("\n--- ラベル0の前処理を開始 ---")
processed_docs_0 = [preprocess_with_stopwords(doc) for doc in documents_0]
print("--- ラベル0の前処理が完了 ---")

# ===================================================================
# これまでの集計関数 (変更なし、そのまま残します)
# ===================================================================
# ... (get_top_n_words_by_freq, get_top_n_words_by_df,
#      calculate_and_print_diff_label1_minus_label0,
#      calculate_and_print_diff_label0_minus_label1)
# ... (上記4つの関数の定義は省略) ...


# ===================================================================
# これまでの実行ブロック (変更なし、そのまま残します)
# ===================================================================
# ... (トップN単語の表示) ...
# ... (差分の表示) ...
# ... (固有単語の全リスト表示) ...


# ===================================================================
# 【新規】ラベル固有単語の「トップ100」を抽出
# ===================================================================

print("\n" + "#" * 70)
print(" ご要望の集計: ラベル固有単語の トップ100 (回数/文書数)")
print("#" * 70)


# --- ステップ1: 固有単語のセットを作成 ---

def get_unique_vocab_sets(docs_1, docs_0):
    # ラベル1の語彙セット
    vectorizer_1 = CountVectorizer()
    vectorizer_1.fit(docs_1)
    vocab_1 = set(vectorizer_1.vocabulary_.keys())
    print(f"\n--- ラベル1の総語彙数: {len(vocab_1)} ---")

    # ラベル0の語彙セット
    vectorizer_0 = CountVectorizer()
    vectorizer_0.fit(docs_0)
    vocab_0 = set(vectorizer_0.vocabulary_.keys())
    print(f"--- ラベル0の総語彙数: {len(vocab_0)} ---")

    # 差集合を計算
    unique_to_1 = vocab_1.difference(vocab_0)
    unique_to_0 = vocab_0.difference(vocab_1)

    print(f"--- ラベル1固有の単語数: {len(unique_to_1)} ---")
    print(f"--- ラベル0固有の単語数: {len(unique_to_0)} ---")

    return unique_to_1, unique_to_0


# 固有単語セットを取得
unique_to_1, unique_to_0 = get_unique_vocab_sets(processed_docs_1, processed_docs_0)


# --- ステップ2: 固有単語セットを使ってトップNを計算する関数 ---

def get_top_n_unique_words(corpus, unique_vocab_set, metric_type, n=100):
    """
    コーパスと「固有単語セット」を受け取り、
    そのセット内の単語だけを対象に、指定された指標でトップNを返す。
    """

    if metric_type == 'frequency':
        # 総出現回数
        vectorizer = CountVectorizer(binary=False)
        metric_name = "総出現回数"
        unit = "回"
    elif metric_type == 'df':
        # 出現文書数
        vectorizer = CountVectorizer(binary=True)
        metric_name = "出現文書数"
        unit = "文書"
    else:
        raise ValueError("metric_type must be 'frequency' or 'df'")

    # 対象コーパスで語彙とカウントを学習
    X = vectorizer.fit_transform(corpus)
    counts = X.sum(axis=0)
    vocab = vectorizer.vocabulary_

    # 固有単語セットに含まれる単語だけをフィルタリング
    filtered_counts = []
    for word, idx in vocab.items():
        if word in unique_vocab_set:
            count = counts[0, idx]
            filtered_counts.append((word, count))

    # フィルタリングされたリストをカウントでソート
    sorted_list = sorted(filtered_counts, key=lambda x: x[1], reverse=True)

    # トップNを返す
    return sorted_list[:n], unit


# --- ステップ3: 実行と結果の出力 ---

# --- ラベル1 ---
print("\n" + "=" * 50)
print(" ラベル1 (詐欺_先行研究) 固有単語のトップ100")
print("=" * 50)

# 1a. ラベル1固有単語 (総出現回数順)
print("\n--- (A) 総出現回数 (Frequency) 順 ---")
top_freq_1, unit = get_top_n_unique_words(processed_docs_1, unique_to_1, 'frequency', 100)
for i, (word, freq) in enumerate(top_freq_1):
    print(f"{i + 1:3d}. {word:<20} : {freq} {unit}")

# 1b. ラベル1固有単語 (出現文書数順)
print("\n--- (B) 出現文書数 (Document Frequency) 順 ---")
top_df_1, unit = get_top_n_unique_words(processed_docs_1, unique_to_1, 'df', 100)
for i, (word, freq) in enumerate(top_df_1):
    print(f"{i + 1:3d}. {word:<20} : {freq} {unit} / {len(processed_docs_1)} 文書中")

# --- ラベル0 ---
print("\n" + "=" * 50)
print(" ラベル0 (詐欺じゃない_先行研究) 固有単語のトップ100")
print("=" * 50)

# 0a. ラベル0固有単語 (総出現回数順)
print("\n--- (A) 総出現回数 (Frequency) 順 ---")
top_freq_0, unit = get_top_n_unique_words(processed_docs_0, unique_to_0, 'frequency', 100)
for i, (word, freq) in enumerate(top_freq_0):
    print(f"{i + 1:3d}. {word:<20} : {freq} {unit}")

# 0b. ラベル0固有単語 (出現文書数順)
print("\n--- (B) 出現文書数 (Document Frequency) 順 ---")
top_df_0, unit = get_top_n_unique_words(processed_docs_0, unique_to_0, 'df', 100)
for i, (word, freq) in enumerate(top_df_0):
    print(f"{i + 1:3d}. {word:<20} : {freq} {unit} / {len(processed_docs_0)} 文書中")
