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
from sklearn.feature_selection import SelectKBest, mutual_info_classif


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



#TF-IDFを行う
# 前処理関数
def preprocess(doc):
    # 3文字未満の単語を削除し、ステミングを適用fd
    tokens = gensim.utils.simple_preprocess(doc, deacc=True, min_len=3)
    stemmed_tokens = [stem_text(token) for token in tokens]
    return ' '.join(stemmed_tokens)

# 文書に前処理を適用
processed_docs = [preprocess(doc) for doc in documents]


# TF-IDFの計算
vectorizer = TfidfVectorizer(
    stop_words='english',  # scikit-learnのストップワードを使用
    min_df=0.02  # 2%未満の出現率の単語を無視
)
X = vectorizer.fit_transform(processed_docs)





X = np.delete(X.toarray(), 152, axis=0)
labels.pop(152)
labels = np.array(labels)

# 選択する単語（特徴）の数 N を設定
# この値を変更することで、動的に特徴数を変えて実験できます
N_top_features = 143

print(f"\n情報利得（IG）に基づいて上位{N_top_features}個の単語を選択します...")

# SelectKBestを使用して、相互情報量（情報利得）が最大のK個の特徴を選択
# mutual_info_classifはラベル付きデータに対する相互情報量を計算し、情報利得の代替として機能します
selector = SelectKBest(mutual_info_classif, k=N_top_features)
X_new = selector.fit_transform(X, labels)

# 選択された特徴（単語）の名前を取得して表示（確認用）
feature_names = np.array(vectorizer.get_feature_names_out())
selected_feature_indices = selector.get_support(indices=True)
selected_features = feature_names[selected_feature_indices]

print(f"\n選択された上位{N_top_features}個の単語:")
print(selected_features)

# 以降の分析では、選択された特徴量（X_new）を使用する
X = X_new
print(f"\n特徴選択後のデータ shape: {X.shape}")
# ▲▲▲【ここまでが追加・変更部分】▲▲▲
print(X)










# 内側の層化k分割交差検証とグリッドサーチ（k=10）
inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# 外側の層化k分割交差検証（k=10）
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

C=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]

best_accurancy = []
best_recall=[]
best_precision=[]
best_f1_score=[]
best_cs = []
gaku_so=True

# 新しい閾値を設定
custom_threshold = [0.00, -2.00, -3.00, -5.00, -10.00]

for custom in custom_threshold:
    best_accurancy = []
    best_recall = []
    best_precision = []
    best_f1_score = []
    best_cs = []
    gaku_so = True

    for train_index, test_index in outer_cv.split(X, labels):
        #print("外側の分割 - 学習データ:", train_index)
        best_score = 0
        best_c = 0

        for c in C:
            score = []
            gaku = True

            for inner_train_index, inner_test_index in inner_cv.split(X[train_index], labels[train_index]):

                inner_train_indices_in_original = train_index[inner_train_index]
                inner_test_indices_in_original = train_index[inner_test_index]
                X_new_in = X[inner_train_indices_in_original]  # スパース行列 X から部分行列を取得
                y_new_in = labels[inner_train_indices_in_original]
                X_new_test_in = X[inner_test_indices_in_original]
                y_new_test_in = labels[inner_test_indices_in_original]

                svm_clf = svm.SVC(kernel='linear', C=c, class_weight='balanced',tol=1e-10)
                # モデルを訓練データで学習
                svm_clf.fit(X_new_in, y_new_in)                # 3. decision_function() を使って (w・x + b) を計算
                decision_values = svm_clf.decision_function(X_new_in)
                y_new_distance= np.where(y_new_in == 0, -1, 1)
                margin = y_new_distance*decision_values
                #print(margin)

                if np.all(margin >= 0.9990):
                    sco = svm_clf.score(X_new_test_in, y_new_test_in)
                    score.append(sco)
                else:
                    gaku = False
                    break

            if gaku:
                average_score = sum(score)/len(score)
                if best_score < average_score:
                    best_score = average_score
                    best_c = c

        if best_score == 0:
            gaku_so = False
            print("内側の検証で学習率100%のものはありませんでした")
            break


        X_new_out = X[train_index]  # スパース行列 X から部分行列を取得
        y_new_out = labels[train_index]
        X_new_test_out = X[test_index]
        y_new_test_out = labels[test_index]

        svm_clf_out = svm.SVC(kernel='linear', C=best_c, class_weight='balanced',tol=1e-10)
        svm_clf_out.fit(X_new_out, y_new_out)

        decision_values = svm_clf_out.decision_function(X_new_out)
        y_new_distance = np.where(y_new_out == 0, -1, 1)
        margin = y_new_distance * decision_values

        if np.all(margin >= 0.9990):
            scores = svm_clf_out.decision_function(X_new_test_out)
            y_pred = (scores > custom).astype(int)
            best_accurancy.append(accuracy_score(y_new_test_out, y_pred))
            best_recall.append(recall_score(y_new_test_out, y_pred))
            best_precision.append(precision_score(y_new_test_out, y_pred))
            best_f1_score.append(f1_score(y_new_test_out, y_pred))
            best_cs.append(best_c)
        else:
            print("外側の検証で学習率が100%になりませんでした。")
            gaku_so = False
            break

    if gaku_so:
        print(f"閾値{custom}の時の")
        print("SVM ダブルクロスバリデーションでの結果")
        print(sum(best_accurancy)/len(best_accurancy))
        print(sum(best_recall) / len(best_recall))
        print(sum(best_precision) / len(best_precision))
        print(sum(best_f1_score) / len(best_f1_score))
        print(best_cs)







"""
# 内側の層化k分割交差検証とグリッドサーチ（k=10）
inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# 外側の層化k分割交差検証（k=10）
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

C=[100, 1000, 10000, 100000]
Gamma=[0.01, 0.1, 1]

custom_threshold = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -0.91, -0.92, -0.93, -0.94, -0.95, -0.96, -0.97, -0.98, -0.99, -1.00, -1000]
best_accurancy = [[] for _ in range(len(custom_threshold))]
best_recall=[[] for _ in range(len(custom_threshold))]
best_precision=[[] for _ in range(len(custom_threshold))]
best_f1_score=[[] for _ in range(len(custom_threshold))]
best_cs = [[] for _ in range(len(custom_threshold))]
best_gammas = [[] for _ in range(len(custom_threshold))]
gaku_so=True

for train_index, test_index in outer_cv.split(X, labels):
    #print("外側の分割 - 学習データ:", train_index)
    best_score = 0
    best_c = 0
    best_gamma = 0

    for c in C:
        for gamma in Gamma:
            score = []
            gaku = True

            for inner_train_index, inner_test_index in inner_cv.split(X[train_index], labels[train_index]):

                inner_train_indices_in_original = train_index[inner_train_index]
                inner_test_indices_in_original = train_index[inner_test_index]
                X_new_in = X[inner_train_indices_in_original]  # スパース行列 X から部分行列を取得
                y_new_in = labels[inner_train_indices_in_original]
                X_new_test_in = X[inner_test_indices_in_original]
                y_new_test_in = labels[inner_test_indices_in_original]

                svm_clf = svm.SVC(kernel='rbf', C=c, gamma =gamma, class_weight='balanced',tol=1e-10)
                # モデルを訓練データで学習
                svm_clf.fit(X_new_in, y_new_in)                # 3. decision_function() を使って (w・x + b) を計算
                decision_values = svm_clf.decision_function(X_new_in)
                y_new_distance= np.where(y_new_in == 0, -1, 1)
                margin = y_new_distance*decision_values
                #print(margin)

                if np.all(margin >= 0.9990):
                    sco = svm_clf.score(X_new_test_in, y_new_test_in)
                    score.append(sco)
                else:
                    gaku = False
                    break

            if gaku:
                average_score = sum(score)/len(score)
                if best_score < average_score:
                    best_score = average_score
                    best_c = c
                    best_gamma = gamma

    if best_score == 0:
        gaku_so = False
        print("内側の検証で学習率100%のものはありませんでした")
        break


    X_new_out = X[train_index]  # スパース行列 X から部分行列を取得
    y_new_out = labels[train_index]
    X_new_test_out = X[test_index]
    y_new_test_out = labels[test_index]

    svm_clf_out = svm.SVC(kernel='rbf', C=best_c, gamma=best_gamma, class_weight='balanced',tol=1e-10)
    svm_clf_out.fit(X_new_out, y_new_out)

    decision_values = svm_clf_out.decision_function(X_new_out)
    y_new_distance = np.where(y_new_out == 0, -1, 1)
    margin = y_new_distance * decision_values

    if np.all(margin >= 0.9990):
        scores = svm_clf_out.decision_function(X_new_test_out)
        for i in range(len(custom_threshold)):
            y_pred = (scores > custom_threshold[i]).astype(int)
            best_accurancy[i].append(accuracy_score(y_new_test_out, y_pred))
            best_recall[i].append(recall_score(y_new_test_out, y_pred))
            best_precision[i].append(precision_score(y_new_test_out, y_pred))
            best_f1_score[i].append(f1_score(y_new_test_out, y_pred))
            best_cs[i].append(best_c)
            best_gammas[i].append(best_gamma)
    else:
        print("外側の検証で学習率が100%になりませんでした。")
        gaku_so = False
        break

if gaku_so:
    for i in range(len(custom_threshold)):
        print(f"閾値{custom_threshold[i]}の時の")
        print("SVM ダブルクロスバリデーションでの結果")
        print(f"{sum(best_accurancy[i])/len(best_accurancy[i]):.3f}")
        print(f"{sum(best_recall[i]) / len(best_recall[i]):.3f}")
        print(f"{sum(best_precision[i]) / len(best_precision[i]):.3f}")
        print(f"{sum(best_f1_score[i]) / len(best_f1_score[i]):.3f}")
        print(best_cs[i])
        print(best_gammas[i])
"""
