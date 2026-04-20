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


import random
def rarara(documnents, labels):
    # documentsとlabelsをペアにしてリスト化
    combined = list(zip(documents, labels))

    # ランダムにシャッフル
    random.shuffle(combined)

    # シャッフル後にそれぞれのリストに分割
    shuffled_documents, shuffled_labels = zip(*combined)

    # 結果をリストに戻す
    shuffled_documents = list(shuffled_documents)
    shuffled_labels = list(shuffled_labels)


    return shuffled_documents,shuffled_labels

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


print(len(documents_1))
print(len(documents_0))
documents = documents_1 + documents_0
print(documents[152])

timings = {}
# 開始時刻の記録
start_time = time.time()
# 最初の30個を詐欺(fraud=1)としてラベル付け、残りを詐欺ではない(fraud=0)とする
labels = [1] * len(documents_1) + [0] * (len(documents) - len(documents_1))
print(labels)
#doc2vecを行う
"""
def preprocess_text(doc):
    # トークン化と小文字化
    tokens = gensim.utils.simple_preprocess(doc, deacc=True, min_len=3)
    # ストップワードの除去
    tokens = [token for token in tokens if token not in STOPWORDS]
    return ' '.join(tokens)

# 文書に前処理を適用

processed_documents = [preprocess_text(doc) for doc in documents]
# 文書をDoc2VecのTaggedDocument形式に変換
tagged_documents = [TaggedDocument(words=gensim.utils.simple_preprocess(doc), tags=[i]) for i, doc in enumerate(processed_documents)]

# Doc2Vecモデルの構築
model = Doc2Vec(vector_size=100, window=5, min_count=2, workers=4, epochs=40, seed=42)
model.build_vocab(tagged_documents)
model.train(tagged_documents, total_examples=model.corpus_count, epochs=model.epochs)

# 文書をベクトル化
X = [model.dv[i] for i in range(len(tagged_documents))]
print(f"Number of document vectors: {len(X)}")
"""

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
"""
# TFの計算（TF-IDFから変更）

vectorizer = CountVectorizer(
    stop_words='english',  # scikit-learnのストップワードを使用
    min_df=0.02  # 2%未満の出現率の単語を無視
)
X = vectorizer.fit_transform(processed_docs)
"""

# 特徴量の数を取得
feature_names = vectorizer.get_feature_names_out()
num_features = len(feature_names)
print("Number of features:", num_features)

# 特徴量の削除

#　誤分類の直前の特徴量の削除
#removed_features = [3584, 5632, 6146, 1539, 4611, 6149, 5634, 2051, 1034, 6169, 6170, 4639, 6175, 5160, 3627, 2096, 4148, 3127, 567, 6204, 4157, 6209, 578, 2113, 1098, 1103, 594, 91, 92, 1119, 6241, 5730, 615, 1644, 4212, 3191, 1143, 2169, 2688, 4741, 646, 3719, 5260, 5778, 5268, 2719, 5284, 2727, 5289, 5290, 5803, 685, 1198, 5805, 4272, 2739, 2740, 1203, 2230, 3254, 696, 3259, 6331, 5821, 4286, 1214, 3265, 3272, 2761, 3786, 713, 4297, 2767, 722, 5843, 5333, 3798, 2274, 2807, 4345, 4857, 1792, 3334, 1287, 3847, 5383, 3340, 3858, 5398, 4375, 3351, 2838, 4891, 4380, 1321, 2857, 1844, 2357, 4416, 2881, 834, 4941, 5973, 1878, 4439, 4953, 860, 4445, 2911, 2912, 352, 354, 4469, 6005, 2428, 894, 384, 4996, 4485, 1926, 2952, 905, 3986, 1939, 5012, 1945, 921, 5532, 1949, 2464, 4000, 6055, 2999, 6071, 5560, 4023, 2492, 4030, 5569, 6083, 5571, 1989, 6086, 2501, 456, 4550, 5580, 4557, 3023, 4561, 4053, 469, 4055, 2522, 1499, 3548, 3036, 2012, 4578, 995, 4069, 487, 4071, 2029, 1518, 1017, 5619, 4083, 5108, 1016, 2041, 1021, 2047]

# 誤分類を起こした枝の特徴量の削除
#removed_features = [4096, 6146, 2051, 4100, 6149, 2052, 4118, 6169, 6170, 28, 6175, 2083, 39, 2096, 52, 4148, 54, 6204, 4157, 2113, 6209, 6212, 2118, 2129, 91, 92, 6241, 97, 102, 4200, 4211, 4212, 4216, 121, 2169, 124, 4222, 2194, 2195, 6299, 2213, 4264, 4272, 6324, 4276, 2230, 4278, 4280, 4282, 6331, 6333, 4286, 4297, 2253, 205, 2257, 6357, 2265, 4314, 6364, 4317, 224, 2274, 4325, 6376, 4329, 239, 4345, 259, 2308, 4357, 4362, 269, 4365, 4375, 4379, 4380, 4388, 297, 2357, 2359, 4409, 4416, 4420, 326, 336, 4439, 4445, 352, 353, 354, 362, 2412, 2415, 371, 4469, 376, 2428, 384, 4484, 4485, 4486, 391, 409, 2464, 4514, 4517, 425, 428, 2477, 2483, 4539, 2492, 2497, 2501, 4550, 455, 456, 4557, 4560, 4561, 467, 469, 470, 2522, 4578, 483, 487, 513, 4611, 4625, 531, 2583, 540, 4638, 2591, 4639, 2593, 567, 576, 578, 2627, 4686, 591, 594, 595, 4692, 2652, 4703, 615, 4713, 4725, 2688, 4738, 4741, 646, 2702, 4752, 2707, 666, 4763, 668, 2719, 2727, 685, 4783, 2739, 2740, 696, 2747, 2750, 706, 4803, 2761, 713, 2764, 4812, 2767, 722, 4822, 726, 733, 2781, 4849, 2807, 4857, 767, 2815, 2819, 2838, 2840, 4891, 4900, 804, 2853, 2857, 814, 823, 4926, 2881, 834, 4941, 852, 4949, 4950, 2905, 4953, 860, 4959, 2912, 2911, 864, 2914, 894, 4996, 2952, 5000, 905, 5012, 5014, 921, 5026, 940, 2999, 971, 974, 3023, 980, 5083, 3036, 5087, 5088, 995, 5108, 1016, 3065, 1017, 1021, 1025, 5124, 1033, 1034, 5135, 3093, 3100, 3103, 3110, 5160, 1067, 3117, 5174, 3127, 5186, 3139, 1098, 1103, 1104, 5200, 1119, 1120, 3173, 3180, 1143, 3191, 1144, 1146, 3201, 5256, 1160, 5260, 5264, 1171, 5268, 5284, 1189, 5289, 5290, 1198, 1203, 3254, 5306, 3259, 5308, 1214, 3265, 3267, 1220, 5315, 3272, 5333, 1238, 5344, 5356, 3317, 3328, 3334, 1287, 1286, 5383, 3340, 5389, 5388, 3349, 5398, 3351, 1321, 3373, 1352, 3406, 5462, 5464, 5476, 5478, 3436, 3439, 1398, 5501, 1409, 3478, 5532, 1441, 1443, 3497, 1451, 1458, 3506, 5560, 1467, 3515, 5569, 5571, 1477, 3527, 5580, 5593, 1499, 3547, 3548, 1507, 1518, 5619, 3584, 5632, 5634, 1539, 5652, 1564, 3627, 3655, 1619, 3670, 5730, 1644, 3710, 3719, 1679, 5778, 5787, 5789, 3742, 5796, 5803, 3755, 5805, 5809, 1714, 3763, 5821, 3786, 5843, 3798, 1760, 3816, 3817, 3820, 1792, 3847, 1803, 1805, 5905, 3858, 5912, 1820, 3871, 3876, 1836, 1837, 5940, 3893, 1844, 1853, 3906, 1868, 1869, 3924, 5973, 1878, 5985, 3939, 3943, 1902, 3952, 6005, 1916, 1926, 3978, 3979, 6028, 3986, 1939, 3990, 3993, 1945, 1949, 3998, 4000, 6055, 1971, 6071, 4024, 4023, 4028, 4029, 4030, 6079, 6083, 1989, 6086, 1992, 6093, 6096, 6099, 4053, 4055, 2010, 6108, 2012, 2013, 4069, 4071, 6120, 4072, 2029, 2032, 4083, 4086, 2041, 2045, 6142, 2047]

#詐欺でない確率をあげる単語を50個削除する（詐欺の可能性が高くなる）
#removed_features = [3710, 567, 5796, 4379, 1098, 6083, 5031, 5087, 5289, 3439, 548, 974, 3670, 2464, 3483, 376, 1867, 3368, 814, 5077, 4365, 2344, 4200, 5310, 6123, 4484, 2569, 5819, 696, 4351, 2194, 3947, 161, 5142, 3893, 607, 1398, 3036, 3888, 5834, 591, 3436, 4312, 4812, 734, 166, 2355, 3686, 5014, 5401]

#誤分類の直前の特徴量の削除＋詐欺の可能性を下げる単語を削除
#removed_features = [3584, 5632, 6146, 1539, 4611, 6149, 5634, 2051, 2569, 1034, 5142, 6169, 6170, 2041, 6175, 4639, 548, 5160, 3627, 2096, 4148, 567, 3127, 6204, 4157, 6209, 578, 2113, 1098, 1103, 591, 594, 3670, 91, 92, 1119, 607, 6241, 5730, 3686, 615, 4200, 1644, 4212, 1143, 3191, 2169, 3710, 2688, 4741, 646, 3719, 5260, 5778, 2194, 5268, 2719, 161, 5284, 5796, 166, 2727, 5289, 5290, 5803, 685, 1198, 5805, 4272, 2739, 2740, 1203, 2230, 3254, 696, 3259, 6331, 5821, 4286, 1214, 5310, 3265, 5819, 3272, 2761, 3786, 713, 4297, 5834, 4812, 2767, 722, 5843, 5333, 3798, 4312, 734, 2274, 2807, 4345, 4857, 4351, 1792, 3334, 1287, 3847, 5383, 3340, 4365, 3858, 5398, 4375, 3351, 2838, 5401, 4891, 4380, 4379, 3368, 1321, 2857, 2344, 814, 3888, 2355, 1844, 2357, 3893, 4416, 2881, 834, 1867, 4941, 5973, 1878, 4439, 4953, 860, 4445, 2911, 2912, 352, 354, 3947, 3436, 3439, 4469, 6005, 1398, 376, 2428, 894, 384, 4996, 4485, 1926, 4484, 2952, 905, 3986, 1939, 5012, 5014, 1945, 921, 3483, 5532, 1949, 2464, 4000, 6055, 5031, 2999, 6071, 5560, 4023, 2492, 4030, 5569, 6083, 5571, 1989, 6086, 2501, 456, 4550, 5580, 4557, 974, 3023, 4561, 4053, 469, 4055, 5077, 2522, 1499, 3548, 3036, 2012, 5087, 4578, 995, 4069, 487, 4071, 6123, 2029, 1518, 5619, 4083, 5108, 1016, 1017, 1021, 2047]

#誤分類を起こした枝の特徴量の削除＋詐欺の可能性を下げる単語の削除
removed_features = [4096, 6146, 2051, 4100, 6149, 2052, 4118, 6169, 6170, 28, 6175, 2083, 39, 2096, 52, 4148, 54, 6204, 4157, 2113, 6209, 6212, 2118, 2129, 91, 92, 6241, 97, 102, 4200, 4211, 4212, 4216, 2169, 121, 124, 4222, 2194, 2195, 6299, 161, 2213, 166, 4264, 4272, 6324, 4276, 2230, 4278, 4280, 4282, 6331, 6333, 4286, 4297, 2253, 205, 2257, 6357, 4312, 2265, 4314, 6364, 4317, 224, 2274, 4325, 6376, 4329, 239, 4345, 4351, 259, 2308, 4357, 4362, 269, 4365, 4375, 4379, 4380, 4388, 2344, 297, 2355, 2357, 2359, 4409, 4416, 4420, 326, 336, 4439, 4445, 352, 353, 354, 362, 2412, 2415, 371, 4469, 376, 2428, 384, 4484, 4485, 4486, 391, 409, 2464, 4514, 4517, 425, 428, 2477, 2483, 4539, 2492, 2497, 2501, 4550, 455, 456, 4557, 4560, 4561, 467, 469, 470, 2522, 4578, 483, 487, 513, 4611, 2569, 4625, 531, 2583, 540, 4638, 2591, 4639, 2593, 548, 567, 576, 578, 2627, 4686, 591, 594, 595, 4692, 2652, 4703, 607, 615, 4713, 4725, 2688, 4738, 4741, 646, 2702, 4752, 2707, 666, 4763, 668, 2719, 2727, 685, 4783, 2739, 2740, 696, 2747, 2750, 706, 4803, 2761, 713, 2764, 4812, 2767, 722, 4822, 726, 733, 2781, 734, 4849, 2807, 4857, 767, 2815, 2819, 2838, 2840, 4891, 4900, 804, 2853, 2857, 814, 823, 4926, 2881, 834, 4941, 852, 4949, 4950, 2905, 4953, 860, 4959, 2912, 2911, 864, 2914, 894, 4996, 2952, 5000, 905, 5012, 5014, 921, 5026, 5031, 940, 2999, 971, 974, 3023, 980, 5077, 5083, 3036, 5087, 5088, 995, 5108, 1016, 3065, 1017, 1021, 1025, 5124, 1033, 1034, 5135, 3093, 5142, 3100, 3103, 3110, 5160, 1067, 3117, 5174, 3127, 5186, 3139, 1098, 1103, 1104, 5200, 1119, 1120, 3173, 3180, 1143, 3191, 1144, 1146, 3201, 5256, 1160, 5260, 5264, 1171, 5268, 5284, 1189, 5289, 5290, 1198, 1203, 3254, 5306, 3259, 5308, 1214, 5310, 3265, 5315, 3267, 1220, 3272, 5333, 1238, 5344, 5356, 3317, 3328, 3334, 1287, 1286, 5383, 3340, 5389, 5388, 3349, 5398, 3351, 5401, 3368, 1321, 3373, 1352, 3406, 5462, 5464, 5476, 5478, 3436, 3439, 1398, 5501, 1409, 3478, 3483, 5532, 1441, 1443, 3497, 1451, 1458, 3506, 5560, 1467, 3515, 5569, 5571, 1477, 3527, 5580, 5593, 1499, 3547, 3548, 1507, 1518, 5619, 3584, 5632, 5634, 1539, 5652, 1564, 3627, 3655, 1619, 3670, 5730, 3686, 1644, 3710, 3719, 1679, 5778, 5787, 5789, 3742, 5796, 5803, 3755, 5805, 5809, 1714, 3763, 5819, 5821, 3786, 5834, 5843, 3798, 1760, 3816, 3817, 3820, 1792, 3847, 1803, 1805, 5905, 3858, 5912, 1820, 3871, 3876, 1836, 1837, 3888, 5940, 3893, 1844, 1853, 3906, 1867, 1868, 1869, 3924, 5973, 1878, 5985, 3939, 3943, 3947, 1902, 3952, 6005, 1916, 1926, 3978, 3979, 6028, 3986, 1939, 3990, 3993, 1945, 1949, 3998, 4000, 6055, 1971, 6071, 4024, 4023, 4028, 4029, 4030, 6079, 6083, 1989, 6086, 1992, 6093, 6096, 6099, 4053, 4055, 2010, 6108, 2012, 2013, 4069, 4071, 6120, 4072, 6123, 2029, 2032, 4083, 4086, 2041, 2045, 6142, 2047]


print("削除した特徴量の数", len(removed_features))
#X_s = np.delete(X.toarray(), removed_features, axis=1)


# 結果の表示
#print("Vocabulary:\n", vectorizer.get_feature_names_out()可能性をあげる単語と下げる単語の出力するプログラム
"""
# 特徴量を詐欺（ラベル1）と対照群（ラベル0）に分割
X_fraud = X[np.array(labels) == 1]  # 詐欺データ
X_control = X[np.array(labels) == 0]  # 対照群データ

# 単語ごとのTF-IDF特徴量の平均値を計算
fraud_mean = X_fraud.mean(axis=0).A1  # 詐欺データの平均
control_mean = X_control.mean(axis=0).A1  # 対照群データの平均

# 平均の差を計算
mean_diff = fraud_mean - control_mean

# 平均の差をデータフレームにまとめ、絶対値でソート
terms = np.array(vectorizer.get_feature_names_out())  # 単語一覧
mean_diff_df = pd.DataFrame({
    'term': terms,
    'mean_diff': mean_diff,
    'abs_mean_diff': np.abs(mean_diff)
})

# 絶対値が上位の単語を符号で分けて抽出
positive_terms = mean_diff_df[mean_diff_df['mean_diff'] > 0].sort_values(by='abs_mean_diff', ascending=False).head(100)
negative_terms = mean_diff_df[mean_diff_df['mean_diff'] < 0].sort_values(by='abs_mean_diff', ascending=False).head(100)

# 結果の確認
print("Positive terms (Top 50):")
print(positive_terms[['term', 'mean_diff']])

print("Negative terms (Top 50):")
print(negative_terms[['term', 'mean_diff']])

# 単語が何番目にあるかを確認（リストにして後で使えるように）
positive_term_indices = [vectorizer.vocabulary_[term] for term in positive_terms['term']]
negative_term_indices = [vectorizer.vocabulary_[term] for term in negative_terms['term']]

print("Indices of Positive terms:", positive_term_indices)
print("Indices of Negative terms:", negative_term_indices)

# トップ50の特徴インデックスを統合
top_indices = positive_term_indices + negative_term_indices

# 元の特徴量行列 X から該当する列だけを抜き出す
X_top_features = X[:, top_indices]

# 結果の確認
print("元の X の形状:", X.shape)
print("抽出後の X_top_features の形状:", X_top_features.shape)
print(X_top_features)
X=X_top_features
"""
#実験環境再現関数
"""
def senkouzikkenn(param_grid, clf_name, X, labels, name):
    clf = clf_name
    # 内側の層化k分割交差検証とグリッドサーチ（k=10）
    inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=inner_cv, scoring='accuracy')

    # 外側の層化k分割交差検証（k=10）
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # cross_val_predict を使って予測を取得
    predictions = cross_val_predict(grid_search, X, labels, cv=outer_cv)

    # 実際のラベルは元のデータセットの labels
    true_labels = labels

    # 実際のラベルと予測ラベルの比較用にTP, TN, FP, FNをカウント
    TP = TN = FP = FN = 0

    for true, pred in zip(true_labels, predictions):
        if true == 1 and pred == 1:
            TP += 1
        elif true == 0 and pred == 0:
            TN += 1
        elif true == 0 and pred == 1:
            FP += 1
        elif true == 1 and pred == 0:
            FN += 1

    # TP, TN, FP, FNの結果を表示
    print(name)
    print(f"True Positives (TP): {TP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")

    # 指定された評価指標を計算
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # 計算結果を表示
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

    # 実際のラベルと予測ラベルをDataFrameで表示
    # pandasのオプションを設定して全ての行を表示できるようにする
"""

#先行研究の再現
"""
#SVMの先行研究
param_grid_1 = {
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'C': [0.1, 0.5, 1, 3, 5],
    'gamma': [0.01, 0.001, 0.0001]
}
SVM_clf = svm.SVC(class_weight='balanced', random_state=42)
senkouzikkenn(param_grid_1, SVM_clf, X, labels, "SVMの先行研究の精度")

#ランダムフォレストの先行研究
param_grid_2 = {
    'n_estimators': [30, 50, 100, 200, 300],
    'max_depth': [None, 2, 3, 5, 10, 30],
}
ran_clf = RandomForestClassifier(class_weight='balanced', random_state=42)
senkouzikkenn(param_grid_2, ran_clf, X, labels, "ランダムフォレストの先行研究の精度")

#XGBoostの先行研究
param_grid_3 = {
    'max_depth': [3, 6, 9],
    'learning_rate': [0.1, 0.3, 0.5],
    'min_child_weight': [1, 3, 5]
}
XGB_clf = XGBClassifier(random_state=42, scale_pos_weight=148/99)
senkouzikkenn(param_grid_3, XGB_clf, X, labels, "XGBoostの先行研究の精度")

#lightGBMの先行研究
param_grid_4 = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [-1, 1, 3, 5],
    'num_leaves': [6, 11, 21, 31]
}
GBM_clf = LGBMClassifier(random_state=42, class_weight='balanced', verbosity=-1)
senkouzikkenn(param_grid_4, GBM_clf, X, labels, "lightGBMの先行研究の精度")
"""

#先行研究の実験環境
"""
param_grid = {
    'n_estimators': [30, 50, 100, 200, 300],
    'max_depth': [None, 2, 3, 5, 10, 30],
}

# ランダムフォレスト分類器の定義
clf = RandomForestClassifier(class_weight='balanced', random_state=42)



# 内側の層化k分割交差検証とグリッドサーチ（k=10）
inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=inner_cv, scoring='accuracy')


# 外側の層化k分割交差検証（k=10）
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# cross_val_predict を使って予測を取得
predictions = cross_val_predict(grid_search, X, labels, cv=outer_cv)

# 実際のラベルは元のデータセットの labels
true_labels = labels

# 実際のラベルと予測ラベルの比較用にTP, TN, FP, FNをカウント
TP = TN = FP = FN = 0

for true, pred in zip(true_labels, predictions):
    if true == 1 and pred == 1:
        TP += 1
    elif true == 0 and pred == 0:
        TN += 1
    elif true == 0 and pred == 1:
        FP += 1
    elif true == 1 and pred == 0:
        FN += 1

# TP, TN, FP, FNの結果を表示
print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")

# 指定された評価指標を計算
accuracy = (TP + TN) / (TP + TN + FP + FN)
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# 計算結果を表示
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1_score:.4f}")

# 実際のラベルと予測ラベルをDataFrameで表示
# pandasのオプションを設定して全ての行を表示できるようにする
pd.set_option('display.max_rows', None)

# 実際のラベルと予測ラベルをDataFrameで表示
results_df = pd.DataFrame({
    'True Label': true_labels,
    'Prediction': predictions
})

# 結果を表示
print(results_df)
"""



#SVMでの実験（ダブルクロスバリデーションver）
"""
labels = np.array(labels)
# 内側の層化k分割交差検証とグリッドサーチ（k=10）
inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# 外側の層化k分割交差検証（k=10）
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

kernel = ['linear', 'rbf', 'sigmoid']
gamma = [0.01, 0.001, 0.0001]
best_accurancy = []
best_recall=[]
best_precision=[]
best_f1_score=[]
best_kernels = []
best_gs = []
gaku_so=True

# 実際のラベルと予測ラベルの比較用にTP, TN, FP, FNをカウント
TP = TN = FP = FN = 0

for train_index, test_index in outer_cv.split(X, labels):
    #print("外側の分割 - 学習データ:", train_index)
    best_score = 0
    best_kernel = 0
    best_g = 0

    for k in kernel:
        for g in gamma:
            score = []
            gaku = True

            for inner_train_index, inner_test_index in inner_cv.split(X[train_index], labels[train_index]):

                inner_train_indices_in_original = train_index[inner_train_index]
                inner_test_indices_in_original = train_index[inner_test_index]
                X_new = X[inner_train_indices_in_original]  # スパース行列 X から部分行列を取得
                y_new = labels[inner_train_indices_in_original]
                X_new_test = X[inner_test_indices_in_original]
                y_new_test = labels[inner_test_indices_in_original]

                # 内側の元のXに対応するインデックスを表示
                #print("X",X_new)
                #print("label")
                #print(y_new)
                #print(len(y_new))
                #print("内側の分割 - 学習データ (元のX):", inner_train_indices_in_original)
                #print(len(inner_train_indices_in_original))
                #print("内側の分割 - テストデータ (元のX):", inner_test_indices_in_original)


                svm_clf = svm.SVC(kernel=k, C=1e10, gamma = g, class_weight='balanced')
                # モデルを訓練データで学習
                svm_clf.fit(X_new, y_new)

                # テストデータで予測
                y_pred = svm_clf.predict(X_new)
                cosuu = np.sum((y_new == 1) & (y_pred == 1))
                kazu = np.sum((y_new == 0) & (y_pred == 0))
                goukei = cosuu + kazu
                #print(k,g)
                #print(goukei)
                #print(y_pred)
                if goukei == len(y_pred):
                    s = svm_clf.score(X_new_test,y_new_test)
                    score.append(s)
                else:
                    gaku = False
                    break

            if gaku:
                average_score = sum(score)/len(score)
                if best_score < average_score:
                    best_score = average_score
                    best_g = g
                    best_kernel = k

    if best_score == 0:
        gaku_so = False
        print("内側の検証で学習率100%のものはありませんでした")
        break


    X_new = X[train_index]  # スパース行列 X から部分行列を取得
    y_new = labels[train_index]
    X_new_test = X[test_index]
    y_new_test = labels[test_index]

    svm_clf = svm.SVC(kernel=best_kernel, C=1e10, gamma=best_g, class_weight='balanced')
    svm_clf.fit(X_new, y_new)

    # テストデータで予測
    y_pred = svm_clf.predict(X_new)
    cosuu = np.sum((y_new == 1) & (y_pred == 1))
    kazu = np.sum((y_new == 0) & (y_pred == 0))
    goukei = cosuu + kazu
    if goukei == len(y_pred):
        y_pred = svm_clf.predict(X_new_test)
        best_accurancy.append(accuracy_score(y_new_test, y_pred))
        best_recall.append(recall_score(y_new_test, y_pred))
        best_precision.append(precision_score(y_new_test, y_pred))
        best_f1_score.append(f1_score(y_new_test, y_pred))
        best_kernels.append(best_kernel)
        best_gs.append(best_g)
        for true, pred in zip(y_new_test, y_pred):
            if true == 1 and pred == 1:
                TP += 1
            elif true == 0 and pred == 0:
                TN += 1
            elif true == 0 and pred == 1:
                FP += 1
            elif true == 1 and pred == 0:
                FN += 1

    else:
        print("外側の検証で学習率が100%になりませんでした。")
        gaku_so = False
        break


if gaku_so:
    print("SVM ダブルクロスバリデーションでの結果")
    print(sum(best_accurancy)/len(best_accurancy))
    print(sum(best_recall) / len(best_recall))
    print(sum(best_precision) / len(best_precision))
    print(sum(best_f1_score) / len(best_f1_score))
    print(best_accurancy)
    print(best_recall)
    print(best_precision)
    print(best_f1_score)
    print(best_gs)
    print(best_kernels)
    print(" ")

    # TP, TN, FP, FNの結果を表示
    print(f"True Positives (TP): {TP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    # 指定された評価指標を計算
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # 計算結果を表示
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(" ")
"""


#SVMでの実験（'linear'クロスバリデーションver）

X = np.delete(X.toarray(), 152, axis=0)
labels.pop(152)

labels = np.array(labels)
# 外側の層化k分割交差検証（k=10）
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

kernel = ['linear', 'rbf', 'sigmoid']
gamma = [1]
C = [10000, 50000, 100000, 500000]

best_accurancy = []
best_recall = []
best_precision = []
best_f1_score = []
best_Cs = []
best_gs = []

gakusyuu_100 = []

gaku_so = True

wrong_data = []

# 実際のラベルと予測ラベルの比較用にTP, TN, FP, FNをカウント
TP = TN = FP = FN = 0

for train_index, test_index in outer_cv.split(X, labels):
    #print("外側の分割 - 学習データ:", train_index)
    best_score = 0
    best_c = 0
    best_g = 0

    X_new = X[train_index]  # スパース行列 X から部分行列を取得
    y_new = labels[train_index]
    X_new_test = X[test_index]
    y_new_test = labels[test_index]

    for c in C:
        for g in gamma:



            svm_clf = svm.SVC(kernel='linear', C=c, gamma=g, class_weight='balanced', random_state=42)
            svm_clf.fit(X_new, y_new)

            # テストデータで予測
            y_pred = svm_clf.predict(X_new)
            cosuu = np.sum((y_new == 1) & (y_pred == 1))
            kazu = np.sum((y_new == 0) & (y_pred == 0))
            goukei = cosuu + kazu
            #print(k, g)
            #print(goukei)
            #print(len(y_pred))
            #print(y_pred)

            if goukei == len(y_pred):
                y_pred_gaku = svm_clf.predict(X_new_test)
                s = svm_clf.score(X_new_test, y_new_test)
                gaku_100 = []
                accuracy = accuracy_score(y_new_test, y_pred_gaku)
                recall = recall_score(y_new_test, y_pred_gaku)
                precision = precision_score(y_new_test, y_pred_gaku)
                f1_scores = f1_score(y_new_test, y_pred_gaku)
                gaku_100.append(c)
                gaku_100.append(g)
                gaku_100.append(accuracy)
                gaku_100.append(recall)
                gaku_100.append(precision)
                gaku_100.append(f1_scores)
                gakusyuu_100.append(gaku_100)

                # 現在の分割で間違った予測を特定
                wrong_indices = [test_index[i] for i, (true, pred) in enumerate(zip(y_new_test, y_pred_gaku)) if
                                 true != pred]

                # インデックス番号を調整
                adjusted_indices = [idx + 1 if idx >= 152 else idx for idx in wrong_indices]


                # 全体のリストに追加
                wrong_data.extend(adjusted_indices)

                if best_score < s:
                    best_score = s
                    best_c = c
                    best_g = g


    if best_score == 0:
        gaku_so = False
        print("外の検証で学習率100%のものはありませんでした")
        break
    else :
        svm_clf = svm.SVC(kernel='linear', C=best_c, gamma=best_g, class_weight='balanced', random_state=42)
        svm_clf.fit(X_new, y_new)
        y_pred = svm_clf.predict(X_new_test)
        accurancy = accuracy_score(y_new_test, y_pred)
        best_accurancy.append(accurancy)
        recall = recall_score(y_new_test, y_pred)
        best_recall.append(recall)
        precision = precision_score(y_new_test, y_pred)
        best_precision.append(precision)
        f1_scores = f1_score(y_new_test, y_pred)
        best_f1_score.append(f1_scores)
        best_Cs.append(best_c)
        best_gs.append(best_g)

        for true, pred in zip(y_new_test, y_pred):
            if true == 1 and pred == 1:
                TP += 1
            elif true == 0 and pred == 0:
                TN += 1
            elif true == 0 and pred == 1:
                FP += 1
            elif true == 1 and pred == 0:
                FN += 1



if gaku_so:
    print("SVMでクロスバリデーションの結果'linear'")
    print(sum(best_accurancy) / len(best_accurancy))
    print(sum(best_recall) / len(best_recall))
    print(sum(best_precision) / len(best_precision))
    print(sum(best_f1_score) / len(best_f1_score))
    print(best_accurancy)
    print(best_recall)
    print(best_precision)
    print(best_f1_score)
    print(best_gs)
    print(best_Cs)
    print(" ")

    # 各リストの最初の2つの要素をキーにしてグループ化
    grouped_elements = defaultdict(list)
    for item in gakusyuu_100:
        key = tuple(item[:2])  # 最初の2つの要素をキーにする
        grouped_elements[key].append(item[2:])  # 後ろの要素をリストに追加

    best_average=0.0
    best_averages= list
    best_key=list()
    # 出現回数が2回以上のキーについて、後ろの要素の平均を計算して出力
    for key, values in grouped_elements.items():
        if len(values) > 9:
            # 各リストの後ろの要素の平均を計算
            averages = [sum(elements) / len(elements) for elements in zip(*values)]
            #print(f"[C,gamma]={list(key)}")
            #print(f"[[accurancy,recall,precision,f1_socre]]")
            #print(averages)
            ab = averages[0]
            if best_average <= ab:
                best_key = list(key)
                best_average = ab
                best_averages = averages

    print(best_key)
    print(best_averages)



    # TP, TN, FP, FNの結果を表示
    print(f"True Positives (TP): {TP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    # 指定された評価指標を計算
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # 計算結果を表示
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(" ")
    wrong_data = sorted(wrong_data)
    #print(len(wrong_data))
    #print(wrong_data)



#SVMでの実験（'rbf'クロスバリデーションver）

#X = np.delete(X.toarray(), 152, axis=0)
#labels.pop(152)
wrong_data = []
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
labels = np.array(labels)
# 外側の層化k分割交差検証（k=10）
outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
kernel = ['linear', 'rbf', 'sigmoid']
C = [1000, 10000, 100000, 1000000]
gamma = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]

best_accurancy = []
best_recall = []
best_precision = []
best_f1_score = []
best_Cs = []
best_gs = []


gakusyuu_100 = []

gaku_so = True

#wrong_data = []

# 実際のラベルと予測ラベルの比較用にTP, TN, FP, FNをカウント
TP = TN = FP = FN = 0

for train_index, test_index in outer_cv.split(X, labels):
    #print("外側の分割 - 学習データ:", train_index)
    best_score = 0
    best_c = 0
    best_g = 0

    X_new = X[train_index]  # スパース行列 X から部分行列を取得
    y_new = labels[train_index]
    X_new_test = X[test_index]
    y_new_test = labels[test_index]

    for c in C:
        for g in gamma:



            svm_clf = svm.SVC(kernel='rbf', C=c, gamma=g, class_weight='balanced', random_state=42)
            svm_clf.fit(X_new, y_new)

            # テストデータで予測
            y_pred = svm_clf.predict(X_new)
            cosuu = np.sum((y_new == 1) & (y_pred == 1))
            kazu = np.sum((y_new == 0) & (y_pred == 0))
            goukei = cosuu + kazu
            #print(k, g)
            #print(goukei)
            #print(len(y_pred))
            #print(y_pred)

            if goukei == len(y_pred):
                y_pred_gaku = svm_clf.predict(X_new_test)
                s = svm_clf.score(X_new_test, y_new_test)
                gaku_100 = []
                accuracy = accuracy_score(y_new_test, y_pred_gaku)
                recall = recall_score(y_new_test, y_pred_gaku)
                precision = precision_score(y_new_test, y_pred_gaku)
                f1_scores = f1_score(y_new_test, y_pred_gaku)
                gaku_100.append(c)
                gaku_100.append(g)
                gaku_100.append(accuracy)
                gaku_100.append(recall)
                gaku_100.append(precision)
                gaku_100.append(f1_scores)
                gakusyuu_100.append(gaku_100)

                # 現在の分割で間違った予測を特定
                wrong_indices = [test_index[i] for i, (true, pred) in enumerate(zip(y_new_test, y_pred_gaku)) if
                                 true != pred]

                # インデックス番号を調整
                adjusted_indices = [idx + 1 if idx >= 152 else idx for idx in wrong_indices]


                # 全体のリストに追加
                wrong_data.extend(adjusted_indices)

                if best_score < s:
                    best_score = s
                    best_c = c
                    best_g = g


    if best_score == 0:
        gaku_so = False
        print("外の検証で学習率100%のものはありませんでした")
        break
    else :
        svm_clf = svm.SVC(kernel='rbf', C=best_c, gamma=best_g, class_weight='balanced', random_state=42)
        svm_clf.fit(X_new, y_new)
        y_pred = svm_clf.predict(X_new_test)
        accurancy = accuracy_score(y_new_test, y_pred)
        best_accurancy.append(accurancy)
        recall = recall_score(y_new_test, y_pred)
        best_recall.append(recall)
        precision = precision_score(y_new_test, y_pred)
        best_precision.append(precision)
        f1_scores = f1_score(y_new_test, y_pred)
        best_f1_score.append(f1_scores)
        best_Cs.append(best_c)
        best_gs.append(best_g)

        for true, pred in zip(y_new_test, y_pred):
            if true == 1 and pred == 1:
                TP += 1
            elif true == 0 and pred == 0:
                TN += 1
            elif true == 0 and pred == 1:
                FP += 1
            elif true == 1 and pred == 0:
                FN += 1



if gaku_so:
    print("SVMでクロスバリデーションの結果'rbf'")
    print(sum(best_accurancy) / len(best_accurancy))
    print(sum(best_recall) / len(best_recall))
    print(sum(best_precision) / len(best_precision))
    print(sum(best_f1_score) / len(best_f1_score))
    print(best_accurancy)
    print(best_recall)
    print(best_precision)
    print(best_f1_score)
    print(best_gs)
    print(best_Cs)
    print(" ")

    # 各リストの最初の2つの要素をキーにしてグループ化
    grouped_elements = defaultdict(list)
    for item in gakusyuu_100:
        key = tuple(item[:2])  # 最初の2つの要素をキーにする
        grouped_elements[key].append(item[2:])  # 後ろの要素をリストに追加

    best_average = 0.0
    best_averages = []  # 最良の平均値を持つリスト
    best_keys = []  # 最良のキーを持つリスト

    # 出現回数が2回以上のキーについて、後ろの要素の平均を計算して出力
    for key, values in grouped_elements.items():
        if len(values) > 9:
            # 各リストの後ろの要素の平均を計算
            averages = [sum(elements) / len(elements) for elements in zip(*values)]
            ab = averages[0]  # 最初の要素（平均）の値

            # 現在の平均値が最良よりも大きい場合
            if best_average < ab:
                best_average = ab
                best_averages = [averages]  # 最良の平均値を更新
                best_keys = [list(key)]  # 最良のキーを更新
            # 現在の平均値が最良と等しい場合
            elif best_average == ab:
                best_averages.append(averages)  # 最良の平均値リストに追加
                best_keys.append(list(key))  # 最良のキーリストに追加

    # best_averagesとbest_keysには同じ平均値を持つ複数の要素が格納されます

    print(best_keys)
    print(best_averages)



    # TP, TN, FP, FNの結果を表示
    print(f"True Positives (TP): {TP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    # 指定された評価指標を計算
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # 計算結果を表示
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(" ")
    wrong_data = sorted(wrong_data)
    #print(len(wrong_data))
    #print(wrong_data)

#ランダムフォレストの実験(ダブルクロスバリデーションver)
"""
#X = np.delete(X.toarray(), 152, axis=0)
#labels.pop(152)

labels = np.array(labels)
# 内側の層化k分割交差検証とグリッドサーチ（k=10）
inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# 外側の層化k分割交差検証（k=10）
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

n_estimators = [8]
max_depth = [35]

best_accurancy = []
best_recall=[]
best_precision=[]
best_f1_score=[]
best_n_estimatorses = []
best_max_depths = []
gaku_so=True

# 実際のラベルと予測ラベルの比較用にTP, TN, FP, FNをカウント
TP = TN = FP = FN = 0

for train_index, test_index in outer_cv.split(X, labels):
    #print("外側の分割 - 学習データ:", train_index)
    best_score = 0
    best_n_estimators = 0
    best_max_depth= 0

    for n in n_estimators:
        for m in max_depth:
            score = []
            gaku = True

            for inner_train_index, inner_test_index in inner_cv.split(X[train_index], labels[train_index]):

                inner_train_indices_in_original = train_index[inner_train_index]
                inner_test_indices_in_original = train_index[inner_test_index]
                X_new = X[inner_train_indices_in_original]  # スパース行列 X から部分行列を取得
                y_new = labels[inner_train_indices_in_original]
                X_new_test = X[inner_test_indices_in_original]
                y_new_test = labels[inner_test_indices_in_original]

                # 内側の元のXに対応するインデックスを表示
                #print("X",X_new)
                #print("label")
                #print(y_new)
                #print(len(y_new))
                #print("内側の分割 - 学習データ (元のX):", inner_train_indices_in_original)
                #print(len(inner_train_indices_in_original))
                #print("内側の分割 - テストデータ (元のX):", inner_test_indices_in_original)


                ran_clf = RandomForestClassifier(n_estimators=n, max_depth=m, class_weight='balanced')
                # モデルを訓練データで学習
                ran_clf.fit(X_new, y_new)

                # テストデータで予測
                y_pred = ran_clf.predict(X_new)
                cosuu = np.sum((y_new == 1) & (y_pred == 1))
                kazu = np.sum((y_new == 0) & (y_pred == 0))
                goukei = cosuu + kazu
                #print(n,m)
                #print(goukei)
                #print(y_pred)
                if goukei == len(y_pred):
                    s = ran_clf.score(X_new_test,y_new_test)
                    score.append(s)
                else:
                    gaku = False
                    break

            if gaku:
                average_score = sum(score)/len(score)
                if best_score < average_score:
                    best_score = average_score
                    best_n_estimators = n
                    best_max_depth = m

    if best_score == 0:
        gaku_so = False
        print("内側の検証で学習率100%のものはありませんでした")
        break


    X_new = X[train_index]  # スパース行列 X から部分行列を取得
    y_new = labels[train_index]
    X_new_test = X[test_index]
    y_new_test = labels[test_index]

    ran_clf=RandomForestClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth, class_weight='balanced')
    ran_clf.fit(X_new, y_new)

    # テストデータで予測
    y_pred = ran_clf.predict(X_new)
    cosuu = np.sum((y_new == 1) & (y_pred == 1))
    kazu = np.sum((y_new == 0) & (y_pred == 0))
    goukei = cosuu + kazu
    if goukei == len(y_pred):
        y_pred = ran_clf.predict(X_new_test)
        best_accurancy.append(accuracy_score(y_new_test, y_pred))
        best_recall.append(recall_score(y_new_test, y_pred))
        best_precision.append(precision_score(y_new_test, y_pred))
        best_f1_score.append(f1_score(y_new_test, y_pred))
        best_n_estimatorses.append(best_n_estimators)
        best_max_depths.append(best_max_depth)
        for true, pred in zip(y_new_test, y_pred):
            if true == 1 and pred == 1:
                TP += 1
            elif true == 0 and pred == 0:
                TN += 1
            elif true == 0 and pred == 1:
                FP += 1
            elif true == 1 and pred == 0:
                FN += 1
    else:
        print("外側の検証で学習率が100%になりませんでした。")
        gaku_so = False
        break


if gaku_so:
    print("ランダムフォレスト　ダブルクロスバリデーションの結果")
    print(sum(best_accurancy) / len(best_accurancy))
    print(sum(best_recall) / len(best_recall))
    print(sum(best_precision) / len(best_precision))
    print(sum(best_f1_score) / len(best_f1_score))
    print(best_accurancy)
    print(best_recall)
    print(best_precision)
    print(best_f1_score)
    print(best_n_estimatorses)
    print(best_max_depths)

    # TP, TN, FP, FNの結果を表示
    print(f"True Positives (TP): {TP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    # 指定された評価指標を計算
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # 計算結果を表示
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(" ")
"""



#ランダムフォレストでの実験（クロスバリデーションver）

#X = np.delete(X.toarray(), 152, axis=0)
#labels.pop(152)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
labels = np.array(labels)
# 外側の層化k分割交差検証（k=10）
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

n_estimators = [ 200, 250, 300, 500]
max_depth = [20, 30, 40]
best_accurancy = []
best_recall=[]
best_precision=[]
best_f1_score=[]
best_n_estimatorses = []
best_max_depths = []
gaku_so = True

gakusyuu_100 = []

#wrong_data = []

# 実際のラベルと予測ラベルの比較用にTP, TN, FP, FNをカウント
TP = TN = FP = FN = 0

for train_index, test_index in outer_cv.split(X, labels):
    #print("外側の分割 - 学習データ:", train_index)
    best_max_depth = 0
    best_n_estimators = 0
    best_score = 0

    X_new = X[train_index]  # スパース行列 X から部分行列を取得
    y_new = labels[train_index]
    X_new_test = X[test_index]
    y_new_test = labels[test_index]

    for m in max_depth:
        for n in n_estimators:
            ran_clf = RandomForestClassifier(n_estimators=n, max_depth=m, class_weight='balanced', random_state=42)
            ran_clf.fit(X_new, y_new)

            # テストデータで予測
            y_pred = ran_clf.predict(X_new)
            cosuu = np.sum((y_new == 1) & (y_pred == 1))
            kazu = np.sum((y_new == 0) & (y_pred == 0))
            goukei = cosuu + kazu
            #print(k, g)
            #print(goukei)
            #print(len(y_pred))
            #print(y_pred)

            if goukei == len(y_pred):
                s = ran_clf.score(X_new_test, y_new_test)
                #print(f"一番いい精度を調べている{s}")
                y_pred_gaku = ran_clf.predict(X_new_test)
                gaku_100 = []
                accuracy = accuracy_score(y_new_test, y_pred_gaku)
                #print(f"学習率100を求めているやつ{accuracy}")
                recall = recall_score(y_new_test, y_pred_gaku)
                precision = precision_score(y_new_test, y_pred_gaku)
                f1_scores = f1_score(y_new_test, y_pred_gaku)
                gaku_100.append(m)
                gaku_100.append(n)
                gaku_100.append(accuracy)
                gaku_100.append(recall)
                gaku_100.append(precision)
                gaku_100.append(f1_scores)
                gakusyuu_100.append(gaku_100)

                # 現在の分割で間違った予測を特定
                wrong_indices = [test_index[i] for i, (true, pred) in enumerate(zip(y_new_test, y_pred_gaku)) if
                                 true != pred]

                # インデックス番号を調整
                adjusted_indices = [idx + 1 if idx >= 152 else idx for idx in wrong_indices]


                # 全体のリストに追加
                wrong_data.extend(adjusted_indices)

                if best_score < s:
                    best_score = s
                    best_max_depth = m
                    best_n_estimators = n
                    #print(f"現状の一番いい精度{best_score}")


    if best_score == 0:
        gaku_so = False
        print("外の検証で学習率100%のものはありませんでした")
        break
    else:
        ran_clf = RandomForestClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth, class_weight='balanced', random_state=42)
        ran_clf.fit(X_new, y_new)
        y_pred = ran_clf.predict(X_new_test)
        accurancy = accuracy_score(y_new_test, y_pred)
        s = ran_clf.score(X_new_test, y_new_test)
        #print(" ")
        #print(f"学習率100で一番いい精度{accurancy}")
        #print(f"確かめ{s}")
        best_accurancy.append(accurancy)
        recall = recall_score(y_new_test, y_pred)
        best_recall.append(recall)
        precision = precision_score(y_new_test, y_pred)
        best_precision.append(precision)
        f1_scores = f1_score(y_new_test, y_pred)
        best_f1_score.append(f1_scores)
        best_max_depths.append(best_max_depth)
        best_n_estimatorses.append(best_n_estimators)

        for true, pred in zip(y_new_test, y_pred):
            if true == 1 and pred == 1:
                TP += 1
            elif true == 0 and pred == 0:
                TN += 1
            elif true == 0 and pred == 1:
                FP += 1
            elif true == 1 and pred == 0:
                FN += 1



if gaku_so:
    print("ランダムフォレストでクロスバリデーションの結果")
    wrong_data = sorted(wrong_data)
    #print(len(wrong_data))
    #print(wrong_data)


    print(sum(best_accurancy) / len(best_accurancy))
    print(sum(best_recall) / len(best_recall))
    print(sum(best_precision) / len(best_precision))
    print(sum(best_f1_score) / len(best_f1_score))
    print(best_accurancy)
    print(best_recall)
    print(best_precision)
    print(best_f1_score)
    print(best_max_depths)
    print(best_n_estimatorses)
    print(" ")

    # 各リストの最初の2つの要素をキーにしてグループ化
    grouped_elements = defaultdict(list)
    for item in gakusyuu_100:
        key = tuple(item[:2])  # 最初の2つの要素をキーにする
        grouped_elements[key].append(item[2:])  # 後ろの要素をリストに追加

    best_average = 0.0
    best_averages = []  # 最良の平均値を持つリスト
    best_keys = []  # 最良のキーを持つリスト

    # 出現回数が2回以上のキーについて、後ろの要素の平均を計算して出力
    for key, values in grouped_elements.items():
        if len(values) > 9:
            # 各リストの後ろの要素の平均を計算
            averages = [sum(elements) / len(elements) for elements in zip(*values)]
            ab = averages[0]  # 最初の要素（平均）の値

            # 現在の平均値が最良よりも大きい場合
            if best_average < ab:
                best_average = ab
                best_averages = [averages]  # 最良の平均値を更新
                best_keys = [list(key)]  # 最良のキーを更新
            # 現在の平均値が最良と等しい場合
            elif best_average == ab:
                best_averages.append(averages)  # 最良の平均値リストに追加
                best_keys.append(list(key))  # 最良のキーリストに追加

    # best_averagesとbest_keysには同じ平均値を持つ複数の要素が格納されます

    print(best_keys)
    print(best_averages)

    # TP, TN, FP, FNの結果を表示
    print(f"True Positives (TP): {TP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    # 指定された評価指標を計算
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # 計算結果を表示
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(" ")


#XGBoostでの実験（ダブルクロスバリデーションver）
"""
#X = np.delete(X.toarray(), 152, axis=0)
#labels.pop(152)

labels = np.array(labels)
# 内側の層化k分割交差検証とグリッドサーチ（k=10）
inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# 外側の層化k分割交差検証（k=10）
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
max_depth=[3,6,9]
learning_rate=[0.1,0.3,0.5]
min_child_weight=[1,3,5]

best_accurancy = []
best_recall=[]
best_precision=[]
best_f1_score=[]
best_max_depths = []
best_learning_rates = []
best_min_child_weights = []
gaku_so=True

# 実際のラベルと予測ラベルの比較用にTP, TN, FP, FNをカウント
TP = TN = FP = FN = 0

for train_index, test_index in outer_cv.split(X, labels):
    #print("外側の分割 - 学習データ:", train_index)
    best_score = 0
    best_max_depth = 0
    best_learning_rate = 0
    best_min_child_weight = 0

    for m in max_depth:
        for l in learning_rate:
            for mi in min_child_weight:
                score = []
                gaku = True

                for inner_train_index, inner_test_index in inner_cv.split(X[train_index], labels[train_index]):

                    inner_train_indices_in_original = train_index[inner_train_index]
                    inner_test_indices_in_original = train_index[inner_test_index]
                    X_new = X[inner_train_indices_in_original]  # スパース行列 X から部分行列を取得
                    y_new = labels[inner_train_indices_in_original]
                    X_new_test = X[inner_test_indices_in_original]
                    y_new_test = labels[inner_test_indices_in_original]

                    # 内側の元のXに対応するインデックスを表示
                    #print("X",X_new)
                    #print("label")
                    #print(y_new)
                    #print(len(y_new))
                    #print("内側の分割 - 学習データ (元のX):", inner_train_indices_in_original)
                    #print(len(inner_train_indices_in_original))
                    #print("内側の分割 - テストデータ (元のX):", inner_test_indices_in_original)


                    
                    xgb_clf = XGBClassifier(max_depth=m, learning_rate=l, min_child_weight=mi, scale_pos_weight=148/99)
                    # モデルを訓練データで学習
                    xgb_clf.fit(X_new, y_new)

                    # テストデータで予測
                    y_pred = xgb_clf.predict(X_new)
                    cosuu = np.sum((y_new == 1) & (y_pred == 1))
                    kazu = np.sum((y_new == 0) & (y_pred == 0))
                    goukei = cosuu + kazu
                    #print(m,l,mi)
                    #print(goukei)
                    #print(y_pred)
                    if goukei == len(y_pred):
                        s = xgb_clf.score(X_new_test,y_new_test)
                        score.append(s)
                    else:
                        gaku = False
                        break

                if gaku:
                    average_score = sum(score)/len(score)
                    if best_score < average_score:
                        best_score = average_score
                        best_max_depth = m
                        best_learning_rate = l
                        best_min_child_weight=mi

    if best_score == 0:
        gaku_so = False
        print("内側の検証で学習率100%のものはありませんでした")
        break


    X_new = X[train_index]  # スパース行列 X から部分行列を取得
    y_new = labels[train_index]
    X_new_test = X[test_index]
    y_new_test = labels[test_index]

    xgb_clf = XGBClassifier(max_depth=best_max_depth, learning_rate=best_learning_rate, min_child_weight=best_learning_rate, scale_pos_weight=148/99)
    xgb_clf.fit(X_new, y_new)

    # テストデータで予測
    y_pred = xgb_clf.predict(X_new)
    cosuu = np.sum((y_new == 1) & (y_pred == 1))
    kazu = np.sum((y_new == 0) & (y_pred == 0))
    goukei = cosuu + kazu
    if goukei == len(y_pred):
        y_pred = xgb_clf.predict(X_new_test)
        best_accurancy.append(accuracy_score(y_new_test, y_pred))
        best_recall.append(recall_score(y_new_test, y_pred))
        best_precision.append(precision_score(y_new_test, y_pred))
        best_f1_score.append(f1_score(y_new_test, y_pred))
        best_max_depths.append(best_max_depth)
        best_learning_rates.append(best_learning_rate)
        best_min_child_weights.append(best_min_child_weight)
        for true, pred in zip(y_new_test, y_pred):
            if true == 1 and pred == 1:
                TP += 1
            elif true == 0 and pred == 0:
                TN += 1
            elif true == 0 and pred == 1:
                FP += 1
            elif true == 1 and pred == 0:
                FN += 1
    else:
        print("外側の検証で学習率が100%になりませんでした。")
        gaku_so = False
        break


if gaku_so:
    print("XGBoost ダブルクロスバリデーションでの結果")
    print(sum(best_accurancy)/len(best_accurancy))
    print(sum(best_recall) / len(best_recall))
    print(sum(best_precision) / len(best_precision))
    print(sum(best_f1_score) / len(best_f1_score))
    print(best_accurancy)
    print(best_recall)
    print(best_precision)
    print(best_f1_score)
    print(best_max_depths)
    print(best_learning_rates)
    print(best_min_child_weights)
    print(" ")

    # TP, TN, FP, FNの結果を表示
    print(f"True Positives (TP): {TP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    # 指定された評価指標を計算
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # 計算結果を表示
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(" ")
"""


#XGBoostでの実験（クロスバリデーションver）
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#X = np.delete(X.toarray(), 152, axis=0)
#labels.pop(152)

labels = np.array(labels)
# 外側の層化k分割交差検証（k=10）
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

max_depth = [12]
learning_rate = [0.1]
min_child_weight = [1]

best_accurancy = []
best_recall = []
best_precision = []
best_f1_score = []
best_max_depths = []
best_learning_rates = []
best_min_child_weights = []
gaku_so = True

gakusyuu_100 = []

#wrong_data = []

# 実際のラベルと予測ラベルの比較用にTP, TN, FP, FNをカウント
TP = TN = FP = FN = 0

for train_index, test_index in outer_cv.split(X, labels):
    #print("外側の分割 - 学習データ:", train_index)
    best_score = 0
    best_max_depth = 0
    best_learning_rate = 0
    best_min_child_weight = 0

    X_new = X[train_index]  # スパース行列 X から部分行列を取得
    y_new = labels[train_index]
    X_new_test = X[test_index]
    y_new_test = labels[test_index]

    for m in max_depth:
        for l in learning_rate:
            for mi in min_child_weight:
                xgb_clf = XGBClassifier(max_depth=m, learning_rate=l,
                                        min_child_weight=mi, scale_pos_weight=148/99, random_state=42)
                xgb_clf.fit(X_new, y_new)

                # テストデータで予測
                y_pred = xgb_clf.predict(X_new)
                cosuu = np.sum((y_new == 1) & (y_pred == 1))
                kazu = np.sum((y_new == 0) & (y_pred == 0))
                goukei = cosuu + kazu
                #print(k, g)
                #print(goukei)
                #print(len(y_pred))
                #print(y_pred)

                if goukei == len(y_pred):
                    s = xgb_clf.score(X_new_test, y_new_test)

                    y_pred_gaku = xgb_clf.predict(X_new_test)
                    gaku_100 = []
                    accuracy = accuracy_score(y_new_test, y_pred_gaku)
                    recall = recall_score(y_new_test, y_pred_gaku)
                    precision = precision_score(y_new_test, y_pred_gaku)
                    f1_scores = f1_score(y_new_test, y_pred_gaku)
                    gaku_100.append(m)
                    gaku_100.append(l)
                    gaku_100.append(mi)
                    gaku_100.append(accuracy)
                    gaku_100.append(recall)
                    gaku_100.append(precision)
                    gaku_100.append(f1_scores)
                    gakusyuu_100.append(gaku_100)
                    # 現在の分割で間違った予測を特定
                    wrong_indices = [test_index[i] for i, (true, pred) in enumerate(zip(y_new_test, y_pred_gaku)) if
                                     true != pred]

                    # インデックス番号を調整
                    adjusted_indices = [idx + 1 if idx >= 152 else idx for idx in wrong_indices]

                    # 全体のリストに追加
                    wrong_data.extend(adjusted_indices)
                    if best_score < s:
                        best_score = s
                        best_max_depth = m
                        best_learning_rate = l
                        best_min_child_weight = mi


    if best_score == 0:
        gaku_so = False
        print("外の検証で学習率100%のものはありませんでした")
        break
    else :
        xgb_clf = XGBClassifier(max_depth=best_max_depth, learning_rate=best_learning_rate,
                                min_child_weight=best_learning_rate, scale_pos_weight=148/99, random_state=42)
        xgb_clf.fit(X_new, y_new)
        y_pred = xgb_clf.predict(X_new_test)
        accurancy = accuracy_score(y_new_test, y_pred)
        best_accurancy.append(accurancy)
        recall = recall_score(y_new_test, y_pred)
        best_recall.append(recall)
        precision = precision_score(y_new_test, y_pred)
        best_precision.append(precision)
        f1_scores = f1_score(y_new_test, y_pred)
        best_f1_score.append(f1_scores)
        best_max_depths.append(best_max_depth)
        best_learning_rates.append(best_learning_rate)
        best_min_child_weights.append(best_min_child_weight)
        for true, pred in zip(y_new_test, y_pred):
            if true == 1 and pred == 1:
                TP += 1
            elif true == 0 and pred == 0:
                TN += 1
            elif true == 0 and pred == 1:
                FP += 1
            elif true == 1 and pred == 0:
                FN += 1



if gaku_so:
    print("XGBoostでクロスバリデーションの結果")
    wrong_data = sorted(wrong_data)
    print(len(wrong_data))
    print(wrong_data)
    """
"""
    print(sum(best_accurancy) / len(best_accurancy))
    print(sum(best_recall) / len(best_recall))
    print(sum(best_precision) / len(best_precision))
    print(sum(best_f1_score) / len(best_f1_score))
    print(best_accurancy)
    print(best_recall)
    print(best_precision)
    print(best_f1_score)
    print(best_max_depths)
    print(best_learning_rates)
    print(best_min_child_weights)
    print(" ")

    # 各リストの最初の2つの要素をキーにしてグループ化
    grouped_elements = defaultdict(list)
    for item in gakusyuu_100:
        key = tuple(item[:3])  # 最初の2つの要素をキーにする
        grouped_elements[key].append(item[3:])  # 後ろの要素をリストに追加

    best_average = 0.0
    best_averages = []  # 最良の平均値を持つリスト
    best_keys = []  # 最良のキーを持つリスト

    # 出現回数が2回以上のキーについて、後ろの要素の平均を計算して出力
    for key, values in grouped_elements.items():
        if len(values) > 9:
            # 各リストの後ろの要素の平均を計算
            averages = [sum(elements) / len(elements) for elements in zip(*values)]
            ab = averages[0]  # 最初の要素（平均）の値

            # 現在の平均値が最良よりも大きい場合
            if best_average < ab:
                best_average = ab
                best_averages = [averages]  # 最良の平均値を更新
                best_keys = [list(key)]  # 最良のキーを更新
            # 現在の平均値が最良と等しい場合
            elif best_average == ab:
                best_averages.append(averages)  # 最良の平均値リストに追加
                best_keys.append(list(key))  # 最良のキーリストに追加

    # best_averagesとbest_keysには同じ平均値を持つ複数の要素が格納されます

    print(best_keys)
    print(best_averages)

    # TP, TN, FP, FNの結果を表示
    print(f"True Positives (TP): {TP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    # 指定された評価指標を計算
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # 計算結果を表示
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(" ")
"""


# lightGBMの実験（ダブルクロスバリデーションver）
"""
#X = np.delete(X.toarray(), 152, axis=0)
#labels.pop(152)

labels = np.array(labels)
# 内側の層化k分割交差検証とグリッドサーチ（k=10）
inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# 外側の層化k分割交差検証（k=10）
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
max_depth = [-1, 1,3,5]
n_estimators = [50, 100, 200, 300, 500]
num_leaves = [6, 11, 21, 31]

best_accurancy = []
best_recall = []
best_precision = []
best_f1_score = []
best_max_depths = []
best_n_estimators = []
best_num_leaveses = []
gaku_so = True

# 実際のラベルと予測ラベルの比較用にTP, TN, FP, FNをカウント
TP = TN = FP = FN = 0

for train_index, test_index in outer_cv.split(X, labels):
    # print("外側の分割 - 学習データ:", train_index)
    best_score = 0
    best_max_depth = 0
    best_n_estimator = 0
    best_num_leaves = 0

    for m in max_depth:
        for n in n_estimators:
            for nu in num_leaves:
                score = []
                gaku = True

                for inner_train_index, inner_test_index in inner_cv.split(X[train_index], labels[train_index]):

                    inner_train_indices_in_original = train_index[inner_train_index]
                    inner_test_indices_in_original = train_index[inner_test_index]
                    X_new = X[inner_train_indices_in_original]  # スパース行列 X から部分行列を取得
                    y_new = labels[inner_train_indices_in_original]
                    X_new_test = X[inner_test_indices_in_original]
                    y_new_test = labels[inner_test_indices_in_original]

                    # 内側の元のXに対応するインデックスを表示
                    # print("X",X_new)
                    # print("label")
                    # print(y_new)
                    # print(len(y_new))
                    # print("内側の分割 - 学習データ (元のX):", inner_train_indices_in_original)
                    # print(len(inner_train_indices_in_original))
                    # print("内側の分割 - テストデータ (元のX):", inner_test_indices_in_original)
                    lgbm_clf = LGBMClassifier(max_depth=m, n_estimators=n, num_leaves=nu, class_weight='balanced', verbosity=-1)
                    
                    # モデルを訓練データで学習
                    lgbm_clf.fit(X_new, y_new)

                    # テストデータで予測
                    y_pred = lgbm_clf.predict(X_new)
                    cosuu = np.sum((y_new == 1) & (y_pred == 1))
                    kazu = np.sum((y_new == 0) & (y_pred == 0))
                    goukei = cosuu + kazu
                    # print(m,n,nu)
                    # print(goukei)
                    # print(y_pred)
                    if goukei == len(y_pred):
                        s = lgbm_clf.score(X_new_test, y_new_test)
                        score.append(s)
                    else:
                        gaku = False
                        break

                if gaku:
                    average_score = sum(score) / len(score)
                    if best_score < average_score:
                        best_score = average_score
                        best_max_depth = m
                        best_n_estimator = n
                        best_num_leaves = nu

    if best_score == 0:
        gaku_so = False
        print("内側の検証で学習率100%のものはありませんでした")
        break

    X_new = X[train_index]  # スパース行列 X から部分行列を取得
    y_new = labels[train_index]
    X_new_test = X[test_index]
    y_new_test = labels[test_index]

    lgbm_clf = LGBMClassifier(max_depth=best_max_depth, n_estimators=best_n_estimator, num_leaves=best_num_leaves, class_weight='balanced', verbosity=-1)
    lgbm_clf.fit(X_new, y_new)

    # テストデータで予測
    y_pred = lgbm_clf.predict(X_new)
    cosuu = np.sum((y_new == 1) & (y_pred == 1))
    kazu = np.sum((y_new == 0) & (y_pred == 0))
    goukei = cosuu + kazu
    if goukei == len(y_pred):
        y_pred = lgbm_clf.predict(X_new_test)
        best_accurancy.append(accuracy_score(y_new_test, y_pred))
        best_recall.append(recall_score(y_new_test, y_pred))
        best_precision.append(precision_score(y_new_test, y_pred))
        best_f1_score.append(f1_score(y_new_test, y_pred))
        best_max_depths.append(best_max_depth)
        best_n_estimators.append(best_n_estimator)
        best_num_leaveses.append(best_num_leaves)
        for true, pred in zip(y_new_test, y_pred):
            if true == 1 and pred == 1:
                TP += 1
            elif true == 0 and pred == 0:
                TN += 1
            elif true == 0 and pred == 1:
                FP += 1
            elif true == 1 and pred == 0:
                FN += 1
    else:
        print("外側の検証で学習率が100%になりませんでした。")
        gaku_so = False
        break

if gaku_so:
    print("XGBoost ダブルクロスバリデーションでの結果")
    print(sum(best_accurancy) / len(best_accurancy))
    print(sum(best_recall) / len(best_recall))
    print(sum(best_precision) / len(best_precision))
    print(sum(best_f1_score) / len(best_f1_score))
    print(best_accurancy)
    print(best_recall)
    print(best_precision)
    print(best_f1_score)
    print(best_max_depths)
    print(best_n_estimators)
    print(best_num_leaveses)
    print(" ")

    # TP, TN, FP, FNの結果を表示
    print(f"True Positives (TP): {TP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    # 指定された評価指標を計算
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # 計算結果を表示
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(" ")
"""


# lightGBMでの実験（クロスバリデーションver）
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# X = np.delete(X.toarray(), 152, axis=0)
# labels.pop(152)
labels = np.array(labels)
# 外側の層化k分割交差検証（k=10）
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

max_depth = [9]
n_estimators = [50]
num_leaves = [21]

best_accurancy = []
best_recall = []
best_precision = []
best_f1_score = []
best_max_depths = []
best_n_estimators = []
best_num_leaveses = []
gaku_so = True

gakusyuu_100 = []

#wrong_data = []

# 実際のラベルと予測ラベルの比較用にTP, TN, FP, FNをカウント
TP = TN = FP = FN = 0

for train_index, test_index in outer_cv.split(X, labels):
    # print("外側の分割 - 学習データ:", train_index)
    best_score = 0
    best_max_depth = 0
    best_n_estimator = 0
    best_num_leaves = 0

    X_new = X[train_index]  # スパース行列 X から部分行列を取得
    y_new = labels[train_index]
    X_new_test = X[test_index]
    y_new_test = labels[test_index]

    for m in max_depth:
        for n in n_estimators:
            for nu in num_leaves:
                lgbm_clf = LGBMClassifier(max_depth=m, n_estimators=n,
                                          num_leaves=nu, class_weight='balanced', verbosity=-1, random_state=42)


                lgbm_clf.fit(X_new, y_new)

                # テストデータで予測
                y_pred = lgbm_clf.predict(X_new)
                cosuu = np.sum((y_new == 1) & (y_pred == 1))
                kazu = np.sum((y_new == 0) & (y_pred == 0))
                goukei = cosuu + kazu
                # print(k, g)
                # print(goukei)
                # print(len(y_pred))
                # print(y_pred)

                if goukei == len(y_pred):
                    s = lgbm_clf.score(X_new_test, y_new_test)

                    y_pred_gaku = lgbm_clf.predict(X_new_test)
                    gaku_100 = []
                    accuracy = accuracy_score(y_new_test, y_pred_gaku)
                    recall = recall_score(y_new_test, y_pred_gaku)
                    precision = precision_score(y_new_test, y_pred_gaku)
                    f1_scores = f1_score(y_new_test, y_pred_gaku)
                    gaku_100.append(m)
                    gaku_100.append(n)
                    gaku_100.append(nu)
                    gaku_100.append(accuracy)
                    gaku_100.append(recall)
                    gaku_100.append(precision)
                    gaku_100.append(f1_scores)
                    gakusyuu_100.append(gaku_100)
                    # 現在の分割で間違った予測を特定
                    wrong_indices = [test_index[i] for i, (true, pred) in enumerate(zip(y_new_test, y_pred_gaku)) if
                                     true != pred]

                    # インデックス番号を調整
                    adjusted_indices = [idx + 1 if idx >= 152 else idx for idx in wrong_indices]

                    # 全体のリストに追加
                    wrong_data.extend(adjusted_indices)

                    if best_score < s:
                        best_score = s
                        best_max_depth = m
                        best_n_estimator = n
                        best_num_leaves = nu

    if best_score == 0:
        gaku_so = False
        print("外の検証で学習率100%のものはありませんでした")
        break
    else:
        lgbm_clf = LGBMClassifier(max_depth=best_max_depth, n_estimators=best_n_estimator, num_leaves=best_num_leaves,
                                  class_weight='balanced', verbosity=-1, random_state=42)

        lgbm_clf.fit(X_new, y_new)
        y_pred = lgbm_clf.predict(X_new_test)
        accurancy = accuracy_score(y_new_test, y_pred)
        best_accurancy.append(accurancy)
        recall = recall_score(y_new_test, y_pred)
        best_recall.append(recall)
        precision = precision_score(y_new_test, y_pred)
        best_precision.append(precision)
        f1_scores = f1_score(y_new_test, y_pred)
        best_f1_score.append(f1_scores)
        best_max_depths.append(best_max_depth)
        best_n_estimators.append(best_n_estimator)
        best_num_leaveses.append(best_num_leaves)
        for true, pred in zip(y_new_test, y_pred):
            if true == 1 and pred == 1:
                TP += 1
            elif true == 0 and pred == 0:
                TN += 1
            elif true == 0 and pred == 1:
                FP += 1
            elif true == 1 and pred == 0:
                FN += 1

if gaku_so:
    print("lightGBMでクロスバリデーションの結果")
    wrong_data = sorted(wrong_data)
    print(len(wrong_data))
    print(wrong_data)
    """
"""
    print(sum(best_accurancy) / len(best_accurancy))
    print(sum(best_recall) / len(best_recall))
    print(sum(best_precision) / len(best_precision))
    print(sum(best_f1_score) / len(best_f1_score))
    print(best_accurancy)
    print(best_recall)
    print(best_precision)
    print(best_f1_score)
    print(best_max_depths)
    print(best_n_estimators)
    print(best_num_leaveses)
    print(" ")

    # 各リストの最初の2つの要素をキーにしてグループ化
    grouped_elements = defaultdict(list)
    for item in gakusyuu_100:
        key = tuple(item[:3])  # 最初の2つの要素をキーにする
        grouped_elements[key].append(item[3:])  # 後ろの要素をリストに追加

    best_average = 0.0
    best_averages = []  # 最良の平均値を持つリスト
    best_keys = []  # 最良のキーを持つリスト

    # 出現回数が2回以上のキーについて、後ろの要素の平均を計算して出力
    for key, values in grouped_elements.items():
        if len(values) > 9:
            # 各リストの後ろの要素の平均を計算
            averages = [sum(elements) / len(elements) for elements in zip(*values)]
            ab = averages[0]  # 最初の要素（平均）の値

            # 現在の平均値が最良よりも大きい場合
            if best_average < ab:
                best_average = ab
                best_averages = [averages]  # 最良の平均値を更新
                best_keys = [list(key)]  # 最良のキーを更新
            # 現在の平均値が最良と等しい場合
            elif best_average == ab:
                best_averages.append(averages)  # 最良の平均値リストに追加
                best_keys.append(list(key))  # 最良のキーリストに追加

    # best_averagesとbest_keysには同じ平均値を持つ複数の要素が格納されます

    print(best_keys)
    print(best_averages)

    # TP, TN, FP, FNの結果を表示
    print(f"True Positives (TP): {TP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    # 指定された評価指標を計算
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # 計算結果を表示
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(" ")
"""

#ロジスティック回帰
"""
#X = np.delete(X.toarray(), 152, axis=0)
#labels.pop(152)

labels = np.array(labels)

# 外側の層化k分割交差検証（k=10）
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


best_accurancy = []
best_recall=[]
best_precision=[]
best_f1_score=[]

gaku_so=True

# 実際のラベルと予測ラベルの比較用にTP, TN, FP, FNをカウント
TP = TN = FP = FN = 0

for train_index, test_index in outer_cv.split(X, labels):
    #print("外側の分割 - 学習データ:", train_index)
    X_new = X[train_index]  # スパース行列 X から部分行列を取得
    y_new = labels[train_index]
    X_new_test = X[test_index]
    y_new_test = labels[test_index]

    rogi_clf = LogisticRegression(class_weight='balanced', C=1e10)
    rogi_clf.fit(X_new, y_new)

    # テストデータで予測
    y_pred = rogi_clf.predict(X_new)
    cosuu = np.sum((y_new == 1) & (y_pred == 1))
    kazu = np.sum((y_new == 0) & (y_pred == 0))
    goukei = cosuu + kazu
    #print(y_pred)
    if goukei == len(y_pred):
        y_pred = rogi_clf.predict(X_new_test)
        best_accurancy.append(accuracy_score(y_new_test, y_pred))
        best_recall.append(recall_score(y_new_test, y_pred))
        best_precision.append(precision_score(y_new_test, y_pred))
        f=f1_score(y_new_test, y_pred)
        best_f1_score.append(f)
        for true, pred in zip(y_new_test, y_pred):
            if true == 1 and pred == 1:
                TP += 1
            elif true == 0 and pred == 0:
                TN += 1
            elif true == 0 and pred == 1:
                FP += 1
            elif true == 1 and pred == 0:
                FN += 1
    else:
        print("外側の検証で学習率が100%になりませんでした。")
        gaku_so = False
        break


if gaku_so:
    print("ロジスティック回帰の結果")
    print(sum(best_accurancy) / len(best_accurancy))
    print(sum(best_recall) / len(best_recall))
    print(sum(best_precision) / len(best_precision))
    print(sum(best_f1_score) / len(best_f1_score))
    print(best_accurancy)
    print(best_recall)
    print(best_precision)
    print(best_f1_score)
    
    # TP, TN, FP, FNの結果を表示
    print(f"True Positives (TP): {TP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    # 指定された評価指標を計算
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # 計算結果を表示
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(" ")
"""
#ロジスティック回帰(クロスバリデーションver)
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
labels = np.array(labels)
# 外側の層化k分割交差検証（k=10）
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


gamma = [0.001]
C = [996]

best_accurancy = []
best_recall = []
best_precision = []
best_f1_score = []
best_Cs = []
best_gs = []

gakusyuu_100 = []

gaku_so = True

#wrong_data = []

# 実際のラベルと予測ラベルの比較用にTP, TN, FP, FNをカウント
TP = TN = FP = FN = 0

for train_index, test_index in outer_cv.split(X, labels):
    #print("外側の分割 - 学習データ:", train_index)
    best_score = 0
    best_c = 0
    best_g = 0

    X_new = X[train_index]  # スパース行列 X から部分行列を取得
    y_new = labels[train_index]
    X_new_test = X[test_index]
    y_new_test = labels[test_index]

    for c in C:
        for g in gamma:

            rogi_clf = LogisticRegression(class_weight='balanced', C=c, random_state=42)
            rogi_clf.fit(X_new, y_new)

            # テストデータで予測
            y_pred = rogi_clf.predict(X_new)
            cosuu = np.sum((y_new == 1) & (y_pred == 1))
            kazu = np.sum((y_new == 0) & (y_pred == 0))
            goukei = cosuu + kazu
            #print(k, g)
            #print(goukei)
            #print(len(y_pred))
            #print(y_pred)

            if goukei == len(y_pred):
                y_pred_gaku = rogi_clf.predict(X_new_test)
                s = rogi_clf.score(X_new_test, y_new_test)
                gaku_100 = []
                accuracy = accuracy_score(y_new_test, y_pred_gaku)
                recall = recall_score(y_new_test, y_pred_gaku)
                precision = precision_score(y_new_test, y_pred_gaku)
                f1_scores = f1_score(y_new_test, y_pred_gaku)
                gaku_100.append(c)
                gaku_100.append(g)
                gaku_100.append(accuracy)
                gaku_100.append(recall)
                gaku_100.append(precision)
                gaku_100.append(f1_scores)
                gakusyuu_100.append(gaku_100)
                # 現在の分割で間違った予測を特定
                wrong_indices = [test_index[i] for i, (true, pred) in enumerate(zip(y_new_test, y_pred_gaku)) if
                                 true != pred]
                # インデックス番号を調整
                adjusted_indices = [idx + 1 if idx >= 152 else idx for idx in wrong_indices]


                # 全体のリストに追加
                wrong_data.extend(adjusted_indices)

                if best_score < s:
                    best_score = s
                    best_c = c
                    best_g = g


    if best_score == 0:
        gaku_so = False
        print("外の検証で学習率100%のものはありませんでした")
        break
    else :
        rogi_clf = LogisticRegression(class_weight='balanced', C=best_c, random_state=42)
        rogi_clf.fit(X_new, y_new)
        y_pred = rogi_clf.predict(X_new_test)
        accurancy = accuracy_score(y_new_test, y_pred)
        best_accurancy.append(accurancy)
        recall = recall_score(y_new_test, y_pred)
        best_recall.append(recall)
        precision = precision_score(y_new_test, y_pred)
        best_precision.append(precision)
        f1_scores = f1_score(y_new_test, y_pred)
        best_f1_score.append(f1_scores)
        best_Cs.append(best_c)
        best_gs.append(best_g)




        for true, pred in zip(y_new_test, y_pred):
            if true == 1 and pred == 1:
                TP += 1
            elif true == 0 and pred == 0:
                TN += 1
            elif true == 0 and pred == 1:
                FP += 1
            elif true == 1 and pred == 0:
                FN += 1



if gaku_so:
    print("ロジスティック回帰でクロスバリデーションの結果")
    wrong_data = sorted(wrong_data)
    print(len(wrong_data))
    print(wrong_data)

    # 出現回数をカウント
    counter = Counter(wrong_data)

    # 3回以上出現する値をリストに格納
    all_wrong_data = [key for key, count in counter.items() if count >= 6]

    print(all_wrong_data)
"""
"""
    print(sum(best_accurancy) / len(best_accurancy))
    print(sum(best_recall) / len(best_recall))
    print(sum(best_precision) / len(best_precision))
    print(sum(best_f1_score) / len(best_f1_score))
    print(best_accurancy)
    print(best_recall)
    print(best_precision)
    print(best_f1_score)
    print(best_gs)
    print(best_Cs)
    print(" ")

    # 各リストの最初の2つの要素をキーにしてグループ化
    grouped_elements = defaultdict(list)
    for item in gakusyuu_100:
        key = tuple(item[:2])  # 最初の2つの要素をキーにする
        grouped_elements[key].append(item[2:])  # 後ろの要素をリストに追加

    best_average = 0.0
    best_averages = []  # 最良の平均値を持つリスト
    best_keys = []  # 最良のキーを持つリスト

    # 出現回数が2回以上のキーについて、後ろの要素の平均を計算して出力
    for key, values in grouped_elements.items():
        if len(values) > 9:
            # 各リストの後ろの要素の平均を計算
            averages = [sum(elements) / len(elements) for elements in zip(*values)]
            ab = averages[0]  # 最初の要素（平均）の値

            # 現在の平均値が最良よりも大きい場合
            if best_average < ab:
                best_average = ab
                best_averages = [averages]  # 最良の平均値を更新
                best_keys = [list(key)]  # 最良のキーを更新
            # 現在の平均値が最良と等しい場合
            elif best_average == ab:
                best_averages.append(averages)  # 最良の平均値リストに追加
                best_keys.append(list(key))  # 最良のキーリストに追加

    # best_averagesとbest_keysには同じ平均値を持つ複数の要素が格納されます

    print(best_keys)
    print(best_averages)


    # TP, TN, FP, FNの結果を表示
    print(f"True Positives (TP): {TP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    # 指定された評価指標を計算
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # 計算結果を表示
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(" ")
"""





#特徴量の削除のためのプログラム
'''
# データとラベルの例
y = np.array(labels) # あなたのラベルセット (1が先で0が後)

# 訓練データ9割、検証データ1割に分割 (ラベル1と0を考慮)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
train_index, test_index = next(sss.split(X, y))

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

# ランダムフォレスト分類器の定義
clf = RandomForestClassifier(class_weight='balanced', random_state=42)

# ハイパーパラメータグリッド
param_grid = {
    'n_estimators': [30, 50, 100, 200, 300],
    'max_depth': [None, 2, 3, 5, 10, 30],
}

# 内側の層化k分割交差検証とグリッドサーチ（k=10）
timings = {}
timings['内側計算はじめ'] = time.time()
inner_cv = StratifiedKFold(n_splits=10, shuffle=True)
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=inner_cv, scoring='accuracy')

# 訓練データに対してグリッドサーチを適用
grid_search.fit(X_train, y_train)
timings['内側計算終わり'] = time.time()

# ベストパラメータでのモデル評価
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

timings['外側計算終わり'] = time.time()

# 最適化された決定木の本数を出力
optimized_n_estimators = grid_search.best_params_['n_estimators']

# 結果表示
print("Best parameters found: ", grid_search.best_params_)
print("Accuracy score: ", accuracy)
print("Timings: ", timings)

# 実際のラベルと予測結果を表示
print("実際のラベル: ", y_test)
print("予測結果のラベル: ", y_pred)

# 各決定木の可視化と解析 (False Negativeの検証)
tn_indices = np.where((y_test == 1) & (y_pred == 0))[0]  # 実際のラベルが1で予測が0であるインデックス（TN）

if len(tn_indices) == 0:
    print("False Negative（TN）が見つかりませんでした。")
else:
    print(f"False Negative（TN）の数: {len(tn_indices)}")

# 各検証データ (TN) ごとにまとめて決定木の結果を出力
for tn_index in tn_indices:
    print(f"検証データ {tn_index}（TN）の解析:")

    for i, tree in enumerate(best_model.estimators_):
        test_leaf_ids = tree.apply(X_test)  # 検証データがどのノードに入っているか
        train_leaf_ids = tree.apply(X_train)  # 学習データがどのノードに入っているか

        tn_leaf_id = test_leaf_ids[tn_index]  # TNデータが含まれるノード
        # このノードに含まれる学習データを確認
        samples_in_node = np.where(train_leaf_ids == tn_leaf_id)[0]
        labels_in_node = y_train[samples_in_node]
        label_1_count = np.sum(labels_in_node == 1)
        label_0_count = np.sum(labels_in_node == 0)

        # ノードでのクラス予測
        node_class = np.argmax(tree.tree_.value[tn_leaf_id][0])
        #print(f"  決定木 ,{i},ノード ,{tn_leaf_id},ラベル詐欺,{label_1_count}, ラベルno詐欺,{label_0_count},ラベル,{node_class}")
        print(f"{i},{tn_leaf_id},{label_1_count},{label_0_count},{node_class}")


# 最終的に学習データでどちらに分類されているか確認
print("全ての決定木におけるTNデータの解析が完了しました。")


# 保存先のディレクトリを作成（存在しない場合）
output_dir = "decision_tree_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 可視化するための決定木を選択（例として最初の木を選択）
for i in range(optimized_n_estimators):
    tree_to_visualize = best_model.estimators_[i]

    # 決定木の可視化
    plt.figure(figsize=(20, 10))
    plot_tree(tree_to_visualize, feature_names=None, filled=True, rounded=True, precision=2, node_ids=True)
    plt.title(f"Decision Tree Visualization - Tree {i}")
    # ファイル名を決定し、保存
    filename = os.path.join(output_dir, f"{i}.png")
    plt.savefig(filename)
    plt.close()  # メモリを節約するためにプロットを閉じる
'''


#学習率を確かめるためのランダムフォレスト
"""
# TF-IDFを適用して得られたX_trainがスパース行列の場合、これをNumPy配列に変換
X_train_dense = X.toarray()
y = np.array(labels)

# ランダムフォレスト分類器の定義
clf = RandomForestClassifier(class_weight='balanced', random_state=42)

# パラメータグリッドの設定
param_grid = {
    'n_estimators': [30, 50, 100, 200, 300],
    'max_depth': [None, 2, 3, 5, 10, 30]
}

# グリッドサーチの設定
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# グリッドサーチを用いて学習
grid_search.fit(X_train_dense, y)

# 最適なパラメータとその精度を出力
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_}")

# 学習データでの精度（学習率）を確認
train_accuracy = grid_search.score(X_train_dense, y)

# 学習データを使って予測
y_pred = grid_search.predict(X_train_dense)

# 結果表示
print(f"Training Accuracy: {train_accuracy}")
print("実際のラベル: ", y)
print("予測結果のラベル: ", y_pred)
"""

"""
# TF-IDFを適用して得られたX_trainがスパース行列の場合、これをNumPy配列に変換
X_train_dense = X.toarray()
y = np.array(labels)

# 全てのデータで学習データを用意
#X_train_ones = X_train_dense


#学習データについて詐欺データと詐欺ではないデータの割合を決める
#詐欺データと詐欺でないデータの割合が9.5:0.5 (148と8　156)
#nosagi = 8
#gokei = 156
#sei_nu = 0.0513
#詐欺データと詐欺でないデータの割合が9:1 (148と16　164)
#nosagi = 16
#gokei = 164
#sei_nu = 0.0976
#詐欺データと詐欺でないデータの割合が8.5:1.5(148と26 174)
nosagi = 26
gokei = 174
sei_nu = 0.149

# ランダムで使用
#X_train_ones = X_train_dense[:gokei]
#y = y[:len(X_train_ones)]

# ランダム以外で使用
X_train_ones = X_train_dense[:148]

#詐欺の可能性が最も低いデータ（詐欺のデータ重心からのユークリッド距離が一番遠いもの）の抽出
X_train_dense = X.toarray()
#最初から148個のデータは詐欺のデータである
X_sagi = X[:148]
#最初から148個より後のデータは詐欺ではないデータである
X_no_sagi = X[148:]

#詐欺データの重心を出す
center_sagi = np.mean(X_sagi, axis = 0)
center_sagi = np.asarray(center_sagi).reshape(1, -1)
#print(center_sagi)
#詐欺データの重心と各詐欺ではないデータとの距離を測る（ユークリッド距離で計算）
distances = pairwise_distances(X_no_sagi, center_sagi, metric = 'euclidean').flatten()
#print(distances)

#出た距離を長い順に並び替える
soreted_indices = np.argsort(-distances)
#print(soreted_indices)
#長い距離から何個データを学習データとして使用するかを決める
top_distance_indices = soreted_indices[:nosagi]
#print(top_distance_indices)
#全体データで考えるために、もともとも詐欺群の数148個を足し合わせる
original_indices = top_distance_indices + X_sagi.shape[0]
print(original_indices)


#SVMの識別超平面による最も詐欺ではないデータの抽出

X_train_dense = X.toarray()
y = np.array(labels)
# SVMモデルを作成（線形カーネルを使用）
clf = svm.SVC(kernel='linear', C=1e10)

# モデルを訓練データで学習
clf.fit(X_train_dense, y)

# テストデータで予測
y_pred = clf.predict(X_train_dense)
#print("実際")
#print(y)
#print("予測")
#print(y_pred)

#識別超平面からの距離を計算
decision_values = clf.decision_function(X_train_dense)
#print(decision_values)

# マイナスの値を持つインデックスを抽出（詐欺ではないデータ）
negative_indices = np.where(decision_values < 0)[0]  # 負の決定値を持つデータのインデックス

# 詐欺でない可能性が高いものから順に、つまり絶対値が大きい順にソート
sorted_negative_indices = negative_indices[np.argsort(decision_values[negative_indices])]
#print(sorted_negative_indices)
# 割合に応じて取得
original_indices = sorted_negative_indices[:nosagi]
print(original_indices)


#ランダムの際は消す結合プログラム

#学習データに、詐欺ではないデータを追加する。
for i in range(len(original_indices)):
    print(original_indices[i])
    X_train_ones = np.vstack([X_train_ones, X_train_dense[original_indices[i]]])


y = y[:len(X_train_ones)]
print("正しい学習データ数:", gokei, "現在の学習データ数", len(X_train_ones), "対応するラベルの長さ", len(y))
"""


#SVMで実行
"""
#グリットサーチの範囲を決める
#kernel =  ['linear', 'poly', 'rbf', 'sigmoid']
kernel = ['linear']
#gamma = ['scale', 'auto', 0.1, 0.01, 0.001, 0.005]  # linearの場合、gamma入らない
gamma = ['scale']
nu = [0.015, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.5, sei_nu]  # nuの候補
#nu = [0.149]
#一番よいパラメータを格納するものを決める
best_cosuu = 1e10

#ワンクラスSVMを実行
for i in kernel:
    for n in gamma:
        for m in nu:
            # ワンクラスサポートベクターマシンの定義
            #ocsvm = OneClassSVM(kernel=i, gamma=n, nu=m)  # gammaやnuの値は調整可能
            ocsvm = OneClassSVM(kernel=i, nu=m)

            # 学習データで学習
            ocsvm.fit(X_train_ones)

            # 全データに対して予測を行う（ラベルが1のデータ以外も含む）
            y_pred = ocsvm.predict(X_train_ones)
            cosuu = np.sum((y == 1) & (y_pred == -1))

            print(i,n,m,cosuu)
            print(y_pred)
            anomalies = np.where(y_pred == -1)[0]
            print(f"異常と判定されたデータのインデックス: ")
            print(anomalies)
            print(len(anomalies))
            print(" ")
            if best_cosuu>=cosuu:
                best_cosuu = cosuu
                best_kernel = i
                best_gamma = n
                best_nu = m

            

print("一番良いときの評価", best_cosuu)
print("一番良い時のカーネル", best_kernel)
print("一番良いときのガンマ", best_gamma)
print("一番良いときの異常率", best_nu)
"""

#k-nnで実行
"""
#グリットサーチの範囲を決める
k = [ 3, 5, 7, 9, 11,13, 15, 17, 19, 21, 23, 25]
#一番よいパラメータを格納するものを決める
best_cosuu = 1e10

# k-NNモデルを作成
for i in k:
    knn = KNeighborsClassifier(n_neighbors=i)
    # モデルを訓練データで学習
    knn.fit(X_train_ones, y)

    # テストデータで予測
    y_pred = knn.predict(X_train_ones)

    cosuu = np.sum((y == 1) & (y_pred == 0))
    print("k=",i)
    print("実際：詐欺、予測：詐欺ではない",cosuu)
    print(y_pred)
    anomalies = np.where(y_pred == 0)[0]
    print(f"異常と判定されたデータのインデックス: ")
    print(anomalies)
    print(len(anomalies))
    print(" ")
    if best_cosuu >= cosuu:
        best_cosuu = cosuu
        best_k = i

print(best_cosuu)
print(best_k)
"""


#LOFで実行
"""


#ハイパラメータの設定
n_neighbors = [1, 2, 3,4,5, 6,7,8,9,10, 15, 20, 25]
contamination = [0.015, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.5, sei_nu, 'auto']

#一番よいパラメータを格納するものを決める
best_cosuu = 1e10

#ワンクラスSVMを実行
for i in contamination:
    for n in n_neighbors:
        #lofの定義
        #lof = LocalOutlierFactor(n_neighbors= i, contamination = n, novelty=True)
        lof = LocalOutlierFactor(n_neighbors=n, contamination= i)

        #学習データで学習
        #lof.fit(X_train_ones)

        #学習データに対して予測を行う（ラベルが1のデータ以外も含む）
        #y_pred = lof.predict(X_train_ones)
        y_pred = lof.fit_predict(X_train_ones)

        cosuu = np.sum((y == 1) & (y_pred == -1))
        
        print("近傍数", n,  "異常割合", i,  "実際詐欺、予測詐欺ではない個数", cosuu)
        print(y_pred)
        anomalies = np.where(y_pred == -1)[0]
        print(f"異常と判定されたデータのインデックス: ")
        print(anomalies)
        print(len(anomalies))
        print(" ")
        if best_cosuu > cosuu:
            best_cosuu = cosuu
            best_n_neighbors = n
            best_contamination = i


print("最善の予測の個数", best_cosuu)
print("最善の予測の時の近傍数", best_n_neighbors, "最善の予測の時の異常割合", best_contamination)
"""

from sklearn.model_selection import train_test_split
# データを訓練用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, shuffle=True)
from sklearn.metrics import classification_report
# SVM分類器のハイパーパラメータグリッドの定義
param_grid = {
    'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],  # 正則化パラメータ
    'kernel': ['rbf'],  # カーネルタイプ
    'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000,'scale', 'auto']  # RBFカーネルに使用するパラメータ
}

# 最適なパラメータを手動で選択する
best_score = 0
best_params = None
best_model = None

# グリッドサーチの実行（クロスバリデーションなし）
for C in param_grid['C']:
    for kernel in param_grid['kernel']:
        for gamma in param_grid['gamma']:
            # SVM分類器の訓練
            clf = svm.SVC(C=C, kernel=kernel, gamma=gamma,class_weight='balanced')
            clf.fit(X_train, y_train)

            # モデルの評価（訓練データに対してスコアを算出）
            score = clf.score(X_train, y_train)

            # 最良スコアのモデルを保存
            if score > best_score:
                best_score = score
                best_params = {'C': C, 'kernel': kernel, 'gamma': gamma}
                best_model = clf

# 最適なパラメータを表示
print("Best Parameters:", best_params)

# 最適なモデルを使ってテストデータで予測
y_pred = best_model.predict(X_test)

# 評価結果の表示
print(classification_report(y_test, y_pred))

# 実際のラベル（y_test）と予測ラベル（y_pred）の表示
for true_label, predicted_label in zip(y_test, y_pred):
    print(f"実際のラベル: {true_label}, 予測ラベル: {predicted_label}")

# さらに、classification_reportを表示して詳細な評価を確認することもできます
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

# 最適なモデルを使ってテストデータで予測
y_pred = best_model.predict(X_train)

# 評価結果の表示
print(classification_report(y_train, y_pred))

# 実際のラベル（y_test）と予測ラベル（y_pred）の表示
for true_label, predicted_label in zip(y_train, y_pred):
    print(f"実際のラベル: {true_label}, 予測ラベル: {predicted_label}")

# さらに、classification_reportを表示して詳細な評価を確認することもできます
print("\nClassification Report:")
print(classification_report(y_train, y_pred, zero_division=1))
