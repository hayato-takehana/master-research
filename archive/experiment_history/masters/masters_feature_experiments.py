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

from pdf_text_loader import Text_road_and_dell
from text_vectorizer import Tf_idf
from tfidf_gap_term_selector import Sagi_word
from term_feature_selector import Find_terms
from nested_leave_one_out import Leave_one_out
import pandas as pd
import numpy as np
from sklearn.svm import SVC

#詐欺のテキスト読み込み
text_sagi = Text_road_and_dell("document_詐欺_実際の詐欺.pkl", "詐欺_実際の詐欺")
document_sagi = text_sagi.read_PDF()

#詐欺ではないテキストの読み込み
text_no_sagi = Text_road_and_dell("document_詐欺じゃない_実際の詐欺.pkl", "詐欺じゃない_実際の詐欺")
document_no_sagi = text_no_sagi.read_PDF()

#詐欺のと詐欺でないドキュメントの結合
documents = document_sagi + document_no_sagi


#tf_idfを行う
tf_idf = Tf_idf(documents, 'True', 3)
labels = tf_idf.labels(document_sagi)
tf_idf.preprocess()
X, feature_names, vectorizer = tf_idf.tf_idf(0.3)
X_freq, feature_names_freq, vectorizer = tf_idf.term_frequency(0.0)

df = pd.DataFrame(X.toarray(), columns=feature_names)
print(df)
#print(X)
#print(feature_names)
print(len(feature_names))

"""
#詐欺によく出てきて、詐欺じゃない文章にあまり出てこない単語
#詐欺じゃないによく出てきて詐欺に文章にあまり出てこない単語
offten_sagi = Sagi_word(X, labels, 2,vectorizer)
offten_sagi.word_output()
"""


X = X.toarray()
find_terms = Find_terms(X_freq, labels, feature_names_freq)
print("”全て”の詐欺の基準")
all_label1_least_label0 = find_terms.all_label1_least_label0(2)
not_in_label1_most_label0 = find_terms.not_label1_most_label0(2)
most_label0 = find_terms.most_label0(5)
#実際の出現頻度を確認
#a = find_terms.most_label1_least_label0(100)
#b = find_terms.least_label1_most_label0(100)

# バイナリー変数で特徴を抜き出す
df_binary_1 = find_terms.create_topn_binary_fueature(all_label1_least_label0)
df_binary_2 = find_terms.create_topn_binary_fueature(not_in_label1_most_label0)
df_combined_binary_1_2 = pd.concat([df_binary_1, df_binary_2], axis=1)  #一番詐欺の単語＋一番詐欺じゃない単語（バイナリー変数）
#出現回数で特徴を抜き出す
term_df = find_terms.extract_terms_features(all_label1_least_label0)
term_df_2 = find_terms.extract_terms_features(not_in_label1_most_label0)
df_combined_term_df_1_2 = pd.concat([term_df, term_df_2], axis=1) #一番詐欺の単語＋一番詐欺じゃない単語(出現回数)
#出現回数のあとに列で正規化
term_df_normalization = find_terms.extract_terms_features_normalization(all_label1_least_label0)
term_df_normalization_2 = find_terms.extract_terms_features_normalization(not_in_label1_most_label0)
df_combined_term_df_normalization_1_2 = pd.concat([term_df_normalization, term_df_normalization_2], axis=1)
#出現回数後に行で正規化
df_combined_term_df_row_normalization_1_2 = pd.concat([term_df, term_df_2], axis=1)
df_combined_term_df_row_normalization_1_2 = find_terms.extract_terms_features_row_normalization(df_combined_term_df_row_normalization_1_2)
print("バイナリー変数での特徴量")
print(df_binary_1)
print(df_binary_2)
print(df_combined_binary_1_2)
print("出現回数での特徴量")
print(term_df)
print(term_df_2)
print(df_combined_term_df_1_2)
print("出現回数の後に列で正規化した特徴量")
print(term_df_normalization)
print(term_df_normalization_2)
print(df_combined_term_df_normalization_1_2)
print("出現回数後に行で正則化した特徴量")
print(df_combined_term_df_row_normalization_1_2)
print("")


# CSVファイルとして保存（カレントディレクトリに保存されます）
df_combined_binary_1_2.to_csv("combined_binary_1_2.csv", index=False, encoding='utf-8-sig')
df_combined_term_df_row_normalization_1_2.to_csv("combined_term_df_normalization_1_2.csv", index=False, encoding='utf-8-sig')




import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# 1. PCAの実行（同じ）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df_combined_binary_1_2)

# 2. 可視化（ラベル付き）
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='Set1', alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('binary')
plt.legend(*scatter.legend_elements(), title="class")
plt.grid(True)
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(df_combined_term_df_row_normalization_1_2)

# 2. 可視化（ラベル付き）
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='Set1', alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('nomalization')
plt.legend(*scatter.legend_elements(), title="class")
plt.grid(True)
plt.show()

















print("それぞれの文書数を加味した割合の基準")
#別の詐欺の基準(それぞれの文書数を加味した割合の基準)
rate_label1_minus_label0 = find_terms.rate_label1_minus_label0(10)
rate_label0_minus_labes1 = find_terms.rate_label0_minus_label1(10)

# バイナリー変数で特徴を抜き出す
rate_binary_1 = find_terms.create_topn_binary_fueature(rate_label1_minus_label0)
rate_binary_2 = find_terms.create_topn_binary_fueature(rate_label0_minus_labes1)
rate_combined_binary_1_2 = pd.concat([rate_binary_1, rate_binary_2], axis=1)  #一番詐欺の単語＋一番詐欺じゃない単語（バイナリー変数）
#出現回数で特徴を抜き出す
rate_term_df = find_terms.extract_terms_features(rate_label1_minus_label0)
rate_term_df_2 = find_terms.extract_terms_features(rate_label0_minus_labes1)
rate_combined_term_df_1_2 = pd.concat([rate_term_df, rate_term_df_2], axis=1) #一番詐欺の単語＋一番詐欺じゃない単語(出現回数)
#出現回数のあとに列で正規化
rate_term_df_normalization = find_terms.extract_terms_features_normalization(rate_label1_minus_label0)
rate_term_df_normalization_2 = find_terms.extract_terms_features_normalization(rate_label0_minus_labes1)
rate_combined_term_df_normalization_1_2 = pd.concat([rate_term_df_normalization, rate_term_df_normalization_2], axis=1)
#出現回数のあとに行で正規化
rate_term_df = find_terms.extract_terms_features(rate_label1_minus_label0)
rate_term_df_2 = find_terms.extract_terms_features(rate_label0_minus_labes1)
rate_combined_term_df_row_normalization_1_2 = pd.concat([rate_term_df, rate_term_df_2], axis=1)
rate_combined_term_df_row_normalization_1_2 = find_terms.extract_terms_features_row_normalization(rate_combined_term_df_row_normalization_1_2)
#出力で確認
print("バイナリー変数での特徴量")
print(rate_binary_1)
print(rate_binary_2)
print(rate_combined_binary_1_2)
print("出現回数での特徴量")
print(rate_term_df)
print(rate_term_df_2)
print(rate_combined_term_df_1_2)
print("出現回数の後に正規化した特徴量")
print(rate_term_df_normalization)
print(rate_term_df_normalization_2)
print(rate_combined_term_df_normalization_1_2)





#使用するXを決める
#”全て”詐欺抽出方法
X = df_combined_binary_1_2.values # バイナリー
#X = df_combined_term_df_1_2.values #出現回数
#X = df_combined_term_df_normalization_1_2.values #出現回数を列で正規化
#X= df_combined_term_df_row_normalization_1_2.values #出現回数を行で正規化

"""
X = rate_combined_binary_1_2.values
#X = rate_combined_term_df_1_2.values
#X = rate_combined_term_df_normalization_1_2.values
"""

y = np.array(labels)

print("")
print("今回使用する特徴")
print(X)


#実行する
#SVM（linear）の実行
# モデル定義
svm_linear = SVC(kernel='linear')
# ハイパーパラメータ範囲
param_grid = {
    'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
    #'C': [1, 10],
}
leave_one = Leave_one_out(X,y,svm_linear, param_grid)
leave_one.leave_one_out_printout()

#SVM(rbf)の実行
# モデル定義
svm_rbf = SVC(kernel='rbf')

# ハイパーパラメータ範囲
param_grid = {
    'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
    'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, "scale", "auto"]
}
leave_one = Leave_one_out(X,y,svm_rbf, param_grid)
leave_one.leave_one_out_printout()

"""
#まとめて実験するために
find_terms = Find_terms(X, labels, feature_names)

for i in range(1,4):
    for j in range(1,4):

        least_used_words_in_labels0 = find_terms.find_terms_least_used_in_labels0(i)
        most_used_words_label0_not_alluse_in_labels1 = find_terms.find_terms_most_used_in_label0_not_in_label1(j)
        # 出力
        print("ラベル１に出てくる単語の中でラベル0の出現が一番少ない単語ランキング")
        for i, term_info in enumerate(least_used_words_in_labels0, 1):
            print(
                f"{i}位: 単語='{term_info['term']}', ラベル1出現数={term_info['label1_count']}, ラベル0出現数={term_info['label0_count']}")

        print("ラベル１に出てこない単語の中でラベル0の出現が一番多い単語ランキング")
        if most_used_words_label0_not_alluse_in_labels1:
            print(
                f"ラベル1で一度も出なかった単語のうち、ラベル0で多く出た上位{len(most_used_words_label0_not_alluse_in_labels1)}個を表示：")
            for i, res in enumerate(most_used_words_label0_not_alluse_in_labels1, 1):
                print(f"{i}位: 単語='{res['term']}', ラベル0での出現文章数: {res['label0_count']}")
        else:
            print("条件を満たす単語がありませんでした。")
            
        # 上位2つだけ使いたい！
        df_binary_1 = find_terms.create_topn_binary_fueature(least_used_words_in_labels0)
        df_binary_2 = find_terms.create_topn_binary_fueature(most_used_words_label0_not_alluse_in_labels1)
        df_combined = pd.concat([df_binary_1, df_binary_2], axis=1)
        
        X = df_combined.values
        y = np.array(labels)
        # モデル定義
        svm_linear = SVC(kernel='linear')
        # ハイパーパラメータ範囲
        param_grid = {
            'C': [100, 1000, 10000],
        }
        leave_one = Leave_one_out(X, y, svm_linear, param_grid)
        # 実行
        result = leave_one.double_leave_one_out()

        # 結果表示
        print("kernel: linear")
        print("外側のleave-one-outの結果")
        print(f"TN: {result['TN']}, FP: {result['FP']}, FN: {result['FN']}, TP: {result['TP']}")
        print(f"Recall（リコール）: {result['recall']:.3f}")
        print(f"F1スコア: {result['f1_score']:.3f}")
        print(f"外側のleave-one-outの学習データへの精度: {result['train_accuracies']}")
        print(f"外側のleave-one-outの学習データの精度はすべて100%か: {result['all_train_accuracies_100']}")

        print(f"内側で選ばれたハイパーパラメータ")
        print(result['best_params_each_fold'])
        print(len(result['best_params_each_fold']))
        print("内側のその時の精度")
        print(result['best_scores_each_fold'])
        print(len(result['best_scores_each_fold']))
        print()

        # モデル定義
        svm_rbf = SVC(kernel='rbf')

        # ハイパーパラメータ範囲
        param_grid = {
            'C': [100, 1000, 10000],
            'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, "scale", "auto"]
        }
        leave_one = Leave_one_out(X, y, svm_rbf, param_grid)
        # 実行
        result = leave_one.double_leave_one_out()

        # 結果表示
        print("kernel: rbf")
        print("外側のleave-one-outの結果")
        print(f"TN: {result['TN']}, FP: {result['FP']}, FN: {result['FN']}, TP: {result['TP']}")
        print(f"Recall（リコール）: {result['recall']:.3f}")
        print(f"F1スコア: {result['f1_score']:.3f}")
        print(f"外側のleave-one-outの学習データへの精度: {result['train_accuracies']}")
        print(f"外側のleave-one-outの学習データの精度はすべて100%か: {result['all_train_accuracies_100']}")

        print(f"内側で選ばれたハイパーパラメータ")
        print(result['best_params_each_fold'])
        print(len(result['best_params_each_fold']))
        print("内側のその時の精度")
        print(result['best_scores_each_fold'])
        print(len(result['best_scores_each_fold']))
        print()
        print()
"""


















from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, StratifiedShuffleSplit
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict, Counter
X = np.delete(X, 152, axis=0)
labels.pop(152)
labels = np.array(labels)
# 外側の層化k分割交差検証（k=10）
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

kernel = ['linear', 'rbf', 'sigmoid']
gamma = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
C = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

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
    #print(sum(best_accurancy) / len(best_accurancy))
    #print(sum(best_recall) / len(best_recall))
    #print(sum(best_precision) / len(best_precision))
    #print(sum(best_f1_score) / len(best_f1_score))
    #print(best_accurancy)
    #print(best_recall)
    #print(best_precision)
    #print(best_f1_score)
    #print(best_gs)
    #print(best_Cs)
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
    #print(f"True Positives (TP): {TP}")
    #print(f"True Negatives (TN): {TN}")
    #print(f"False Positives (FP): {FP}")
    #print(f"False Negatives (FN): {FN}")
    # 指定された評価指標を計算
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # 計算結果を表示
    #print(f"Accuracy: {accuracy:.4f}")
    #print(f"Recall: {recall:.4f}")
    #print(f"Precision: {precision:.4f}")
    #print(f"F1 Score: {f1_score:.4f}")
    #print(" ")
    wrong_data = sorted(wrong_data)
    #print(len(wrong_data))
    #print(wrong_data)






wrong_data = []
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
labels = np.array(labels)
# 外側の層化k分割交差検証（k=10）
outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
kernel = ['linear', 'rbf', 'sigmoid']
C = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
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
    #print(sum(best_accurancy) / len(best_accurancy))
    #print(sum(best_recall) / len(best_recall))
    #print(sum(best_precision) / len(best_precision))
    #print(sum(best_f1_score) / len(best_f1_score))
    #print(best_accurancy)
    #print(best_recall)
    #print(best_precision)
    #print(best_f1_score)
    #print(best_gs)
    #print(best_Cs)
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
    #print(f"True Positives (TP): {TP}")
    #print(f"True Negatives (TN): {TN}")
    #print(f"False Positives (FP): {FP}")
    #print(f"False Negatives (FN): {FN}")
    # 指定された評価指標を計算
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # 計算結果を表示
    #print(f"Accuracy: {accuracy:.4f}")
    #print(f"Recall: {recall:.4f}")
    #print(f"Precision: {precision:.4f}")
    #print(f"F1 Score: {f1_score:.4f}")
    #print(" ")
    wrong_data = sorted(wrong_data)
    #print(len(wrong_data))
    #print(wrong_data)








