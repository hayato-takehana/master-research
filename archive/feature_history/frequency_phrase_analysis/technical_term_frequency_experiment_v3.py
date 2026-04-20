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
from text_vectorizer import Tf_idf # (ステミング削除版のTf_idfクラスを読み込む)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, StratifiedShuffleSplit
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


text_sagi = Text_road_and_dell("document_詐欺_先行研究.pkl", "詐欺_先行研究")
document_sagi = text_sagi.read_PDF()

# 詐欺ではないテキストの読み込み
text_no_sagi = Text_road_and_dell("document_詐欺じゃない_先行研究.pkl", "詐欺じゃない_先行研究")
document_no_sagi = text_no_sagi.read_PDF()

documents= document_sagi+document_no_sagi

count= Tf_idf(documents, True, 3)
labels = count.labels(document_sagi)


import numpy as np
import pandas as pd
from scipy import sparse

# ===== ここから追記（修正版） =====

# 【重要】この追記内でベクトル化を実施します
# まずは Term Frequency（実質カウント行列）を試し、無ければ TF-IDF を使う
# min_df は必要に応じて変更してください
_MIN_DF = 1
def _build_matrix_and_features(count, min_df=_MIN_DF):
    """
    count: Tf_idf インスタンス
    戻り値: (M, feature_names)
      M: 文書×語の疎行列（float or int）
      feature_names: 語彙リスト（列順）
    """
    # 可能なら生カウント（Bag-of-Words相当）で計算
    #countはインスタンス（tf-idfをクラスで呼び出すときの名称）
    #hasattrは属性を持っているかの確認
    #つまりオブジェクト（インスタンス）countがterm_frequencyという属性（関数や変数）を持っているのかどうかを確認する
    #callableは呼び出し可能かどうかつまり関数として使用できるかどうかを確認するための
    if hasattr(count, "term_frequency") and callable(count.term_frequency):
        try:
            #term_frequencyを実行し、それを返す
            M, feature_names, _ = count.term_frequency(min_df=min_df)
            return M, feature_names
        except Exception:
            #Exceptionはどんなエラーでもという意味
            #passは何もしないで次に進む。
            pass
    # フォールバック：TF-IDF
    M, feature_names, _ = count.tf_idf(min_df=min_df)
    return M, feature_names

# 3) スパース行列の安全な合計・文書頻度計算
def _sum_by_class(M, mask):
    #maskは対象となるラベルをtrueとしたもの→詐欺ラベルの合計を確認したいとき、詐欺:true, 詐欺じゃない：falseとなる。
    #Mは文書と単語の数が記述されているもの：(0, 14553) 13.0のようなものが沢山記述。(文書番号, 単語番号)　出てくる単語回数　

    #if分で入力の型を判別。スパース（疎）行列かそれ以外numpy行列では適切な処理の仕方が違うため
    #スパース行列とはほとんど値が0の行列。言語系では０ばっかりになるので、値がある所だけ保存することで処理速度を早くする役割
    if sparse.issparse(M):
        #Trueの行列の身を抜き出す
        sub = M[mask]
        term_sum = np.asarray(sub.sum(axis=0)).ravel()           # 各語の総和（回数相当）
        doc_freq = np.asarray((sub > 0).sum(axis=0)).ravel()     # 各語の出現文書数
    else:
        sub = M[mask]
        term_sum = sub.sum(axis=0)
        doc_freq = (sub > 0).sum(axis=0)
        term_sum = np.asarray(term_sum).ravel()
        doc_freq = np.asarray(doc_freq).ravel()
    return term_sum, doc_freq

# 4) 上位N語を取り出し、DataFrame化
def _top_n(diff_vec, feature_names, n=100, desc=True, colname="diff"):
    order = np.argsort(diff_vec)
    if desc:
        order = order[::-1]
    order = [i for i in order if not np.isnan(diff_vec[i])]
    top_idx = order[:n]
    df = pd.DataFrame({
        "rank": np.arange(1, len(top_idx)+1),
        "token": [feature_names[i] for i in top_idx],
        colname: diff_vec[top_idx]
    })
    return df
# ===== 集計処理 =====
# ※ ここで「内部で」行列化してから差分計算します（X は使いません）
M, feature_names = _build_matrix_and_features(count, min_df=_MIN_DF)

y = np.array(labels).astype(int)   # 1=詐欺群, 0=対照群

mask1 = (y == 1)
mask0 = (y == 0)
print(mask1)
print(mask0)

sum1, df1 = _sum_by_class(M, mask1)  # 出現回数・出現文書数（詐欺群）
sum0, df0 = _sum_by_class(M, mask0)  # 出現回数・出現文書数（対照群）

# 差分ベクトル
diff_count_1_0 = sum1 - sum0            # 詐欺群の出現回数 − 対照群の出現回数
diff_doc_1_0   = df1  - df0             # 詐欺群の出現文書数 − 対照群の出現文書数
diff_count_0_1 = sum0 - sum1            # 対照群の出現回数 − 詐欺群の出現回数
diff_doc_0_1   = df0  - df1             # 対照群の出現文書数 − 詐欺群の出現文書数

"""
... (ここは変更なし) ...
# ===== ここまで追記（修正版） =====
"""

# ===== 【ここから変更】=====
# 必要なライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
# from gensim.parsing.preprocessing import stem_text # <--- ステミングしないので不要

# 可視化したい語（ステミング「しない」単語をそのまま指定）
tokens_target = ["block", "root", "consensus", "message", "hash", "node"]
# 念のため重複を除去（順序は保持）
_seen = set()
tokens_target = [t for t in tokens_target if not (t in _seen or _seen.add(t))]


# M, feature_names が未定義なら再構築（既にあるはずだが念のため）
try:
    M
    feature_names
except NameError:
    M, feature_names = _build_matrix_and_features(count, min_df=_MIN_DF)

# ラベル配列と行インデックス（整数）
y = np.array(labels).astype(int)
mask1 = (y == 1)
mask0 = (y == 0)
idx1 = np.where(mask1)[0]
idx0 = np.where(mask0)[0]

# 文書ID文字列（あなたの既存命名と揃える）
x_label1 = [f"L1_{i}" for i in range(1, len(idx1)+1)]
x_label0 = [f"L0_{i}" for i in range(1, len(idx0)+1)]
x_all = x_label1 + x_label0

# 語彙→列index
vocab_index = {w: i for i, w in enumerate(feature_names)}
# print(vocab_index['aa']) # <--- 'aa'が語彙にないとエラーになるためコメントアウト

def _extract_counts_for_rows(col_idx, row_index_array):
    """列 col_idx の値を、row_index_array で指定した行から取り出して1次元配列で返す"""
    sub = M[row_index_array, col_idx]
    if hasattr(sub, "toarray"):
        return np.asarray(sub.toarray()).ravel()
    return np.asarray(sub).ravel()

# ---- 全ドキュメント色分け図（各語1枚） ----

for tok in tokens_target:
    if tok not in vocab_index:
        print(f"--- 警告: '{tok}' は（ステミング無しの）語彙に存在しません。---")
        vals_l1 = np.zeros(len(idx1), dtype=float)
        vals_l0 = np.zeros(len(idx0), dtype=float)
    else:
        j = vocab_index[tok]
        vals_l1 = _extract_counts_for_rows(j, idx1)
        vals_l0 = _extract_counts_for_rows(j, idx0)

    # 全ドキュメント色分け図（L1=青, L0=赤）
    vals_all = np.concatenate([vals_l1, vals_l0])
    colors = (["blue"] * len(vals_l1)) + (["red"] * len(vals_l0))

    # 図示：1語につき1枚、文書を横軸、色でラベルを区別

    plt.figure()
    plt.bar(range(len(vals_all)), vals_all, color=colors)
    plt.xticks(range(len(vals_all)), x_all, rotation=90)
    plt.title(f"All documents (colored by label): '{tok}'")
    plt.xlabel("Documents (L1 then L0)")
    plt.ylabel("Count")
    # 簡易凡例（色のみ）
    legend_handles = [Patch(facecolor="blue", label="Label 1"),
                      Patch(facecolor="red",  label="Label 0")]
    plt.legend(handles=legend_handles, loc="upper right", frameon=False)

    plt.tight_layout()
    plt.savefig(f"{tok}_all_docs_colored.png", dpi=150)
    plt.show()

# [変更]
# TOP10のCSV出力や、それに関連するdf_l1, df_l0のCSV出力処理は
# ご要望に基づき、すべて削除しました。
# ===== 修正ここまで =====


# (ご提示いただいた可視化コードの最後の plt.show() の後から)
# ===== ここから続き =====

# スコア計算（90パーセンタイル版）に必要なヘルパー関数を定義します。
# 可視化コードで定義された _extract_counts_for_rows を利用します。

def _extract_counts_for_label(col_idx, mask):
    """
    指定された列 (col_idx) について、
    マスク (mask) がTrueの行の値を1次元配列で返す。
    （_extract_counts_for_rows と np.where を使って実装）
    """
    # mask (ブール配列) から行インデックスの配列に変換
    row_index_array = np.where(mask)[0]

    # _extract_counts_for_rows は上記の可視化コードブロックで
    # 定義されている前提
    return _extract_counts_for_rows(col_idx, row_index_array)


# 1. 入力：分析対象とする単語のリスト

# 辞書から単語リストを取得
target_words = tokens_target

# 2. 計算結果を格納するリスト
results_list = []

# 3. ラベル1とラベル0の総文書数をあらかじめ計算
# (mask1, mask0 は可視化コードの時点で定義済み)
total_docs_l1 = np.sum(mask1)
total_docs_l0 = np.sum(mask0)

print(f"\n=== L1 90パーセンタイル閾値 スコア計算 (L0-L1) (L1 Docs: {total_docs_l1}, L0 Docs: {total_docs_l0}) ===")

# 4. 各単語について計算ループ
for word in target_words:

    # 語彙に存在するかチェック (vocab_index も可視化コードで定義済み)
    if word not in vocab_index:
        print(f"警告: '{word}' は語彙(vocab_index)に存在しません。スキップします。")
        results_list.append({
            "word": word,
            "n (threshold)": np.nan,  # 閾値が計算できなかった
            "ratio_L0 (2)": 0.0,
            "ratio_L1 (1)": 0.0,
            "score (L0-L1)": 0.0,
            "docs_gt_n_L0": 0,
            "docs_gt_n_L1": 0,
            "error": "Not in vocab"
        })
        continue

    # 該当する単語の列インデックス
    j = vocab_index[word]

    # --- ラベル1のカウント取得（閾値決定のため先に実行） ---
    # (上で定義したヘルパー関数を使用)
    counts_l1 = _extract_counts_for_label(j, mask1)

    # --- 閾値n を L1の90パーセンタイル値として自動計算 ---
    if total_docs_l1 > 0 and len(counts_l1) > 0:
        n = np.percentile(counts_l1, 90)
    else:
        n = 0.0  # L1文書がない場合は閾値0とする

    # --- ラベル1の計算（自動計算した n を使用） ---
    count_gt_n_l1 = np.sum(counts_l1 > n)
    if total_docs_l1 > 0:
        ratio_l1 = count_gt_n_l1 / total_docs_l1
    else:
        ratio_l1 = 0.0

    # --- ラベル0の計算（自動計算した n を使用） ---
    counts_l0 = _extract_counts_for_label(j, mask0)
    count_gt_n_l0 = np.sum(counts_l0 > n)
    if total_docs_l0 > 0:
        ratio_l0 = count_gt_n_l0 / total_docs_l0
    else:
        ratio_l0 = 0.0

    # --- スコア計算 (L0 - L1) ---
    score = ratio_l0 - ratio_l1

    # 結果を保存 (n (threshold) には自動計算した n を格納)
    results_list.append({
        "word": word,
        "n (threshold)": n,  # 自動計算された閾値
        "ratio_L0 (2)": ratio_l0,
        "ratio_L1 (1)": ratio_l1,
        "score (L0-L1)": score,
        "docs_gt_n_L0": count_gt_n_l0,
        "docs_gt_n_L1": count_gt_n_l1,
        "error": None
    })

# 5. 結果をDataFrameにまとめて整形
df_scores = pd.DataFrame(results_list)

# 表示順を 2 (L0), 1 (L1), スコア の順に変更
df_scores = df_scores[[
    "word", "n (threshold)",
    "ratio_L0 (2)", "ratio_L1 (1)", "score (L0-L1)",
    "docs_gt_n_L0", "docs_gt_n_L1", "error"
]]

# 小数点以下4桁で丸める (対象列を明示)
round_cols = ["n (threshold)", "ratio_L0 (2)", "ratio_L1 (1)", "score (L0-L1)"]
df_scores[round_cols] = df_scores[round_cols].round(4)

# スコア (score (L0-L1)) が高い順に並べ替え
df_scores = df_scores.sort_values(by="score (L0-L1)", ascending=False)

print(df_scores.to_string(index=False))

# (任意) CSVに保存
df_scores.to_csv("word_n_count_ratio_scores_L1_90p_threshold_sorted.csv", index=False, encoding="utf-8-sig")

# 1. まず全単語の行列を X に一旦入れる
if hasattr(M, "toarray"):
    X_all = M.toarray()
else:
    X_all = M

# 2. 語彙（feature_names）と列番号の対応表を作る
vocab_index = {w: i for i, w in enumerate(feature_names)}

# 3. tokens_target にある単語の列番号（インデックス）だけをリストアップ
target_indices = []
found_words = []

print("\n--- 特徴量抽出の確認 ---")
for word in tokens_target:
    if word in vocab_index:
        idx = vocab_index[word]
        target_indices.append(idx)
        found_words.append(word)
    else:
        # ステミング等で形が変わっている、または出現回数が少なすぎて消えている場合
        print(f"除外（語彙に無し）: {word}")

# 4. 指定した列だけを抜き出して、新しい X とする
# X_all[:, target_indices] は「全行」かつ「指定した列」のみを抜き出す記述です
X = X_all[:, target_indices]
print(X)
print(f"\n抽出された単語数: {len(found_words)}")
print(f"修正後のデータ形状 X: {X.shape}")
# ↑ ここで (文書数, 指定単語数) になっていることを確認してください（例: (153, 45) など）

# ==========================================
# データ削除処理 (元のコードの続き)
# ==========================================

# 2. labels がリストなら numpy配列に変換（コピーしておく）
if isinstance(labels, list):
    y_work = np.array(labels)
else:
    y_work = labels.copy()

# 3. 152番目のデータを削除する処理
if X.shape[0] > 152:
    print(f"削除前: X={X.shape}, labels={y_work.shape}")

    X = np.delete(X, 152, axis=0)
    y_work = np.delete(y_work, 152, axis=0)

    print(f"削除後: X={X.shape}, labels={y_work.shape}")
    print("インデックス 152 のデータを削除しました。")

labels = y_work


# 内側の層化k分割交差検証とグリッドサーチ（k=10）
inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# 外側の層化k分割交差検証（k=10）
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

C=[1000]

best_accurancy = []
best_recall=[]
best_precision=[]
best_f1_score=[]
best_cs = []
gaku_so=True

# 新しい閾値を設定
custom_threshold = [0.0]

for custom in custom_threshold:
    print("はじめ")
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
            print("動いてる？")

            for inner_train_index, inner_test_index in inner_cv.split(X[train_index], labels[train_index]):
                print("ここで確認しようか")

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


# 内側の層化k分割交差検証とグリッドサーチ（k=10）
inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# 外側の層化k分割交差検証（k=10）
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

C=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
Gamma=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]

custom_threshold = [0.0]
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
            print("確認")
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



#先行研究の実験環境
param_grid = {
    'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],
    'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
}

# ランダムフォレスト分類器の定義
svm_clf_out = svm.SVC(kernel='rbf', class_weight='balanced',tol=1e-10)



# 内側の層化k分割交差検証とグリッドサーチ（k=10）
inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=svm_clf_out, param_grid=param_grid, cv=inner_cv, scoring='accuracy')


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
