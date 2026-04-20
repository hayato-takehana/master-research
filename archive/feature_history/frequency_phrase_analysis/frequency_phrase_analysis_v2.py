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
# TOP100 をDataFrame化
top_count_1_0 = _top_n(diff_count_1_0, feature_names, n=100, desc=True, colname="count_diff_1_minus_0")
top_doc_1_0   = _top_n(diff_doc_1_0,   feature_names, n=100, desc=True, colname="docfreq_diff_1_minus_0")
top_count_0_1 = _top_n(diff_count_0_1, feature_names, n=100, desc=True, colname="count_diff_0_minus_1")
top_doc_0_1   = _top_n(diff_doc_0_1,   feature_names, n=100, desc=True, colname="docfreq_diff_0_minus_1")

# 表示
print("\n=== 詐欺群の出現回数 − 対照群の出現回数（TOP100） ===")
print(top_count_1_0.to_string(index=False))

print("\n=== 詐欺群の出現文書数 − 対照群の出現文書数（TOP100） ===")
print(top_doc_1_0.to_string(index=False))

print("\n=== 対照群の出現回数 − 詐欺群の出現回数（TOP100） ===")
print(top_count_0_1.to_string(index=False))

print("\n=== 対照群の出現文書数 − 詐欺群の出現文書数（TOP100） ===")
print(top_doc_0_1.to_string(index=False))

# CSV保存（任意）
top_count_1_0.to_csv("top100_count_diff_1_minus_0.csv", index=False, encoding="utf-8-sig")
top_doc_1_0.to_csv("top100_docfreq_diff_1_minus_0.csv", index=False, encoding="utf-8-sig")
top_count_0_1.to_csv("top100_count_diff_0_minus_1.csv", index=False, encoding="utf-8-sig")
top_doc_0_1.to_csv("top100_docfreq_diff_0_minus_1.csv", index=False, encoding="utf-8-sig")

# ===== ここまで追記（修正版） =====
"""
# ===== ラベル別：各文書×指定語の出現回数を図示（文書を横軸） =====
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gensim.parsing.preprocessing import stem_text

# 可視化したい語（ステミング済み想定。重複は除去）
tokens_target = ["liveness"]
# 1. まず、全ての単語をステミングする
stemmed_list = [stem_text(t) for t in tokens_target]

# 2. 次に、ステミングされたリストの重複を、順序を保ったまま削除する
_seen = set()
tokens_target = [t for t in stemmed_list if not (t in _seen or _seen.add(t))]


# 語→列indexの辞書
vocab_index = {w: i for i, w in enumerate(feature_names)}

def _extract_counts_for_label(col_idx, mask):
    """列col_idx（語）の文書ごとの値を、指定ラベルの文書だけ抜き出して1次元ndarrayで返す"""
    col = M[:, col_idx]
    #print('hello__ここから処理が始まっています型を見てね')
    #print(col)
    sub = col[mask]
    #print('ここからsubの処理が始まっているよ')
    #print(sub)
    # 疎・密どちらでも1次元配列へ
    if hasattr(sub, "toarray"):
        return np.asarray(sub.toarray()).ravel()
    return np.asarray(sub).ravel()

def _doc_ids(mask, prefix):
    """そのラベル内での文書番号（1始まり）からなる表示用ラベル配列"""
    n = int(mask.sum())
    return [f"{prefix}{i}" for i in range(1, n+1)]

# 各ラベル用のx軸（文書ID）
x_label1 = _doc_ids(mask1, "L1_")
x_label0 = _doc_ids(mask0, "L0_")

# 集計結果CSVも出す（任意）
df_l1 = pd.DataFrame(index=x_label1)
df_l0 = pd.DataFrame(index=x_label0)

# ===== ラベル1（詐欺群）：各文書を横軸に、語ごとに1枚ずつ図示 =====
for tok in tokens_target:
    if tok not in vocab_index:
        vals = np.zeros(mask1.sum(), dtype=float)
    else:
        j = vocab_index[tok]
        vals = _extract_counts_for_label(j, mask1)
    # CSV用
    df_l1[tok] = vals
    print("valsについて")
    print(vals)

# ===== ラベル0（対照群）：各文書を横軸に、語ごとに1枚ずつ図示 =====
for tok in tokens_target:
    if tok not in vocab_index:
        vals = np.zeros(mask0.sum(), dtype=float)
    else:
        j = vocab_index[tok]
        vals = _extract_counts_for_label(j, mask0)
    # CSV用
    df_l0[tok] = vals


# 集計CSV（任意）
df_l1.to_csv("per_doc_counts_label1.csv", encoding="utf-8-sig")
df_l0.to_csv("per_doc_counts_label0.csv", encoding="utf-8-sig")
# ===== ここまで =====


# ===== 追加：全ドキュメントを1枚に色分け（L1=青, L0=赤）で図示 =====
import matplotlib.pyplot as plt

# 全ドキュメントの表示用ラベル（L1_* を先、続いて L0_*）
x_all = x_label1 + x_label0

for tok in tokens_target:
    # ラベル1とラベル0の値をそれぞれ取得（上の処理で既に計算済みのものを再利用）
    if tok in df_l1.columns:
        vals_l1 = df_l1[tok].to_numpy()
    else:
        vals_l1 = np.zeros(len(x_label1), dtype=float)

    if tok in df_l0.columns:
        vals_l0 = df_l0[tok].to_numpy()
    else:
        vals_l0 = np.zeros(len(x_label0), dtype=float)

    # 全ドキュメント順に結合（順序：L1群 → L0群）
    vals_all = np.concatenate([vals_l1, vals_l0])

    # ラベルに応じた色配列を作成（L1=青, L0=赤）
    colors = (["blue"] * len(vals_l1)) + (["red"] * len(vals_l0))

    # 図示：1語につき1枚、文書を横軸、色でラベルを区別
    plt.figure()
    plt.bar(range(len(vals_all)), vals_all, color=colors)
    plt.xticks(range(len(vals_all)), x_all, rotation=90)
    plt.title(f"All documents (colored by label): '{tok}'")
    plt.xlabel("Documents (L1 then L0)")
    plt.ylabel("Count")
    # 簡易凡例（色のみ）
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor="blue", label="Label 1"),
                      Patch(facecolor="red",  label="Label 0")]
    plt.legend(handles=legend_handles, loc="upper right", frameon=False)

    plt.tight_layout()
    plt.savefig(f"a_{tok}_all_docs_colored.png", dpi=150)
    plt.show()
# ===== 追加ここまで =====
# ===== 追加（全面差し替え）：全ドキュメント色分け図＋TOP10（L1/L0） =====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# tokens_target は既に上で定義済みの想定（重複除去済み）
# M, feature_names が未定義なら再構築（term_frequency → 失敗時 tf-idf）
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

def _extract_counts_for_rows(col_idx, row_index_array):
    """列 col_idx の値を、row_index_array で指定した行から取り出して1次元配列で返す"""
    sub = M[row_index_array, col_idx]
    if hasattr(sub, "toarray"):
        return np.asarray(sub.toarray()).ravel()
    return np.asarray(sub).ravel()

# ラベル別のカウント表も、この追加内で正しく再構築（TOP10用）
df_l1 = pd.DataFrame(index=x_label1)
df_l0 = pd.DataFrame(index=x_label0)

# ---- 全ドキュメント色分け図（各語1枚）＋ ラベル別カウント表の作成 ----
for tok in tokens_target:
    if tok not in vocab_index:
        vals_l1 = np.zeros(len(idx1), dtype=float)
        vals_l0 = np.zeros(len(idx0), dtype=float)
    else:
        j = vocab_index[tok]
        vals_l1 = _extract_counts_for_rows(j, idx1)
        vals_l0 = _extract_counts_for_rows(j, idx0)

    # カウント表に格納（TOP10抽出で使用）
    df_l1[tok] = vals_l1
    df_l0[tok] = vals_l0

    # 全ドキュメント色分け図（L1=青, L0=赤）
    vals_all = np.concatenate([vals_l1, vals_l0])


# --- 参考：必要ならラベル別の集計CSVも保存（既存と同名で上書き） ---
df_l1.to_csv("per_doc_counts_label1.csv", encoding="utf-8-sig")
df_l0.to_csv("per_doc_counts_label0.csv", encoding="utf-8-sig")

# ---- 指定語ごとの出現回数が多い文書TOP10（ラベル1/ラベル0） ----
# ---- 指定語ごとの出現回数が多い文書（回数>0）をリストアップ（ラベル1/ラベル0） ----
def _ensure_series(df, tok):
    if tok in df.columns:
        s = df[tok].copy()
    else:
        s = pd.Series(np.zeros(len(df), dtype=float), index=df.index, name=tok)
    return s.fillna(0.0)


all_top = []
for tok in tokens_target:
    # L1側 (Label 1)
    s1 = _ensure_series(df_l1, tok)
    s1_sorted = s1.sort_values(ascending=False)
    # 回数が0より大きいものだけを抽出
    top1 = s1_sorted[s1_sorted > 0].reset_index()

    top1.columns = ["doc_id", "count"]
    # ファイル名とPrint文言を修正
    csv_filename_l1 = f"positive_count_docs_label1_{tok}.csv"
    top1.to_csv(csv_filename_l1, index=False, encoding="utf-8-sig")
    print(f"\n=== Positive Count Docs (Label 1) for '{tok}' (Saved to {csv_filename_l1}) ===")
    print(top1.to_string(index=False))

    t1 = top1.copy()
    t1.insert(0, "label", 1)
    t1.insert(1, "token", tok)
    all_top.append(t1)

    # L0側 (Label 0)
    s0 = _ensure_series(df_l0, tok)
    s0_sorted = s0.sort_values(ascending=False)
    # 回数が0より大きいものだけを抽出
    top0 = s0_sorted[s0_sorted > 0].reset_index()

    top0.columns = ["doc_id", "count"]
    # ファイル名とPrint文言を修正
    csv_filename_l0 = f"positive_count_docs_label0_{tok}.csv"
    top0.to_csv(csv_filename_l0, index=False, encoding="utf-8-sig")
    print(f"\n=== Positive Count Docs (Label 0) for '{tok}' (Saved to {csv_filename_l0}) ===")
    print(top0.to_string(index=False))

    t0 = top0.copy()
    t0.insert(0, "label", 0)
    t0.insert(1, "token", tok)
    all_top.append(t0)

# まとめCSV
df_all_top = pd.concat(all_top, ignore_index=True)
# ファイル名を修正
csv_filename_all = "positive_count_docs_per_token_by_label_all.csv"
df_all_top.to_csv(csv_filename_all, index=False, encoding="utf-8-sig")
print(f"\n=== 集約版（全語×ラベル）を {csv_filename_all} に保存しました ===")
# ===== 修正ここまで =====

"""
# ===== ここから N-gram/フレーズ 分析 (上記とは別処理) =====
#
# ここでは、Tf_idf クラスやステミングは使用しません。
# 元の「生の」文書テキスト (documents) に、
# 指定した「フレーズ」が文字列として何回出現するかを
# 直接カウントします。
#
# =========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 1. 検索したいフレーズのリスト
phrases_target = ["beacon node", "block builder", "block proposer", "builder API specification", "execution payload", "plausible liveness", "validator client", "probabilistic liveness"]
# (例: "validator client", "attestation subnet" のように追加できます)

# 2. フレーズ用のカウントを格納するDataFrameを準備
# (x_label1, x_label0 は上の処理で作成済みの想定)
df_phrases_l1 = pd.DataFrame(index=x_label1)
df_phrases_l0 = pd.DataFrame(index=x_label0)

# 3. 生の文書リスト (documents) と ラベル (labels) を使ってカウント
for phrase in phrases_target:
    phrase_lower = phrase.lower()  # 検索フレーズを小文字化

    counts_l1 = []
    counts_l0 = []

    # documents と labels は 1:1 で対応している
    for i, doc_text in enumerate(documents):
        label = labels[i]

        # 文書テキストを小文字化し、フレーズの出現回数をカウント
        doc_lower = doc_text.lower()
        count = doc_lower.count(phrase_lower)

        if label == 1:
            counts_l1.append(count)
        else:
            counts_l0.append(count)

    # 結果をDataFrameに格納
    df_phrases_l1[phrase] = counts_l1
    df_phrases_l0[phrase] = counts_l0

print("\n=== フレーズ分析 (L1) ===")
print(df_phrases_l1.head())
print("\n=== フレーズ分析 (L0) ===")
print(df_phrases_l0.head())

# 4. (任意) フレーズカウントのCSVを保存
df_phrases_l1.to_csv("per_doc_counts_phrase_label1.csv", encoding="utf-8-sig")
df_phrases_l0.to_csv("per_doc_counts_phrase_label0.csv", encoding="utf-8-sig")

# ===== 5. フレーズ分析：全ドキュメントを色分け図示 =====
# (x_all は上の処理で作成済みの想定)

for phrase in phrases_target:
    # ラベル1とラベル0の値を取得
    vals_l1 = df_phrases_l1[phrase].to_numpy()
    vals_l0 = df_phrases_l0[phrase].to_numpy()

    # 全ドキュメント順に結合
    vals_all = np.concatenate([vals_l1, vals_l0])

    # ラベルに応じた色配列を作成（L1=青, L0=赤）
    colors = (["blue"] * len(vals_l1)) + (["red"] * len(vals_l0))

    plt.figure()
    plt.bar(range(len(vals_all)), vals_all, color=colors)
    plt.xticks(range(len(vals_all)), x_all, rotation=90)
    plt.title(f"All docs (Phrase): '{phrase}'")  # タイトル修正
    plt.xlabel("Documents (L1 then L0)")
    plt.ylabel("Phrase Count")  # Y軸ラベル修正

    legend_handles = [Patch(facecolor="blue", label="Label 1"),
                      Patch(facecolor="red", label="Label 0")]
    plt.legend(handles=legend_handles, loc="upper right", frameon=False)

    plt.tight_layout()
    # ファイル名を変更 (a_phrase_...)
    plt.savefig(f"a_phrase_{phrase.replace(' ', '_')}_all_docs_colored.png", dpi=150)
    plt.show()


# ===== 6. フレーズ分析：出現回数が多い文書（回数>0）をリストアップ =====

# (上の _ensure_series 関数を再利用)
def _ensure_series(df, tok):
    if tok in df.columns:
        s = df[tok].copy()
    else:
        s = pd.Series(np.zeros(len(df), dtype=float), index=df.index, name=tok)
    return s.fillna(0.0)


all_top_phrases = []
for phrase in phrases_target:
    # L1側 (Label 1)
    s1 = _ensure_series(df_phrases_l1, phrase)  # df_phrases_l1 を使用
    s1_sorted = s1.sort_values(ascending=False)
    top1 = s1_sorted[s1_sorted > 0].reset_index()

    top1.columns = ["doc_id", "count"]
    # ファイル名とPrint文言を修正 (phrase_)
    csv_filename_l1 = f"positive_count_docs_phrase_label1_{phrase.replace(' ', '_')}.csv"
    top1.to_csv(csv_filename_l1, index=False, encoding="utf-8-sig")
    print(f"\n=== Positive Count Docs (Phrase Label 1) for '{phrase}' (Saved to {csv_filename_l1}) ===")
    print(top1.to_string(index=False))

    t1 = top1.copy()
    t1.insert(0, "label", 1)
    t1.insert(1, "token", phrase)  # "token"列にフレーズを格納
    all_top_phrases.append(t1)

    # L0側 (Label 0)
    s0 = _ensure_series(df_phrases_l0, phrase)  # df_phrases_l0 を使用
    s0_sorted = s0.sort_values(ascending=False)
    top0 = s0_sorted[s0_sorted > 0].reset_index()

    top0.columns = ["doc_id", "count"]
    # ファイル名とPrint文言を修正 (phrase_)
    csv_filename_l0 = f"positive_count_docs_phrase_label0_{phrase.replace(' ', '_')}.csv"
    top0.to_csv(csv_filename_l0, index=False, encoding="utf-8-sig")
    print(f"\n=== Positive Count Docs (Phrase Label 0) for '{phrase}' (Saved to {csv_filename_l0}) ===")
    print(top0.to_string(index=False))

    t0 = top0.copy()
    t0.insert(0, "label", 0)
    t0.insert(1, "token", phrase)  # "token"列にフレーズを格納
    all_top_phrases.append(t0)

# まとめCSV
df_all_top_phrases = pd.concat(all_top_phrases, ignore_index=True)
# ファイル名を修正
csv_filename_all = "positive_count_docs_per_phrase_by_label_all.csv"
df_all_top_phrases.to_csv(csv_filename_all, index=False, encoding="utf-8-sig")
print(f"\n=== フレーズの集約版（全語×ラベル）を {csv_filename_all} に保存しました ===")
# ===== フレーズ分析ここまで =====
"""
# ===== ここから：単語ごとのN回超過文書割合スコア (修正版：スコア = 2-1) =====
import pandas as pd
import numpy as np

# ===== ここから：単語ごとのN回超過文書割合スコア (修正版：スコア = 2-1, ソート付き) =====
import pandas as pd
import numpy as np

# 1. 入力：{"単語": n} の辞書
# ※辞書のキー（単語）は、M行列の語彙（feature_names）に
#   含まれているステミング済みの単語を指定してください。
target_word_thresholds = {
    "committe": 2,
    "staker": 0,
    "relai": 2,
    "valid": 9,
    "block": 10,
    "node": 10,
    "tree": 0,
    "asic": 0,
    "assert": 0,
    "checkpoint": 0,
    "dag": 2,
    "dex": 0,
    "defi": 1,
    "epoch": 14,
    "fork": 7,
    "ga": 9,
    "index": 4,
    "nonc": 5,
    "rlp": 0,
    "sidechain": 3
}

# 2. 計算結果を格納するリスト
results_list = []

# 3. ラベル1とラベル0の総文書数をあらかじめ計算
total_docs_l1 = np.sum(mask1)
total_docs_l0 = np.sum(mask0)

print(f"\n=== N回超過 文書割合スコアの計算 (L0-L1) (L1 Docs: {total_docs_l1}, L0 Docs: {total_docs_l0}) ===")

# 4. 各単語について計算ループ
for word, n in target_word_thresholds.items():

    # 語彙に存在するかチェック
    if word not in vocab_index:
        print(f"警告: '{word}' は語彙(vocab_index)に存在しません。スキップします。")
        results_list.append({
            "word": word,
            "n (threshold)": n,
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

    # --- ラベル1の計算 ---
    counts_l1 = _extract_counts_for_label(j, mask1)
    count_gt_n_l1 = np.sum(counts_l1 > n)
    if total_docs_l1 > 0:
        ratio_l1 = count_gt_n_l1 / total_docs_l1
    else:
        ratio_l1 = 0.0

    # --- ラベル0の計算 ---
    counts_l0 = _extract_counts_for_label(j, mask0)
    count_gt_n_l0 = np.sum(counts_l0 > n)
    if total_docs_l0 > 0:
        ratio_l0 = count_gt_n_l0 / total_docs_l0
    else:
        ratio_l0 = 0.0

    # --- スコア計算 (2 - 1 に変更) ---
    score = ratio_l0 - ratio_l1

    # 結果を保存 (列名も実態に合わせる)
    results_list.append({
        "word": word,
        "n (threshold)": n,
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
df_scores = df_scores.round(4)  # 小数点以下4桁で丸める

# ▼▼▼▼▼ 修正箇所 ▼▼▼▼▼
# スコア (score (L0-L1)) が高い順に並べ替え
df_scores = df_scores.sort_values(by="score (L0-L1)", ascending=False)
# ▲▲▲▲▲ 修正ここまで ▲▲▲▲▲

print(df_scores.to_string(index=False))

# (任意) CSVに保存
df_scores.to_csv("word_n_count_ratio_scores_L0_minus_L1_sorted.csv", index=False, encoding="utf-8-sig")

# ===== スコア計算ここまで =====
