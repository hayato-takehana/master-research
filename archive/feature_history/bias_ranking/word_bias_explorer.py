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

from text_vectorizer import Tf_idf
import numpy as np
from scipy import sparse
import pandas as pd
from gensim.parsing.preprocessing import stem_text
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ------------------------------------------------------------------
# 単語の出現頻度や文書数を分析し、詐欺群と対照群の差異を可視化・抽出するクラス
# TF-IDFクラスと連携して行列を作成し、統計的な比較を行います。
# ------------------------------------------------------------------
class word_chose:

    # ------------------------------------------------------------------
    # コンストラクタ
    # 分析に必要なデータ（文書リスト、ラベルリスト）を受け取り、初期設定を行います。
    # ------------------------------------------------------------------
    # [引数]
    # documents : 分析対象の文書リスト（文字列のリスト）
    # labels    : 各文書の正解ラベルのリスト（1:詐欺, 0:対照群）
    def __init__(self, documents, labels):
    # .count は text_vectorizer.py のクラスを呼び出すために保持している。
        # _MIN_DFは出現た割合の低い単語を落とすための変数、0.2=20%以下である、整数である場合、その回数以下を落とすという意味。今回は1にすることで単語を1つも落とさないようにしている
        # ｙはラベル（詐欺：１、詐欺じゃない：0）を扱いやすいようにしたもの
        # Mは文書と単語の数が記述されているもの：(0, 14553) 13.0のようなものが沢山記述。(文書番号, 単語番号)　出てくる単語回数　
        # feature_namesは各単語の名称を記述している[14553]とすることでその単語番号がどのような単語であるかを判別することができる
        # mask1, mask0は対象となるクラス(1:詐欺、0：詐欺じゃない)に対してその文書をtrueとするもの
        # nは出現回数、出現文書数を見るときに、上位何単語を抜き出すかを決める変数TOP○○を決める者
        # vocab_indexは単語名と番号をより検索しやすくまとめたもの。{'aa':0, 'human':10,}のようにすることでその単語番号をわかりやすく記述できる.vocab_index['human']とすることでその単語名の単語番号を教えてくれる。

        # 外部クラス Tf_idf のインスタンスを作成
        # documentsを渡し、deacc=True(アクセント記号除去), min_len=3(3文字以下の単語削除) で初期化
        self.count = Tf_idf(documents, True, 3)

        self._MIN_DF = 1  # 最低出現数の閾値（1なので足切りなし）

        # ラベルを整数型(int)のnumpy配列に変換（後の計算やマスク処理のため）
        self.y = np.array(labels).astype(int)

        # 行列構築メソッドを実行し、文書単語行列(M)と単語リスト(feature_names)を取得
        self.M, self.feature_names = self._build_matrix_and_features()

        # ラベルごとのブールマスク（True/False配列）を作成
        # これを使うことで、特定のラベルに該当する行だけを簡単に抽出できる
        self.mask1 = (self.y == 1)  # 詐欺データのインデックスがTrue
        self.mask0 = (self.y == 0)  # 対照データのインデックスがTrue

        self.n = 100  # ランキングで表示する上位単語数
        self.desc = True  # ソート順（Trueなら降順＝大きい順）

        # 単語名から列番号（インデックス）を高速に検索するための辞書を作成
        self.vocab_index = {w: i for i, w in enumerate(self.feature_names)}

    # ------------------------------------------------------------------
    # 行列構築メソッド
    # 外部クラスTf_idfの機能を使って、文書×単語の行列を作成します。
    # ------------------------------------------------------------------
    # [引数]
    # min_df : 最小出現頻度（指定がなければクラス変数の_MIN_DFを使用）
    # [戻り値]
    # M             : 文書×単語の行列（値は出現回数またはTF-IDF値）
    # feature_names : 列に対応する単語名のリスト
    def _build_matrix_and_features(self, min_df=None):
        """
        count: Tf_idf インスタンス
        戻り値: (M, feature_names)
          M: 文書×語の疎行列（float or int）
          feature_names: 語彙リスト（列順）
        """
        if min_df is None:
            min_df = self._MIN_DF
        # 可能なら生カウント（Bag-of-Words相当）で計算
        # countはインスタンス（tf-idfをクラスで呼び出すときの名称）
        # hasattrは属性を持っているかの確認
        # つまりオブジェクト（インスタンス）countがterm_frequencyという属性（関数や変数）を持っているのかどうかを確認する
        # callableは呼び出し可能かどうかつまり関数として使用できるかどうかを確認するための
        if hasattr(self.count, "term_frequency") and callable(self.count.term_frequency):
            try:
                # term_frequencyを実行し、それを返す（単純な出現回数行列を取得）
                # 戻り値の3つ目(vectorizer)はここでは不要なので _ で受ける
                M, feature_names, _ = self.count.term_frequency(min_df=min_df)
                return M, feature_names
            except Exception:
                # Exceptionはどんなエラーでもという意味
                # passは何もしないで次に進む。
                pass
        # フォールバック：TF-IDF
        # term_frequencyが使えない場合は、通常のtf_idfメソッドを使用
        M, feature_names, _ = self.count.tf_idf(min_df=min_df)
        return M, feature_names

    # スパース行列の安全な合計・文書頻度計算
    # ------------------------------------------------------------------
    # クラス別集計メソッド
    # 指定されたクラス（マスク）に対応する文書だけを抜き出し、単語ごとの合計値を計算します。
    # ------------------------------------------------------------------
    # [引数]
    # mask : 集計対象の行（ラベル）を示すブール配列
    # [戻り値]
    # term_sum : 各単語の総出現回数（または値の合計）
    # doc_freq : 各単語が出現した文書の数
    def _sum_by_class(self, mask):
        # maskは対象となるラベルをtrueとしたもの→詐欺ラベルの合計を確認したいとき、詐欺:true, 詐欺じゃない：falseとなる。
        # Mは文書と単語の数が記述されているもの：[(0, 14553) 13.0]のようなものが沢山記述。[(文書番号, 単語番号)　出てくる単語回数]　

        # if分で入力の型を判別。スパース（疎）行列かそれ以外numpy行列では適切な処理の仕方が違うため
        # スパース行列とはほとんど値が0の行列。言語系では０ばっかりになるので、値がある所だけ保存することで処理速度を早くする役割
        if sparse.issparse(self.M):
            # Trueの行列の身を抜き出す（サブセットを作成）
            sub = self.M[mask]

            # 縦方向(axis=0)に合計して、各単語の総和を計算
            # np.asarray(...).ravel()で1次元配列に変換
            term_sum = np.asarray(sub.sum(axis=0)).ravel()  # 各語の総和（回数相当）

            # (sub > 0)で値がある場所を1にし、それを合計することで「出現文書数」を計算
            doc_freq = np.asarray((sub > 0).sum(axis=0)).ravel()  # 各語の出現文書数
        else:
            # 密行列（numpy array）の場合の処理
            sub = self.M[mask]
            term_sum = sub.sum(axis=0)
            doc_freq = (sub > 0).sum(axis=0)
            # 配列形状を整える
            term_sum = np.asarray(term_sum).ravel()
            doc_freq = np.asarray(doc_freq).ravel()
        return term_sum, doc_freq

    # ------------------------------------------------------------------
    # 上位単語抽出メソッド
    # 差分ベクトルを受け取り、値が大きい順（または小さい順）に並べてDataFrame化します。
    # ------------------------------------------------------------------
    # [引数]
    # diff_vec : 単語ごとの差分値が入った配列
    # colname  : 結果のDataFrameでの列名
    def _top_n(self, diff_vec, colname):
        # 値に基づいてソートしたインデックスを取得
        order = np.argsort(diff_vec)
        # 降順フラグがあれば逆順にする
        if self.desc:
            order = order[::-1]

        # NaN（欠損値）を除外
        order = [i for i in order if not np.isnan(diff_vec[i])]

        # 上位n個のインデックスを取得
        top_idx = order[:self.n]

        # 結果をDataFrameに格納
        df = pd.DataFrame({
            "rank": np.arange(1, len(top_idx) + 1),  # 順位
            "token": [self.feature_names[i] for i in top_idx],  # 単語名
            colname: diff_vec[top_idx]  # 値
        })
        return df

    # ------------------------------------------------------------------
    # 分析実行・結果出力メソッド
    # 詐欺群と対照群の差異（回数、文書数）を計算し、上位単語を表示・保存します。
    # ------------------------------------------------------------------
    # [引数]
    # to_CSV : CSV保存を行うかどうかのフラグ
    def _print_top_word(self, to_CSV):

        sum1, df1 = self._sum_by_class(self.mask1)  # 出現回数・出現文書数（詐欺群）
        sum0, df0 = self._sum_by_class(self.mask0)  # 出現回数・出現文書数（対照群）

        # 差分ベクトルを計算
        diff_count_1_0 = sum1 - sum0  # 詐欺群の出現回数 − 対照群の出現回数
        diff_doc_1_0 = df1 - df0  # 詐欺群の出現文書数 − 対照群の出現文書数

        # 逆方向の差分も計算（対照群の特徴を見るため）
        diff_count_0_1 = sum0 - sum1  # 対照群の出現回数 − 詐欺群の出現回数
        diff_doc_0_1 = df0 - df1  # 対照群の出現文書数 − 詐欺群の出現文書数

        # TOP100 をDataFrame化
        top_count_1_0 = self._top_n(diff_count_1_0, colname="count_diff_1_minus_0")
        top_doc_1_0 = self._top_n(diff_doc_1_0, colname="docfreq_diff_1_minus_0")
        top_count_0_1 = self._top_n(diff_count_0_1, colname="count_diff_0_minus_1")
        top_doc_0_1 = self._top_n(diff_doc_0_1, colname="docfreq_diff_0_minus_1")

        # 表示
        print("\n=== 詐欺群の出現回数 − 対照群の出現回数（TOP100） ===")
        print(top_count_1_0.to_string(index=False))

        print("\n=== 詐欺群の出現文書数 − 対照群の出現文書数（TOP100） ===")
        print(top_doc_1_0.to_string(index=False))

        print("\n=== 対照群の出現回数 − 詐欺群の出現回数（TOP100） ===")
        print(top_count_0_1.to_string(index=False))

        print("\n=== 対照群の出現文書数 − 詐欺群の出現文書数（TOP100） ===")
        print(top_doc_0_1.to_string(index=False))

        if to_CSV:
            # CSV保存（任意）
            top_count_1_0.to_csv("top100_count_diff_1_minus_0.csv", index=False, encoding="utf-8-sig")
            top_doc_1_0.to_csv("top100_docfreq_diff_1_minus_0.csv", index=False, encoding="utf-8-sig")
            top_count_0_1.to_csv("top100_count_diff_0_minus_1.csv", index=False, encoding="utf-8-sig")
            top_doc_0_1.to_csv("top100_docfreq_diff_0_minus_1.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # 単語正規化・検索メソッド
    # 指定された単語リストをステミング（語幹化）し、重複を除いて返します。
    # ------------------------------------------------------------------
    def serch_word(self, token_target):
        stemmed_list = [stem_text(t) for t in token_target]
        _seen = set()
        return [t for t in stemmed_list if not (t in _seen or _seen.add(t))]

    # ------------------------------------------------------------------
    # 特定単語データ抽出メソッド
    # 指定された単語（列）のデータを、指定されたラベル（行）でフィルタリングして取得します。
    # ------------------------------------------------------------------
    # [引数]
    # col_idx : 単語の列番号
    # mask    : 抽出したい行のマスク
    def _extract_counts_for_label(self, col_idx, mask):
        """列col_idx（語）の文書ごとの値を、指定ラベルの文書だけ抜き出して1次元ndarrayで返す"""
        # 全ての行のcol_idxに該当するものを選んでくる
        # M= (文書番号,　単語番号) 回数 で記述しているので、単語番号を抜き出すことで全ての文書の該当する番号を引っ張ってくるこれがcol
        # maskにはあるクラスがTrueになっている。よってsubでは、そのクラスのcolのみを抜き出している
        col = self.M[:, col_idx]
        sub = col[mask]
        # print('ここからsubの処理が始まっているよ')
        # print(sub)
        # 疎・密どちらでも1次元配列へ
        if hasattr(sub, "toarray"):
            return np.asarray(sub.toarray()).ravel()
        return np.asarray(sub).ravel()

    # ------------------------------------------------------------------
    # 文書ID生成メソッド
    # グラフのX軸ラベル用に、文書IDのリストを作成します。
    # ------------------------------------------------------------------
    def _doc_ids(self, mask, prefix):
        """そのラベル内での文書番号（1始まり）からなる表示用ラベル配列"""
        n = int(mask.sum())
        return [f"{prefix}{i}" for i in range(1, n + 1)]

    # ------------------------------------------------------------------
    # 複数単語カウント集計メソッド
    # 指定された単語リストについて、各文書での出現数をDataFrameにまとめます。
    # ------------------------------------------------------------------
    def _token_count(self, tokens_target, mask, prefix):
        df_l = pd.DataFrame(index=self._doc_ids(mask, prefix))

        vals = None  # 変数初期化

        for tok in tokens_target:
            # 語彙にない単語は0埋め
            if tok not in self.vocab_index:
                vals = np.zeros(mask.sum(), dtype=float)
            else:
                # 語彙にある場合はデータを抽出
                j = self.vocab_index[tok]
                vals = self._extract_counts_for_label(j, mask)
            df_l[tok] = vals

        return vals, df_l

    # ------------------------------------------------------------------
    # 単語出現可視化メソッド
    # 指定された単語について、全文書での出現数を棒グラフで表示します（ラベル別色分け）。
    # ------------------------------------------------------------------
    # [引数]
    # token_target : 可視化したい単語のリスト
    def _token_plot(self, token_target):
        # 単語の正規化（ステミング・重複排除）
        tokens_target = self.serch_word(token_target)

        # X軸ラベルの作成
        X_label1 = self._doc_ids(self.mask1, "L1_")
        X_label0 = self._doc_ids(self.mask0, "L0_")

        # 各ラベルごとのデータを取得
        val_1, df_l1 = self._token_count(tokens_target, self.mask1, "L1_")
        val_0, df_l0 = self._token_count(tokens_target, self.mask0, "L0_")

        X_all = X_label1 + X_label0

        # 各単語ごとにグラフを描画
        for tok in tokens_target:
            # ラベル1とラベル0の値をそれぞれ取得（上の処理で既に計算済みのものを再利用）
            if tok in df_l1.columns:
                vals_l1 = df_l1[tok].to_numpy()
            else:
                vals_l1 = np.zeros(len(X_label1), dtype=float)

            if tok in df_l0.columns:
                vals_l0 = df_l0[tok].to_numpy()
            else:
                vals_l0 = np.zeros(len(X_label0), dtype=float)

            # 全ドキュメント順に結合（順序：L1群 → L0群）
            vals_all = np.concatenate([vals_l1, vals_l0])

            # ラベルに応じた色配列を作成（L1=青, L0=赤）
            colors = (["blue"] * len(vals_l1)) + (["red"] * len(vals_l0))

            # 図示：1語につき1枚、文書を横軸、色でラベルを区別
            plt.figure()
            plt.bar(range(len(vals_all)), vals_all, color=colors)

            # X軸の目盛り設定
            plt.xticks(range(len(vals_all)), X_all, rotation=90)

            plt.title(f"All documents (colored by label): '{tok}'")
            plt.xlabel("Documents (L1 then L0)")
            plt.ylabel("Count")
            # 簡易凡例（色のみ）
            legend_handles = [Patch(facecolor="blue", label="Label 1"),
                              Patch(facecolor="red", label="Label 0")]
            plt.legend(handles=legend_handles, loc="upper right", frameon=False)

            plt.tight_layout()
            # 画像ファイルとして保存
            plt.savefig(f"{tok}_all_docs_colored.png", dpi=150)
            plt.show()
