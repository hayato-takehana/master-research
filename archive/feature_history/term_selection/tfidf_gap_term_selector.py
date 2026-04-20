import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# ------------------------------------------------------------------
# 詐欺群と対照群（正常群）における単語のTF-IDF平均値の差を利用して、
# 判別に寄与する重要な単語を抽出し、特徴量行列を圧縮するクラス
#
# ロジック:
# 1. 詐欺群のTF-IDF平均 - 対照群のTF-IDF平均 を計算
# 2. 値が大きくプラスなら「詐欺によく出る」、マイナスなら「対照群によく出る」と判断
# 3. 両方の方向で絶対値が大きい単語を抽出し、それ以外の単語をデータから削除する
# ------------------------------------------------------------------
class Sagi_word:

    # ------------------------------------------------------------------
    # コンストラクタ
    # 分析に必要なデータと設定を受け取ります。
    # ------------------------------------------------------------------
    # [引数]
    # X          : 全文書のTF-IDF行列（行：文書、列：単語）。sparse matrixまたはdense matrix。
    # labels     : 正解ラベルのリスト（1:詐欺, 0:対照群/正常）
    # number     : 上位何個の単語を抽出するか（詐欺寄り、正常寄りそれぞれ抽出する数）
    # vectorizer : 使用したTfidfVectorizerのインスタンス（単語名と列番号の対応付けに必要）
    def __init__(self, X, labels, number, vectorizer):
        self.X = X
        self.labels = labels
        self.number = number
        self.vectorizer = vectorizer

    # ------------------------------------------------------------------
    # 特徴語抽出・行列圧縮メソッド
    # 平均値の差を計算し、重要語のみを残した新しい行列を返します。
    # ------------------------------------------------------------------
    # [戻り値]
    # X_top_features : 選ばれた重要な単語の列のみに絞り込まれた特徴量行列
    def word_output(self):
        # 1. データをラベルごとに分割
        #    ラベルが1（詐欺）の行と、0（対照群）の行をそれぞれ抽出してサブセットを作成
        X_fraud = self.X[np.array(self.labels) == 1]  # 詐欺データ群
        X_control = self.X[np.array(self.labels) == 0]  # 対照群データ群

        # 2. 単語ごとのTF-IDF特徴量の平均値を計算
        #    axis=0 は「列方向（縦）の平均」、つまり単語ごとの平均値を算出
        #    .A1 は np.matrix 形式を平坦な1次元配列(array)に変換する処理（計算用）
        fraud_mean = X_fraud.mean(axis=0).A1  # 詐欺データの平均ベクトル
        control_mean = X_control.mean(axis=0).A1  # 対照群データの平均ベクトル

        # 3. 平均値の差（Mean Difference）を計算
        #    プラスの値 -> 詐欺群での平均TF-IDFが高い（詐欺の特徴語）
        #    マイナスの値 -> 対照群での平均TF-IDFが高い（正常の特徴語）
        mean_diff = fraud_mean - control_mean

        # 4. 分析結果を扱いやすくするためにDataFrameにまとめる
        #    vectorizerから全単語のリストを取得
        terms = np.array(self.vectorizer.get_feature_names_out())

        mean_diff_df = pd.DataFrame({
            'term': terms,  # 単語名
            'mean_diff': mean_diff,  # 平均差（符号あり）
            'abs_mean_diff': np.abs(mean_diff)  # 平均差の絶対値（特徴としての強さ）
        })

        # 5. 絶対値に基づいてランキングを作成し、単語を抽出

        # [Positive Terms] 詐欺寄りの単語
        # mean_diffがプラス(>0)のものを対象に、絶対値が大きい順に並べ替え、上位number個を取得
        positive_terms = mean_diff_df[mean_diff_df['mean_diff'] > 0].sort_values(
            by='abs_mean_diff', ascending=False
        ).head(self.number)

        # [Negative Terms] 対照群寄りの単語
        # mean_diffがマイナス(<0)のものを対象に、絶対値が大きい順に並べ替え、上位number個を取得
        negative_terms = mean_diff_df[mean_diff_df['mean_diff'] < 0].sort_values(
            by='abs_mean_diff', ascending=False
        ).head(self.number)

        # --- 結果の確認（コンソール出力） ---
        print(f"sagi terms (Top {self.number}):")
        print(positive_terms[['term', 'mean_diff']])

        print(f"no_sagi terms (Top {self.number}):")
        print(negative_terms[['term', 'mean_diff']])

        # 6. 単語名から列インデックス（行列の何列目か）を取得
        #    vectorizer.vocabulary_ 辞書を使用して、単語文字列を整数のインデックスに変換
        positive_term_indices = [self.vectorizer.vocabulary_[term] for term in positive_terms['term']]
        negative_term_indices = [self.vectorizer.vocabulary_[term] for term in negative_terms['term']]

        print("Indices of sagi terms:", positive_term_indices)
        print("Indices of no_sagi terms:", negative_term_indices)

        # 7. 抽出したインデックスを結合
        #    詐欺語インデックスリストと正常語インデックスリストを合わせて、1つのリストにする
        top_indices = positive_term_indices + negative_term_indices

        # 8. 元の特徴量行列 X から、該当する列だけをスライス（抜き出し）
        #    self.X[:, top_indices] -> 行はすべて保持し、列はtop_indicesにあるものだけを残す
        X_top_features = self.X[:, top_indices]

        # --- 最終的な行列形状の確認 ---
        print("元の X の形状:", self.X.shape)
        print("抽出後の X_top_features の形状:", X_top_features.shape)
        # print(X_top_features) # 必要に応じて中身を表示

        return X_top_features