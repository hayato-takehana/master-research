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

from dataset_loader import common
from term_feature_selector import Find_terms
from nested_leave_one_out import Leave_one_out
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from prior_study_pipeline import decide_X


# ------------------------------------------------------------------
# 実際の詐欺データ分析用メイン関数
# 抽出した特徴語を用いてデータセットを作成し、可視化とSVMによる分類精度評価を行います。
# ------------------------------------------------------------------
# [引数]
# all_label1      : 抽出方法1の特徴数（詐欺文書すべてに出現する単語）
# not_in_label1   : 抽出方法2の特徴数（詐欺文書には出現しない単語）
# number          : 特徴量生成モード
#                   1: バイナリー変数（有無）
#                   2: 出現回数
#                   3: 列での正規化（単語ごとのスケーリング）
#                   4: 行での正規化（文書ごとのスケーリング）
# CSV_true        : 特徴量をCSV保存するかどうか
# PCA_true        : 主成分分析による可視化を行うかどうか
# svm_true        : SVMによる分類精度の検証を行うかどうか
def real_scam(all_label1, not_in_label1, number, CSV_true, PCA_true, svm_true):
    # ------------------------------------------------------------------
    # 1. データの読み込みと準備
    # ------------------------------------------------------------------
    # common関数を呼び出してデータを取得
    # 第1引数 True  : 実際の詐欺データ(real_scam)を使用する
    # 第2引数 False : TF-IDFではなく、頻度ベースなどの生データを使用する設定
    X, labels, feature_names, vectorizer = common(True, False)

    # 疎行列を密行列(numpy array)に変換して扱いやすくする
    X = X.toarray()

    # 特徴語抽出クラスのインスタンス化
    find_terms = Find_terms(X, labels, feature_names)

    # ------------------------------------------------------------------
    # 2. 特徴語（使用する単語）の決定
    # ------------------------------------------------------------------

    # --- 抽出方法1: 「詐欺(1)には必ずあり、正常(0)には少ない」単語 ---
    if all_label1 > 0:
        # 指定された数だけ単語リストを取得
        all_label1_least_label0 = find_terms.all_label1_least_label0(all_label1)
    else:
        print("抽出方法1は使用しません")

    # --- 抽出方法2: 「詐欺(1)には全くなく、正常(0)には多い」単語 ---
    if not_in_label1 > 0:
        # 指定された数だけ単語リストを取得
        not_in_label1_most_label0 = find_terms.not_label1_most_label0(not_in_label1)
    else:
        print("抽出方法2は使用しません")

    # ------------------------------------------------------------------
    # 3. 特徴量データセット(X)の生成
    # numberの値（1〜4）に応じて、データの加工方法を切り替える
    # ------------------------------------------------------------------

    if number == 1:
        # [モード1: バイナリー変数]
        # 単語の出現有無(0 or 1)で特徴量を作成
        # decide_X関数を利用して、結合・保存・PCAを一括処理
        X = decide_X(all_label1, not_in_label1, CSV_true, PCA_true,
                     lambda: find_terms.create_topn_binary_fueature(all_label1_least_label0),
                     lambda: find_terms.create_topn_binary_fueature(not_in_label1_most_label0),
                     "バイナリー変数での特徴量",
                     "実際の詐欺でのバイナリー変数の特徴量.csv", 'real_scam_binary', labels)

    elif number == 2:
        # [モード2: 出現回数]
        # 単語の出現回数をそのまま使用
        X = decide_X(all_label1, not_in_label1, CSV_true, PCA_true,
                     lambda: find_terms.extract_terms_features(all_label1_least_label0),
                     lambda: find_terms.extract_terms_features(not_in_label1_most_label0),
                     "出現回数での特徴量",
                     "実際の詐欺での出現回数の特徴量.csv", 'real_scam_number', labels)

    elif number == 3:
        # [モード3: 列での正規化]
        # 単語ごとに0〜1に正規化（MinMaxScaling）
        X = decide_X(all_label1, not_in_label1, CSV_true, PCA_true,
                     lambda: find_terms.extract_terms_features_normalization(all_label1_least_label0),
                     lambda: find_terms.extract_terms_features_normalization(not_in_label1_most_label0),
                     "出現回数を列で正規化した特徴量",
                     "実際の詐欺での出現回数を列で正規化したの特徴量.csv", 'real_scam_column', labels)

    elif number == 4:
        # [モード4: 行での正規化]
        # 文書ごとに単語出現数のバランスを正規化
        # ※このモードだけdecide_Xを使わず、個別に実装されています

        # パターンA: 両方の特徴量を使う場合
        if all_label1 > 0 and not_in_label1 > 0:
            term_df = find_terms.extract_terms_features(all_label1_least_label0)
            term_df_2 = find_terms.extract_terms_features(not_in_label1_most_label0)

            # 結合してから行正規化を実行
            df_combined_term_df_row_normalization_1_2 = pd.concat([term_df, term_df_2], axis=1)
            df_combined_term_df_row_normalization_1_2 = find_terms.extract_terms_features_row_normalization(
                df_combined_term_df_row_normalization_1_2)

            print("出現回数を行で正規化した特徴量")
            # print(term_df)
            # print(term_df_2)
            print(df_combined_term_df_row_normalization_1_2)

        # パターンB: 抽出方法1のみ使う場合
        elif all_label1 > 0 and not_in_label1 <= 0:
            df_combined_term_df_row_normalization_1_2 = find_terms.extract_terms_features(all_label1_least_label0)
            df_combined_term_df_row_normalization_1_2 = find_terms.extract_terms_features_row_normalization(
                df_combined_term_df_row_normalization_1_2)

            print("抽出方法１のみ使用しています")
            print("出現回数を行で正規化した特徴量")
            print(df_combined_term_df_row_normalization_1_2)

        # パターンC: 抽出方法2のみ使う場合
        elif all_label1 <= 0 and not_in_label1 > 0:
            df_combined_term_df_row_normalization_1_2 = find_terms.extract_terms_features(not_in_label1_most_label0)
            df_combined_term_df_row_normalization_1_2 = find_terms.extract_terms_features_row_normalization(
                df_combined_term_df_row_normalization_1_2)

            print("抽出方法2のみ使用しています")
            print("出現回数を行で正規化した特徴量")
            print(df_combined_term_df_row_normalization_1_2)
        else:
            return print("扱える特徴が存在しません")

        # 機械学習用に変数をセット
        X = df_combined_term_df_row_normalization_1_2
        PCA_features = all_label1 + not_in_label1

        # CSV保存処理（手動）
        if CSV_true:
            df_combined_term_df_row_normalization_1_2.to_csv("実際の詐欺での出現回数を行で正規化したの特徴量.csv",
                                                             index=False, encoding='utf-8-sig')

        # PCA実行と可視化（手動）
        if PCA_true and PCA_features > 1:
            # 1. PCAの計算
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(df_combined_term_df_row_normalization_1_2)

            # 2. 散布図の描画
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='Set1', alpha=0.7)
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title('real_scam_row')  # グラフタイトル
            plt.legend(*scatter.legend_elements(), title="class")
            plt.grid(True)
            plt.show()
    else:
        return print("正しい変数でないため実行ができません")

    print("")

    # ------------------------------------------------------------------
    # 4. SVMによる分類精度の検証
    # svm_trueフラグが立っている場合のみ実行
    # ------------------------------------------------------------------
    if svm_true:
        # DataFrameの場合があるため、numpy配列またはarray形式に統一
        # ※Leave_one_outクラスはarray入力を期待しているため
        if isinstance(X, pd.DataFrame):
            X = X.values
        y = np.array(labels)

        print("")
        print("今回使用する特徴量データ（先頭部分などを確認）:")
        # print(X) # 必要であれば表示

        # --- (1) 線形SVM (Linear Kernel) の実行 ---
        # モデル定義
        svm_linear = SVC(kernel='linear')

        # 探索するハイパーパラメータCの範囲
        # C: 誤分類をどれだけ許容するか（大きいほど厳しく、小さいほど緩やか）
        param_grid = {
            'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
            # 'C': [1, 10], # テスト用に範囲を狭める場合はこちらを使用
        }

        # Leave-One-Out交差検証クラスの呼び出しと実行
        leave_one = Leave_one_out(X, y, svm_linear, param_grid)
        print("--- Linear SVM Result ---")
        leave_one.leave_one_out_printout()

        # --- (2) RBFカーネルSVM (Radial Basis Function) の実行 ---
        # 非線形分離が可能なモデル
        svm_rbf = SVC(kernel='rbf')

        # ハイパーパラメータ範囲
        # gamma: 決定境界の複雑さを決める（大きいほど複雑、小さいほど滑らか）
        param_grid = {
            'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
            'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, "scale", "auto"]
        }

        # Leave-One-Out実行
        leave_one = Leave_one_out(X, y, svm_rbf, param_grid)
        print("--- RBF SVM Result ---")
        leave_one.leave_one_out_printout()
    else:
        return print("svmの実行は行わないで終了")
