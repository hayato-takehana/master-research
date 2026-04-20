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
from sklearn.decomposition import PCA
from term_feature_selector import Find_terms
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid  # パラメータ生成用に使いやすいため追加
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # 可視化のためにseabornをインポート
import numpy as np
import itertools
from sklearn.model_selection import cross_val_score


# ------------------------------------------------------------------
# CSV保存用関数
# 生成された特徴量データフレームに正解ラベルを付与し、CSV形式で保存します。
# ------------------------------------------------------------------
# [引数]
# rate_combined_1_2 : 特徴量のDataFrame（結合済み）
# labels            : 各データの正解ラベル（0 or 1）
# CSV_true          : 保存を実行するかどうかのフラグ（Trueで実行）
# csv_word          : 保存するファイル名
def CSV(rate_combined_1_2, labels, CSV_true, csv_word):
    # 保存フラグがTrueの場合のみ実行
    if CSV_true:
        # 元のデータフレームを変更しないようにコピーを作成
        rate_combined_1_2_labels = rate_combined_1_2.copy()

        # 新しい列 'label' を作成し、正解ラベルを代入
        rate_combined_1_2_labels['label'] = labels

        # CSVファイルとして書き出し
        # index=False: 行番号（0, 1, 2...）をファイルに含めない
        # encoding='utf-8-sig': Excelで開いた際の文字化けを防ぐエンコーディング
        rate_combined_1_2_labels.to_csv(csv_word, index=False, encoding='utf-8-sig')


# 旧 prior study パイプラインの補助関数


# ------------------------------------------------------------------
# PCA（主成分分析）結果と正解ラベルの関係性分析関数
# 第一主成分（PC1）が、詐欺（Label 1）と正常（Label 0）をどれくらい分離できているかを分析します。
# ------------------------------------------------------------------
# [引数]
# X_pca  : PCAによって変換されたデータ（主成分得点）
# labels : 正解ラベル
def analyze_pc_label_relationship(X_pca, labels):
    """
    第一主成分(PC1)と正解ラベルの関係性を分析し、可視化する。

    Args:
        X_pca (np.array): 主成分スコア（PCAの実行結果）
        labels (list or np.array): 正解ラベル
    """
    # 主成分データが存在しない（列数が1未満）場合は処理をスキップ
    if X_pca.shape[1] < 1:
        print("主成分が存在しないため、分析をスキップします。")
        return

    # 全データの第一主成分（PC1）のスコア（0列目）のみを抽出
    pc1_scores = X_pca[:, 0]

    # 分析・可視化しやすいようにDataFrame化
    df = pd.DataFrame({'PC1_Score': pc1_scores, 'label': labels})

    print("\n\n========================================================")
    print("=== 第一主成分(PC1)と正解ラベルの関係性 分析レポート ===")
    print("========================================================")

    # 1. 相関係数の計算
    #    PC1の値とラベル(0, 1)の相関を見ることで、PC1軸が詐欺判別に有効かを確認
    #    1に近い：PC1が大きいほど詐欺、-1に近い：PC1が小さいほど詐欺
    correlation = df['PC1_Score'].corr(df['label'])
    print(f"\n[相関係数]")
    print(f"PC1スコアと正解ラベルの相関係数: {correlation:.4f}")

    # 2. グループごとの平均スコアを計算
    #    詐欺群(1)と正常群(0)で、PC1の平均値に有意な差があるかを見る
    mean_scores = df.groupby('label')['PC1_Score'].mean()
    print("\n[グループごとの平均スコア]")
    print(mean_scores)

    # 3. 分析結果の自動解釈と表示
    print("\n[軸の解釈]")
    if abs(correlation) < 0.2:
        # 相関が弱い場合
        print("PC1は、詐欺と非詐欺の分類にあまり関連がない軸のようです。")
    elif correlation > 0:
        # 正の相関がある場合
        print("PC1は「詐欺らしさの軸」の可能性が高いです。")
        print("-> PC1スコアが【大きい】ほど、詐欺(ラベル1)である傾向があります。")
    else:  # correlation < 0
        # 負の相関がある場合
        print("PC1は「非詐欺らしさの軸」の可能性が高いです。")
        print("-> PC1スコアが【小さい】ほど、詐欺(ラベル1)である傾向があります（大きいほど非詐欺）。")

    # 4. 可視化 (Box Plot / 箱ひげ図)
    #    ラベルごとのデータの分布（中央値、四分位範囲、外れ値）を視覚的に比較
    print("\n[分布の可視化]")
    print("各ラベルのPC1スコアの分布を箱ひげ図で表示します。")
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='label', y='PC1_Score', data=df)
    plt.title('PC1 Scores by Label')
    plt.xlabel('Label (0: Not Scam, 1: Scam)')
    plt.ylabel('PC1 Score')
    plt.grid(True)
    plt.show()
    print("=" * 56)


# ------------------------------------------------------------------
# 主成分分析(PCA)実行関数
# 多次元データを低次元（ここでは可視化用に主成分）に圧縮し、データの構造を可視化します。
# ------------------------------------------------------------------
# [引数]
# rate_combined_1_2 : 分析対象の特徴量DataFrame
# PCA_features      : 特徴量の総数（2つ以上ないとPCAの意味がないため確認用）
# labels            : プロットの色分け用ラベル
# PCA_true          : PCAを実行するかどうかのフラグ
# PCA_word          : グラフタイトル用文字列
def PCA_do(rate_combined_1_2, PCA_features, labels, PCA_true, PCA_word):
    # PCAフラグがTrue かつ 特徴量が2個以上ある場合のみ実行
    if PCA_true and PCA_features > 1:
        # --- 1. 主成分分析の実行 ---
        # 可視化や分析のために、上位10個の主成分を計算する
        N_COMPONENTS_FOR_VIZ = 10
        print(f"\n--- 主成分分析 (PCA) を {N_COMPONENTS_FOR_VIZ} 個の主成分で実行します ---")
        pca = PCA(n_components=N_COMPONENTS_FOR_VIZ)

        # データを主成分空間に射影（fit_transform）
        # X_pca には各データの主成分得点（座標）が入る
        X_pca = pca.fit_transform(rate_combined_1_2)

        # --- 2. 主成分得点の表示 ---
        print("\n【主成分得点】")
        print("各データが新しい主成分軸上でどの座標に位置するかを示します。")
        df_scores = pd.DataFrame(X_pca, columns=[f'PC{i + 1}' for i in range(N_COMPONENTS_FOR_VIZ)])
        print(df_scores.head())  # 先頭5行のみ表示

        # --- 3. 主成分負荷量（Loadings）の表示 ---
        # 各主成分が、元のどの単語（特徴量）から強く影響を受けているかを確認
        print("\n【主成分負荷量】")
        print("各主成分を構成する元の特徴量（単語）の重み（寄与度）を示します。")
        loadings = pca.components_
        feature_names = rate_combined_1_2.columns.values
        num_display_words = 5  # 上位何単語を表示するか

        # 各主成分ごとにループ
        for i, component in enumerate(loadings):
            # 単語名と負荷量をペアにしてDataFrame化し、負荷量の値でソート
            loadings_df = pd.DataFrame({'feature': feature_names, 'loading': component})
            sorted_loadings = loadings_df.sort_values(by='loading', ascending=False)

            print(f"\n--- 第 {i + 1} 主成分 (PC{i + 1}) ---")
            # プラスの影響が大きい単語
            print("  [+] 正の寄与が大きい上位単語:")
            print(sorted_loadings.head(num_display_words).to_string(index=False))

            # マイナスの影響が大きい単語
            print("\n  [-] 負の寄与が大きい上位単語:")
            print(sorted_loadings.tail(num_display_words).sort_values(by='loading').to_string(index=False))

        # --- 4. 寄与率の表示 ---
        # 各主成分が元のデータセットの情報量をどれくらい説明できているか
        print("\n【寄与率】")
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = explained_variance.cumsum()  # 累積寄与率
        print(f"各主成分の寄与率: {np.round(explained_variance, 4)}")
        print(f"累積寄与率: {np.round(cumulative_variance, 4)}")

        # --- 5. 2次元プロットによる可視化 ---
        print("\n--- 主成分の2次元プロットを生成します ---")
        plt.figure(figsize=(10, 8))
        # PC1をX軸、PC2をY軸、ラベルで色分けして散布図を描画
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='Set1', alpha=0.7)
        plt.xlabel(f'PC1 ({explained_variance[0]:.2%})')
        plt.ylabel(f'PC2 ({explained_variance[1]:.2%})')
        plt.title(PCA_word)
        plt.legend(*scatter.legend_elements(), title="class")
        plt.grid(True)
        plt.show()

        # ★★★ 追加 ★★★
        # PC1とラベルの関係性を詳しく分析する関数を呼び出し
        analyze_pc_label_relationship(X_pca, labels)


# ------------------------------------------------------------------
# 特徴量結合・実行判断関数
# 指定された2つの特徴量抽出方法の結果を結合し、最終的なデータセットを作成します。
# ------------------------------------------------------------------
# [引数]
# label1_minus_label0 : 抽出方法1（詐欺寄り）の特徴数
# label0_minus_label1 : 抽出方法2（正常寄り）の特徴数
# term_1, term_2      : 特徴量DFを生成する関数（lambda関数などで遅延実行）
# word                : ログ出力用の説明文
# csv_word            : CSVファイル名
# PCA_word            : PCAタイトル
# labels              : 正解ラベル
def decide_X(label1_minus_label0, label0_minus_label1, CSV_true, PCA_true, term_1, term_2, word, csv_word, PCA_word,
             labels):
    # ケースA: 両方の特徴量を使用する場合
    if label1_minus_label0 > 0 and label0_minus_label1 > 0:
        rate_1 = term_1()  # 特徴量1を計算・取得
        rate_2 = term_2()  # 特徴量2を計算・取得
        # 横方向（axis=1）に結合して1つのテーブルにする
        rate_combined_1_2 = pd.concat([rate_1, rate_2], axis=1)
        print(word)
        # print(rate_1)
        # print(rate_2)
        # print(rate_combined_1_2)

    # ケースB: 抽出方法1のみ使用
    elif label1_minus_label0 > 0 and label0_minus_label1 <= 0:
        rate_combined_1_2 = term_1()
        print("抽出方法１のみ使用しています")
        print(word)
        # print(rate_combined_1_2)

    # ケースC: 抽出方法2のみ使用
    elif label1_minus_label0 <= 0 and label0_minus_label1 > 0:
        rate_combined_1_2 = term_2()
        print("抽出方法2のみ使用しています")
        print(word)
        # print(rate_combined_1_2)

    # ケースD: どちらも使用しない（エラー）
    else:
        return print("扱える特徴が存在しません")

    # 機械学習モデルに入力するために、DataFrameをnumpy配列(values)に変換
    X = rate_combined_1_2.values

    # 特徴量の総数
    PCA_features = label1_minus_label0 + label0_minus_label1

    # CSV保存を実行
    CSV(rate_combined_1_2, labels, CSV_true, csv_word)

    # PCA分析を実行
    PCA_do(rate_combined_1_2, PCA_features, labels, PCA_true, PCA_word)

    return X


# ------------------------------------------------------------------
# モデルの手動グリッドサーチ＆交差検証関数
# ※元のコードでは途中で処理が切れていたため、passの追加を行っています。
# ------------------------------------------------------------------
def model_do(X, labels, param_grid, model_class):
    # 特定のインデックス(152)のデータを削除（外れ値やエラーデータの除外処理）
    X = np.delete(X, 152, axis=0)
    labels.pop(152)

    # ラベルをnumpy配列に変換
    labels = np.array(labels)

    # パラメータグリッドのキーと値をリスト化
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    # 外側の交差検証（Outer CV）の設定：10分割
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # itertools.productを使って、ハイパーパラメータの全組み合わせをループ
    for combination in itertools.product(*values):
        # 現在のパラメータセットを辞書化
        current_params = dict(zip(keys, combination))

        # モデルのインスタンス化
        # random_state引数が使えるモデルと使えないモデルに対応
        try:
            model = model_class(**current_params, random_state=42)
        except TypeError:
            # random_state引数を持たないモデルの場合
            model = model_class(**current_params)

        # 外側の交差検証ループ（学習データとテストデータに分割）
        for train_index, test_index in outer_cv.split(X, labels):
            X_new_train = X[train_index]
            y_new_train = labels[train_index]
            X_new_test = X[test_index]
            y_new_test = labels[test_index]

            # モデルを学習
            model.fit(X_new_train, y_new_train)

            # 学習データに対する予測精度（Accuracy）を確認
            # ※過学習のチェックなどに利用
            y_train_pred = model.predict(X_new_train)
            accuracy_train = accuracy_score(y_new_train, y_train_pred)

            # ★修正箇所★
            # 元のコードではここに処理が記述されていませんでした。
            # 構文エラー回避のため pass を入れています。
            if accuracy_train == 1:
                pass

            # ------------------------------------------------------------------


# 最適パラメータ探索関数
# グリッドサーチを行い、最も精度の高いパラメータセットを見つけます。
# ※元のコードではループ処理等が欠落していたため、推測で補完しています。
# ------------------------------------------------------------------
# ★修正箇所★ 関数定義末尾にコロン(:)を追加
def find_best_params_cv(X_train, y_train, model_class, param_grid, cv=5, scoring='accuracy'):
    # ベストスコア記録用の変数を初期化
    best_score = 0
    best_params = None

    # ★修正箇所★
    # パラメータの組み合わせをループする処理（ParameterGrid）を追加しました。
    # これがないと、grid searchとして機能しません。
    for current_params in ParameterGrid(param_grid):

        # --------------------------------------------------------------------------
        # モデルのインスタンス化
        # モデルクラスと、現在のパラメータの組み合わせ（**current_params）から
        # モデルのインスタンスを作成します。
        # random_stateを持つモデルと持たないモデル両方に対応するため、例外処理を入れています。
        # --------------------------------------------------------------------------
        try:
            model = model_class(**current_params, random_state=42)
        except TypeError:
            model = model_class(**current_params)

        # 交差検証による性能評価
        # cv=5 ならデータを5分割して検証を行い、その平均スコアを算出
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        mean_score = np.mean(scores)

        # 途中経過の表示（デバッグ用）
        # print(f"パラメータ: {current_params} | 平均Accuracy: {mean_score:.4f}")

        # 最良スコアが更新された場合、スコアとパラメータを記録
        if mean_score > best_score:
            best_score = mean_score
            best_params = current_params

    # 最終的な結果の表示
    print("\n--- 探索終了 ---")
    print(f"最良スコア: {best_score:.4f}")
    print(f"最適なパラメータ: {best_params}")

    return best_score, best_params


# ------------------------------------------------------------------
# メイン処理関数: pre_research
# 前処理、特徴量生成、CSV保存、PCA、モデル学習までを一括管理する関数
# ------------------------------------------------------------------
# [引数]
# label1_minus_label0 : 抽出方法1の特徴数（詐欺率 - 正常率）
# label0_minus_label1 : 抽出方法2の特徴数（正常率 - 詐欺率）
# number              : 特徴量モード (1:バイナリ, 2:頻度, 3:列正規化, 4:行正規化)
# CSV_true, PCA_true  : 保存/分析実行フラグ
# svm_linear_true...  : SVM実行フラグ
def pre_research(label1_minus_label0, label0_minus_label1, number, CSV_true, PCA_true, svm_linear_true, svm_rbf_true):
    # 関数内でのインポート（元のコードに従います）
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # データの読み込みと前処理（common関数を使用）
    X, labels, feature_names, vectorizer = common(False, False)
    # 疎行列を密行列に変換
    X = X.toarray()

    # 特徴語抽出用クラスの初期化
    find_terms = Find_terms(X, labels, feature_names)

    # --- 特徴数の決定と特徴語リストの取得 ---

    # 抽出方法1（詐欺寄り）の実行
    if label1_minus_label0 > 0:
        rate_label1_minus_label0 = find_terms.rate_label1_minus_label0(label1_minus_label0)
    else:
        print("抽出方法1は使用しません")

    # 抽出方法2（正常寄り）の実行
    if label0_minus_label1 > 0:
        rate_label0_minus_label1 = find_terms.rate_label0_minus_label1(label0_minus_label1)
    else:
        print("抽出方法2は使用しません")

    print(f"合計の特徴数{label1_minus_label0 + label0_minus_label1}")

    # --- 特徴量データセットの生成と処理 ---
    # number引数に応じて処理を分岐

    if number == 1:
        # [モード1: バイナリー変数]
        # 単語の有無（0 or 1）で特徴量を作成
        # lambda関数を使って、「後で実行される関数」としてdecide_Xに渡している
        X = decide_X(label1_minus_label0, label0_minus_label1, CSV_true, PCA_true,
                     lambda: find_terms.create_topn_binary_fueature(rate_label1_minus_label0),
                     lambda: find_terms.create_topn_binary_fueature(rate_label0_minus_label1),
                     "バイナリー変数での特徴量",
                     "先行研究でのバイナリー変数の特徴量.csv", 'pre_research_binary', labels)

    elif number == 2:
        # [モード2: 出現回数]
        # 単語の出現回数をそのまま特徴量とする
        X = decide_X(label1_minus_label0, label0_minus_label1, CSV_true, PCA_true,
                     lambda: find_terms.extract_terms_features(rate_label1_minus_label0),
                     lambda: find_terms.extract_terms_features(rate_label0_minus_label1),
                     "出現回数での特徴量",
                     "先行研究での出現回数の特徴量.csv", 'pre_research_number', labels)

    elif number == 3:
        # [モード3: 列正規化]
        # 出現回数を数えた後、列（単語）ごとに0〜1に正規化する
        X = decide_X(label1_minus_label0, label0_minus_label1, CSV_true, PCA_true,
                     lambda: find_terms.extract_terms_features_normalization(rate_label1_minus_label0),
                     lambda: find_terms.extract_terms_features_normalization(rate_label0_minus_label1),
                     "出現回数を列で正規化した特徴量",
                     "先行研究での出現回数を列で正規化したの特徴量.csv", 'pre_research_column', labels)

    elif number == 4:
        # [モード4: 行正規化]
        # 出現回数を数えた後、行（文書）ごとに正規化する
        # ※行正規化は結合後に行う必要があるため、decide_Xを使わず個別に処理フローを記述

        if label1_minus_label0 > 0 and label0_minus_label1 > 0:
            rate_1 = find_terms.extract_terms_features(rate_label1_minus_label0)
            rate_2 = find_terms.extract_terms_features(rate_label0_minus_label1)
            # 結合
            rate_combined_1_2 = pd.concat([rate_1, rate_2], axis=1)
            # 行正規化を実行
            rate_combined_1_2 = find_terms.extract_terms_features_row_normalization(rate_combined_1_2)

            print("出現回数を行で正規化した特徴量")
            print(rate_1)
            print(rate_2)
            print(rate_combined_1_2)

        elif label1_minus_label0 > 0 and label0_minus_label1 <= 0:
            rate_combined_1_2 = find_terms.extract_terms_features(rate_label1_minus_label0)
            rate_combined_1_2 = find_terms.extract_terms_features_row_normalization(rate_combined_1_2)
            print("抽出方法１のみ使用しています")
            print("出現回数を行で正規化した特徴量")
            print(rate_combined_1_2)

        elif label1_minus_label0 <= 0 and label0_minus_label1 > 0:
            rate_combined_1_2 = find_terms.extract_terms_features(rate_label0_minus_label1)
            rate_combined_1_2 = find_terms.extract_terms_features_row_normalization(rate_combined_1_2)
            print("抽出方法2のみ使用しています")
            print("出現回数を行で正規化した特徴量")
            print(rate_combined_1_2)

        else:
            return print("扱える特徴が存在しません")

        # 機械学習用に配列化
        X = rate_combined_1_2.values
        PCA_features = label1_minus_label0 + label0_minus_label1

        # 保存とPCA実行（decide_Xを通さないため手動実行）
        CSV(rate_combined_1_2, labels, CSV_true, "先行研究での出現回数を行で正規化したの特徴量.csv")
        PCA_do(rate_combined_1_2, PCA_features, labels, PCA_true, 'pre_research_column')


    else:
        return print("正しい変数でないため実行ができません")

    # print("")

    # SVM実行フラグの確認
    if not svm_rbf_true and not svm_linear_true:
        return print("svmの実行は行わないで終了")
