from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.metrics import confusion_matrix, recall_score, f1_score, accuracy_score
from sklearn.base import clone


# ------------------------------------------------------------------
# ダブル・リーブワンアウト（Nested Leave-One-Out）交差検証を行うクラス
#
# データ数が少ない場合に、モデルの性能評価とハイパーパラメータチューニングを
# 同時に、かつ公平（データリークなし）に行うための厳密な手法です。
# ------------------------------------------------------------------
class Leave_one_out:

    # ------------------------------------------------------------------
    # コンストラクタ
    # 検証に必要なデータとモデル、探索するパラメータを受け取ります。
    # ------------------------------------------------------------------
    # [引数]
    # X          : 特徴量行列（numpy array または pandas DataFrame）
    # Y          : 正解ラベル（numpy array または list）。例: [0, 1, 0, ...]
    # model      : 学習に使用する機械学習モデルのインスタンス（例: SVM, Random Forestなど）
    # param_grid : グリッドサーチで探索するハイパーパラメータの辞書
    #              例: {'C': [1, 10], 'kernel': ['linear', 'rbf']}
    def __init__(self, X, Y, model, param_grid):
        self.X = X
        self.Y = Y
        self.model = model
        self.param_grid = param_grid

    # ------------------------------------------------------------------
    # ダブル・リーブワンアウト実行のメインメソッド
    # 外側のループでテストデータを1つ確保し、内側のループで最適なパラメータを探します。
    # ------------------------------------------------------------------
    # [戻り値]
    # 辞書型（dict）: 以下のキーを含む
    #   - 'TN', 'FP', 'FN', 'TP' : 混同行列の各成分
    #   - 'recall'               : 再現率（感度）
    #   - 'f1_score'             : F1スコア
    #   - 'best_params_each_fold': 各試行で選ばれた最適パラメータのリスト
    #   - 'best_scores_each_fold': 内側の検証での最高スコアのリスト
    #   - 'train_accuracies'     : 外側の学習データに対する予測精度のリスト（過学習チェック用）
    #   - 'all_train_accuracies_100': 学習データでの精度が全て100%だったかどうかのフラグ
    def double_leave_one_out(self):
        # LeaveOneOut分割器のインスタンス化
        loo = LeaveOneOut()

        # 結果を保存するためのリストを初期化
        y_true = []  # 実際のラベル（正解）
        y_pred = []  # モデルによる予測ラベル
        best_params_list = []  # 各ループで選ばれたベストなパラメータ
        best_scores = []  # その時の内側での検証スコア
        train_accuracies = []  # 学習データ自身に対する精度の記録用

        # ----------------------------------------------------------
        # 【外側のループ】: 性能評価用
        # データの数(N)だけ繰り返します。
        # 毎回、1個をテスト用(test)、残り(N-1個)を学習用(train)に分けます。
        # ----------------------------------------------------------
        for train_idx, test_idx in loo.split(self.X):
            # インデックスを使ってデータを分割
            # X_train: N-1個の学習データ, X_test: 1個のテストデータ
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.Y[train_idx], self.Y[test_idx]

            # ------------------------------------------------------
            # 【内側のループ】: ハイパーパラメータチューニング用
            # 外側で作られた「学習用データ(X_train)」だけを使って、
            # さらにLeave-One-Outを行い、どのパラメータが良いかを探します。
            # これにより、テストデータ(X_test)の情報が学習に漏れるのを防ぎます。
            # ------------------------------------------------------
            grid_search = GridSearchCV(
                estimator=self.model,  # ベースとなるモデル
                param_grid=self.param_grid,  # 試行するパラメータの組み合わせ
                cv=loo,  # 内側でもLeave-One-Out分割を使用
                scoring='accuracy',  # 評価指標は正解率
                n_jobs=-1  # 並列処理ですべてのコアを使用（高速化）
            )

            # グリッドサーチの実行（内側の学習と評価）
            grid_search.fit(X_train, y_train)

            # ------------------------------------------------------
            # ベストモデルの再構築と最終予測
            # ------------------------------------------------------
            # 1. 内側のループで見つかった、最も性能が良かったパラメータを取得
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_

            # 2. モデルを複製(clone)し、見つかったベストパラメータをセットする
            #    cloneを使うのは、学習済みの重みをリセットし、純粋な未学習状態にするため
            best_model = clone(self.model).set_params(**best_params)

            # 3. 外側の学習データ(N-1個)すべてを使って、ベストな設定で再学習
            best_model.fit(X_train, y_train)

            # 4. 外側のテストデータ(取っておいた1個)に対して予測を行う
            pred = best_model.predict(X_test)

            # 5. 結果をリストに追加
            #    y_test, predは要素数1の配列なので、[0]で値を取り出す
            y_true.append(y_test[0])
            y_pred.append(pred[0])

            # 6. 分析用にパラメータやスコアも記録
            best_params_list.append(best_params)
            best_scores.append(best_score)

            # 7. 過学習の確認用：学習に使ったデータ自身(X_train)を予測させてみる
            #    もしこれが1.0（100%）より著しく低い場合、モデルが学習不足の可能性がある
            pred_train = best_model.predict(X_train)
            train_accuracy = accuracy_score(y_train, pred_train)
            train_accuracies.append(train_accuracy)

        # ----------------------------------------------------------
        # 全ループ終了後の集計
        # N回分の「正解」と「予測」が溜まったので、全体の精度を計算します。
        # ----------------------------------------------------------

        # 混同行列（Confusion Matrix）の作成
        cm = confusion_matrix(y_true, y_pred)

        # 行列の成分を展開して取得
        # TN: True Negative (実際0で予測も0)
        # FP: False Positive (実際0だが予測は1) -> 誤検知
        # FN: False Negative (実際1だが予測は0) -> 見逃し
        # TP: True Positive (実際1で予測も1)
        TN, FP, FN, TP = cm.ravel()

        # リコール（再現率）: 実際1のうち、どれだけ正しく1と予測できたか
        recall = recall_score(y_true, y_pred)

        # F1スコア: 適合率と再現率の調和平均（バランスの良い指標）
        f1 = f1_score(y_true, y_pred)

        # すべての試行において、学習データへの当てはまりが完璧(1.0)だったかチェック
        all_train_accuracies_100 = all(acc == 1.0 for acc in train_accuracies)

        # 計算結果をまとめて辞書として返す
        return {
            'TN': TN,
            'FP': FP,
            'FN': FN,
            'TP': TP,
            'recall': recall,
            'f1_score': f1,
            'best_params_each_fold': best_params_list,
            'best_scores_each_fold': best_scores,
            'train_accuracies': train_accuracies,
            'all_train_accuracies_100': all_train_accuracies_100
        }

    # ------------------------------------------------------------------
    # 結果表示用メソッド
    # double_leave_one_outを実行し、その結果を見やすくprint出力します。
    # ------------------------------------------------------------------
    # [引数] なし
    # [戻り値] なし（標準出力にprintするのみ）
    def leave_one_out_printout(self):
        # 計算を実行して結果辞書を取得
        result = self.double_leave_one_out()

        # 結果を整形して表示
        print("外側のleave-one-outの結果")
        print(f"実際詐欺じゃない、予測詐欺じゃない(TN): {result['TN']}")
        print(f"実際詐欺じゃない、予測詐欺(FP)      : {result['FP']}")
        print(f"実際詐欺、予測詐欺じゃない(FN)      : {result['FN']}")
        print(f"実際詐欺、予測詐欺(TP)              : {result['TP']}")
        print(f"Recall（リコール）                  : {result['recall']:.3f}")
        print(f"F1スコア                            : {result['f1_score']:.3f}")

        print("-" * 50)
        print(f"外側のleave-one-outの学習データへの精度リスト:")
        print(f"{result['train_accuracies']}")
        print(f"外側のleave-one-outの学習データの精度はすべて100%か: {result['all_train_accuracies_100']}")

        print("-" * 50)
        print(f"内側で選ばれたハイパーパラメータのリスト (全{len(result['best_params_each_fold'])}回):")
        print(result['best_params_each_fold'])

        print("内側のその時の検証スコア:")
        print(result['best_scores_each_fold'])
        print("")

        return