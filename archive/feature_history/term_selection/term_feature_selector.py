import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.preprocessing import MinMaxScaler


# ------------------------------------------------------------------
# 文書ごとの単語出現状況（TF-IDFや頻度）を分析し、
# 詐欺（ラベル1）と正常（ラベル0）を区別する重要な単語を抽出・加工するクラス
# ------------------------------------------------------------------
class Find_terms:

    # ------------------------------------------------------------------
    # コンストラクタ（初期化メソッド）
    # クラスのインスタンスを作成した時点で、分析対象のデータを保持します。
    # ------------------------------------------------------------------
    # [引数]
    # X             : 文書×単語の行列（TF-IDF値や単語カウント）。(行数=文書数, 列数=単語数)
    #                 ※疎行列(sparse matrix)でも密行列(dense matrix)でも可。
    # labels        : 各文書の正解ラベルのリスト（例: [1, 0, 1, ...]）。1が詐欺、0が正常を想定。
    # feature_names : 行列Xの各列に対応する単語名のリスト（例: ['apple', 'banana', ...]）。
    def __init__(self, X, labels, feature_names):
        self.X = X
        self.labels = labels
        self.feature_names = feature_names

    # ------------------------------------------------------------------
    # データの分割メソッド
    # 全データを「ラベル1（詐欺）」群と「ラベル0（正常）」群に分割します。
    # ------------------------------------------------------------------
    # [引数]
    # なし（インスタンス変数のself.Xとself.labelsを使用）
    # [戻り値]
    # X_label1 : ラベル1に該当する文書行のみを抜き出した行列
    # X_label0 : ラベル0に該当する文書行のみを抜き出した行列
    def _label_split(self):
        # 1. ブールインデックス参照を使用してデータをフィルタリング
        #    self.labelsをnumpy配列化し、値が1の場所(True)に対応するXの行を抽出する
        X_label1 = self.X[np.array(self.labels) == 1]

        #    同様に、値が0の場所に対応するXの行を抽出する
        X_label0 = self.X[np.array(self.labels) == 0]

        # 2. 疎行列（sparse matrix）から密行列（dense array）への変換
        #    データがcsr_matrixなどの形式の場合、後の計算（スライスや判定）が複雑になるため、
        #    一般的なnumpy配列（ndarray）に変換する。
        #    ※ hasatttr で .toarray メソッドを持っているか確認してから実行
        if hasattr(X_label1, "toarray"):
            X_label1 = X_label1.toarray()
        if hasattr(X_label0, "toarray"):
            X_label0 = X_label0.toarray()

        return X_label1, X_label0

    # ------------------------------------------------------------------
    # 特徴語抽出: 「ラベル1には必ずあり、ラベル0には少ない」単語
    # 詐欺文書の特徴（必須キーワード）でありながら、一般文書にはあまり出ない単語を探す
    # ------------------------------------------------------------------
    # [引数]
    # top_n : ランキング上位何件の単語を抽出するか（int）
    # [戻り値]
    # top_terms : 抽出された単語情報のリスト（辞書のリスト）
    def all_label1_least_label0(self, top_n):
        # 1. データをラベルごとに分割して取得
        X_label1, X_label0 = self._label_split()

        # 2. ラベル1の文書すべて(100%)に出現している単語を探索
        terms_used_in_all_label1 = []
        for idx, term in enumerate(self.feature_names):
            # idx列目（ある単語）のデータを縦にすべて取得
            column = X_label1[:, idx]

            # np.all(column != 0) は「その列の全ての値が0以外か？」を判定
            # ひとつでも0（出現なし）があればFalseになる
            if np.all(column != 0):
                terms_used_in_all_label1.append((idx, term))

        # 3. 候補となった単語について、ラベル0での出現数をカウント
        term_info_list = []
        for idx, term in terms_used_in_all_label1:
            col_label1 = X_label1[:, idx]
            col_label0 = X_label0[:, idx]

            # np.sum(col != 0) で非ゼロの要素数（＝出現文書数）をカウント
            label1_count = np.sum(col_label1 != 0)
            label0_count = np.sum(col_label0 != 0)

            # 結果を辞書形式でリストに追加
            term_info_list.append({
                'term': term,
                'label1_count': label1_count,
                'label0_count': label0_count
            })

        # 4. ラベル0での出現数が少ない順（昇順）にソート
        #    key=lambda x: x['label0_count'] は、辞書の'label0_count'の値を基準にするという意味
        sorted_terms = sorted(term_info_list, key=lambda x: x['label0_count'])

        # 5. 上位 top_n 件をスライスで取得
        top_terms = sorted_terms[:top_n]

        print("ラベル１に出てくる単語の中でラベル0の出現が一番少ない単語ランキング")
        for i, term_info in enumerate(top_terms, 1):
            print(
                f"{i}位: 単語='{term_info['term']}', ラベル1出現数={term_info['label1_count']}, ラベル0出現数={term_info['label0_count']}")
        print("")
        return top_terms

    # ------------------------------------------------------------------
    # 特徴語抽出: 「ラベル1には全くなく、ラベル0には多い」単語
    # 「この単語があれば詐欺ではない可能性が高い」という安心キーワードを探す
    # ------------------------------------------------------------------
    # [引数]
    # top_n : 上位何件抽出するか
    # [戻り値]
    # top_terms : 抽出された単語情報のリスト。対象がない場合はNone。
    def not_label1_most_label0(self, top_n):
        X_label1, X_label0 = self._label_split()

        # 1. ラベル1で一度も出てこなかった単語（出現数0）をリストアップ
        terms_not_in_label1 = []
        for idx, term in enumerate(self.feature_names):
            column = X_label1[:, idx]
            # np.all(column == 0) は「その列の全ての値が0か？」を判定
            if np.all(column == 0):
                terms_not_in_label1.append((idx, term))

        # 2. それらの単語についてラベル0での出現数を数える
        term_info_list = []
        for idx, term in terms_not_in_label1:
            col_label0 = X_label0[:, idx]
            label0_count = np.sum(col_label0 != 0)

            term_info_list.append({
                'term': term,
                'label0_count': label0_count
            })

        # 対象となる単語が一つもなかった場合のガード処理
        if not term_info_list:
            return None

        # 3. ラベル0出現数が多い順（降順）にソート
        #    reverse=True を指定することで降順になる
        sorted_terms = sorted(term_info_list, key=lambda x: x['label0_count'], reverse=True)
        top_terms = sorted_terms[:top_n]

        print("ラベル１に出てこない単語の中でラベル0の出現が一番多い単語ランキング")
        if top_terms:
            for i, res in enumerate(top_terms, 1):
                print(f"{i}位: 単語='{res['term']}', ラベル0での出現文章数: {res['label0_count']}")
        else:
            print("条件を満たす単語がありませんでした。")
        print("")
        return top_terms

    # ------------------------------------------------------------------
    # 特徴語抽出: 「ラベル0に最もよく出る」単語
    # 詐欺かどうかに関わらず、正常な文書における頻出語を調べる
    # ------------------------------------------------------------------
    # [引数]
    # top_n : 上位何件抽出するか
    # [戻り値]
    # top_terms : 抽出された単語情報のリスト
    def most_label0(self, top_n):
        X_label1, X_label0 = self._label_split()

        # 1. 全ての単語について、ラベル0での出現数をカウント
        term_info_list = []
        for idx, term in enumerate(self.feature_names):
            col_label0 = X_label0[:, idx]
            label0_count = np.sum(col_label0 != 0)

            term_info_list.append({
                'term': term,
                'label0_count': label0_count
            })

        if not term_info_list:
            return None

            # 2. 出現数が多い順（降順）にソート
        sorted_terms = sorted(term_info_list, key=lambda x: x['label0_count'], reverse=True)
        top_terms = sorted_terms[:top_n]

        print("ラベル0の出現単語ランキング")
        for i, term_info in enumerate(top_terms, 1):
            print(f"{i}位: 単語='{term_info['term']}', ラベル0での出現文章数: {term_info['label0_count']}")
        print("")
        return top_terms

    # ------------------------------------------------------------------
    # 特徴語抽出: 「ラベル1に多く、ラベル0に少ない」単語
    # 最も典型的な「詐欺ワード」を探すロジック
    # ------------------------------------------------------------------
    # [引数]
    # top_n : 上位何件抽出するか
    # [戻り値]
    # top_terms : 抽出された単語情報のリスト
    def most_label1_least_label0(self, top_n):
        X_label1, X_label0 = self._label_split()
        term_info_list = []

        # 1. 全単語について、ラベル1とラベル0それぞれの出現数をカウント
        for idx, term in enumerate(self.feature_names):
            col_label1 = X_label1[:, idx]
            col_label0 = X_label0[:, idx]

            label1_count = np.sum(col_label1 != 0)
            label0_count = np.sum(col_label0 != 0)

            term_info_list.append({
                'term': term,
                'label1_count': label1_count,
                'label0_count': label0_count
            })

        if not term_info_list:
            return None

        # 2. 複合条件でのソートを実行
        #    Pythonのソートはタプルを渡すと、1要素目→2要素目の順で評価する
        #    要素1: -x['label1_count']
        #          マイナスをつけることで、値が大きいほど小さくなる → 昇順ソートすると実質「降順」になる
        #    要素2: x['label0_count']
        #          そのままなので「昇順」（少ないほうが優先）
        sorted_terms = sorted(
            term_info_list,
            key=lambda x: (-x['label1_count'], x['label0_count'])
        )

        top_terms = sorted_terms[:top_n]

        print("ラベル1の出現単語ランキング（ラベル0での出現数も表示）")
        for i, term_info in enumerate(top_terms, 1):
            print(
                f"{i}位: 単語='{term_info['term']}', ラベル1での出現文書数: {term_info['label1_count']}, ラベル0での出現文書数: {term_info['label0_count']}")
        print("")

        return top_terms

    # ------------------------------------------------------------------
    # 特徴語抽出: 「ラベル1に少なく、ラベル0に多い」単語
    # ------------------------------------------------------------------
    # [引数]
    # top_n : 上位何件抽出するか
    # [戻り値]
    # top_terms : 抽出された単語情報のリスト
    def least_label1_most_label0(self, top_n):
        X_label1, X_label0 = self._label_split()
        term_info_list = []

        # 1. 出現数のカウント
        for idx, term in enumerate(self.feature_names):
            col_label1 = X_label1[:, idx]
            col_label0 = X_label0[:, idx]

            label1_count = np.sum(col_label1 != 0)
            label0_count = np.sum(col_label0 != 0)

            term_info_list.append({
                'term': term,
                'label1_count': label1_count,
                'label0_count': label0_count
            })

        if not term_info_list:
            return None

        # 2. 複合条件でのソート
        #    要素1: x['label1_count'] (昇順 = 少ない順)
        #    要素2: -x['label0_count'] (実質降順 = 多い順)
        sorted_terms = sorted(
            term_info_list,
            key=lambda x: (x['label1_count'], -x['label0_count'])
        )

        top_terms = sorted_terms[:top_n]

        print("ラベル1に少なく出現し、ラベル0に多く出現する単語ランキング")
        for i, term_info in enumerate(top_terms, 1):
            print(
                f"{i}位: 単語='{term_info['term']}', ラベル1での出現文書数: {term_info['label1_count']}, ラベル0での出現文書数: {term_info['label0_count']}")
        print("")

        return top_terms

    # ------------------------------------------------------------------
    # 特徴量生成: 抽出された特定単語を「バイナリ変数」にして返す
    # TF-IDF値などの「重み」を無視し、単語が「ある(1)か ない(0)か」だけのDataFrameを作成します。
    # ------------------------------------------------------------------
    # [引数]
    # ranked_terms : 抽出したい単語のリスト。
    #                {'term': 'word'} 形式の辞書リスト、または単純な文字列リスト ['word1', 'word2'] のどちらでも可。
    # [戻り値]
    # df_binary : 指定単語の有無を表すDataFrame（行：文書、列：指定単語、値：0か1）
    def create_topn_binary_fueature(self, ranked_terms):

        # 1. 元のデータが疎行列(sparse)なら、計算のために密行列(dense)へ変換
        if hasattr(self.X, "toarray"):
            self.X = self.X.toarray()

        # 2. 引数の型チェック：辞書のリストなら単語名を取り出し、文字列リストならそのまま使う
        if isinstance(ranked_terms[0], dict):
            terms = [t['term'] for t in ranked_terms]
        else:
            terms = ranked_terms

        # 3. 単語名から列番号(index)を高速に検索するための辞書を作成
        #    { 'apple': 0, 'banana': 1, ... } のような形
        term_indices = {term: idx for idx, term in enumerate(self.feature_names)}

        binary_features = []

        # 4. 指定された各単語についてループ処理
        for term in terms:
            idx = term_indices.get(term)
            if idx is None:
                raise ValueError(f"単語 '{term}' がfeature_namesに存在しません。")

            # 5. その単語の列を取り出し、値が0でなければ1、0なら0に変換
            #    (self.X[:, idx] != 0) はTrue/Falseの配列になる
            #    .astype(int) で True→1, False→0 に変換される
            binary_feature = (self.X[:, idx] != 0).astype(int)
            binary_features.append(binary_feature)

        # 6. 配列を転置してDataFrame化
        #    現在の binary_features は (単語数, 文書数) の形なので、
        #    .T で転置して (文書数, 単語数) の形にする
        df_binary = pd.DataFrame(np.array(binary_features).T, columns=terms)

        return df_binary

    # ------------------------------------------------------------------
    # 特徴量抽出: 特定単語の元の値（TF-IDF値など）をそのまま抽出する
    # ------------------------------------------------------------------
    # [引数]
    # term_infos : 抽出したい単語情報のリスト（辞書リスト）。{'term': 'word', ...} の形式。
    # [戻り値]
    # df_terms : 指定単語のTF-IDF値などが格納されたDataFrame
    def extract_terms_features(self, term_infos):
        # 1. 単語名のリストを作成
        terms = [term_info['term'] for term_info in term_infos]

        # 2. 単語名に対応する列インデックスを検索
        term_indices = []
        missing_terms = []
        for term in terms:
            # np.where でfeature_namesの中から該当単語の位置を探す
            indices = np.where(self.feature_names == term)[0]
            if len(indices) == 0:
                missing_terms.append(term)
            else:
                term_indices.append(indices[0])

        if missing_terms:
            raise ValueError(f"指定した単語 '{missing_terms}' は特徴量に存在しません。")

        # 3. 指定された列のみを行列Xから抜き出す
        X_terms = self.X[:, term_indices]

        # 4. 疎行列なら配列に変換
        if issparse(X_terms):
            X_terms = X_terms.toarray()

        # 5. 結果をDataFrameにして返す
        df_terms = pd.DataFrame(X_terms, columns=terms)

        return df_terms

    # ------------------------------------------------------------------
    # 特徴量抽出 & 正規化: 特定単語の値を抽出し、列ごとに0〜1に正規化する
    # 単語ごとの値のスケールを揃えたい場合に使用（例: 機械学習モデルへの入力用）
    # ------------------------------------------------------------------
    # [引数]
    # term_infos : 抽出したい単語情報のリスト（辞書リスト）。
    # [戻り値]
    # df_terms : 正規化された値を持つDataFrame
    def extract_terms_features_normalization(self, term_infos):
        # 1. 単語名の取得
        terms = [term_info['term'] for term_info in term_infos]

        # 2. インデックス検索
        term_indices = []
        missing_terms = []
        for term in terms:
            indices = np.where(self.feature_names == term)[0]
            if len(indices) == 0:
                missing_terms.append(term)
            else:
                term_indices.append(indices[0])

        if missing_terms:
            raise ValueError(f"指定した単語 '{missing_terms}' は特徴量に存在しません。")

        # 3. 該当列の抽出
        X_terms = self.X[:, term_indices]

        if issparse(X_terms):
            X_terms = X_terms.toarray()

        # 4. 列ごとのMin-Max正規化を実行
        #    各列の中で 最大値=1, 最小値=0 になるように変換
        scaler = MinMaxScaler()
        X_terms_scaled = scaler.fit_transform(X_terms)

        df_terms = pd.DataFrame(X_terms_scaled, columns=terms)

        return df_terms

    # ------------------------------------------------------------------
    # 特徴量抽出 & 行正規化: 特定単語を抽出し、行ごと（文書ごと）に正規化する
    # 抽出された特定の単語群の中での、相対的な強さを見たい場合などに使用
    # ------------------------------------------------------------------
    # [引数]
    # df : 列名を取得するためのDataFrame（対象とする単語が列名になっているもの）
    #      ※実際の数値データはクラス内のself.Xから再取得されます。
    # [戻り値]
    # df_terms : 行単位で正規化されたDataFrame
    def extract_terms_features_row_normalization(self, df):
        # 1. 入力DataFrameの列名を、抽出対象の単語リストとする
        terms = df.columns.tolist()

        # 2. インデックス検索
        term_indices = []
        missing_terms = []
        for term in terms:
            indices = np.where(self.feature_names == term)[0]
            if len(indices) == 0:
                missing_terms.append(term)
            else:
                term_indices.append(indices[0])

        if missing_terms:
            raise ValueError(f"指定した単語 '{missing_terms}' は特徴量に存在しません。")

        # 3. 再度、元のXからデータを抽出
        X_terms = self.X[:, term_indices]
        if issparse(X_terms):
            X_terms = X_terms.toarray()

        # 4. 行方向（axis=1）での最大値・最小値を計算
        min_vals = X_terms.min(axis=1, keepdims=True)
        max_vals = X_terms.max(axis=1, keepdims=True)
        range_vals = max_vals - min_vals

        # 5. 行ごとのMin-Max正規化を手動計算
        #    np.errstate: ゼロ除算（range_valsが0の場合）のエラー警告を無視するブロック
        with np.errstate(divide='ignore', invalid='ignore'):
            X_terms_scaled = (X_terms - min_vals) / range_vals
            # 最大値と最小値が同じ場合(range=0)は計算結果がNaNになるので、0で埋める
            X_terms_scaled[np.isnan(X_terms_scaled)] = 0.0

        df_terms = pd.DataFrame(X_terms_scaled, columns=terms)
        return df_terms

    # ------------------------------------------------------------------
    # 割合スコア算出: 「ラベル1での出現率 - ラベル0での出現率」が高い単語を探す
    # 文書数の偏り（ラベル1が極端に少ない等）がある場合に、絶対数ではなく「率」で比較するメソッド
    # ------------------------------------------------------------------
    # [引数]
    # top_n : 上位何件の単語を抽出するか（int）
    # [戻り値]
    # top_terms : 抽出された単語情報のリスト（スコア情報を含む）
    def rate_label1_minus_label0(self, top_n):
        X_label1, X_label0 = self._label_split()

        # 1. 各ラベルの総文書数（分母）を取得
        n_label1 = X_label1.shape[0]
        n_label0 = X_label0.shape[0]

        term_info_list = []
        for idx, term in enumerate(self.feature_names):
            col_label0 = X_label0[:, idx]
            col_label1 = X_label1[:, idx]

            # 2. 単語ごとの出現文書数をカウント
            label0_count = np.sum(col_label0 != 0)
            label1_count = np.sum(col_label1 != 0)

            # 3. 出現率（Rate）の計算： 出現数 / 全文書数
            #    0除算を防ぐため、分母が0より大きい場合のみ計算
            rate_label1 = label1_count / n_label1 if n_label1 > 0 else 0.0
            rate_label0 = label0_count / n_label0 if n_label0 > 0 else 0.0

            # 4. スコア計算： ラベル1の率 - ラベル0の率
            #    プラスが大きいほど「詐欺文書(1)に偏って出現する」ことを示す
            score = rate_label1 - rate_label0

            term_info_list.append({
                'term': term,
                'label0_count': label0_count,
                'label1_count': label1_count,
                'rate_label1': rate_label1,
                'rate_label0': rate_label0,
                'score': score
            })

        # 5. スコアが大きい順（降順）にソート
        sorted_terms = sorted(term_info_list, key=lambda x: x['score'], reverse=True)

        # 6. 上位top_n件を抽出
        top_terms = sorted_terms[:top_n]

        print(f"スコアが高い上位 {top_n} 単語")
        for i, term_info in enumerate(top_terms, 1):
            print(f"{i}位: 単語='{term_info['term']}', "
                  f"スコア: {term_info['score']:.4f} "
                  f"(ラベル1: {term_info['rate_label1']:.2%}, ラベル0: {term_info['rate_label0']:.2%})")
        print("")

        return top_terms

    # ------------------------------------------------------------------
    # 割合スコア算出: 「ラベル0での出現率 - ラベル1での出現率」が高い単語を探す
    # 上記の逆バージョン（正常文書に特有の単語を探す）
    # ------------------------------------------------------------------
    # [引数]
    # top_n : 上位何件の単語を抽出するか（int）
    # [戻り値]
    # top_terms : 抽出された単語情報のリスト
    def rate_label0_minus_label1(self, top_n):
        X_label1, X_label0 = self._label_split()
        n_label1 = X_label1.shape[0]
        n_label0 = X_label0.shape[0]

        term_info_list = []
        for idx, term in enumerate(self.feature_names):
            col_label0 = X_label0[:, idx]
            col_label1 = X_label1[:, idx]

            label0_count = np.sum(col_label0 != 0)
            label1_count = np.sum(col_label1 != 0)

            rate_label0 = label0_count / n_label0 if n_label0 > 0 else 0.0
            rate_label1 = label1_count / n_label1 if n_label1 > 0 else 0.0

            # 4. スコア計算： ラベル0の率 - ラベル1の率
            #    プラスが大きいほど「正常文書(0)に偏って出現する」ことを示す
            score = rate_label0 - rate_label1

            term_info_list.append({
                'term': term,
                'label0_count': label0_count,
                'label1_count': label1_count,
                'rate_label0': rate_label0,
                'rate_label1': rate_label1,
                'score': score
            })

        sorted_terms = sorted(term_info_list, key=lambda x: x['score'], reverse=True)
        top_terms = sorted_terms[:top_n]

        print(f"ラベル0偏りスコアが高い上位 {top_n} 単語")
        for i, term_info in enumerate(top_terms, 1):
            print(f"{i}位: 単語='{term_info['term']}', "
                  f"スコア: {term_info['score']:.4f} "
                  f"(ラベル0: {term_info['rate_label0']:.2%}, ラベル1: {term_info['rate_label1']:.2%})")
        print("")

        return top_terms