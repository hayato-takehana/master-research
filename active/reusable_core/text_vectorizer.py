import gensim
from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS
from gensim.parsing.preprocessing import stem_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

# TF-IDF計算や前処理を行うためのクラス定義
class Tf_idf:
    # ------------------------------------------------------------------
    # コンストラクタ（初期化メソッド）
    # クラスのインスタンスを作成した時点で自動的に呼び出され、設定の保存と前処理を実行します。
    # ------------------------------------------------------------------
    # [引数]
    # documents : 分析対象となる「文書のリスト」（例: ["This is a pen.", "I love AI."]）
    # deacc     : Trueの場合、アクセント記号を除去する（例: "résumé" → "resume"）
    # min_len      : この文字数以下の短い単語はノイズとして削除する（例: 2なら "a" や "is" などが消える可能性がある）
    # use_stemming : True の場合はステミングを適用する
    def __init__(self, documents, deacc, min_len, use_stemming=False):
        # 引数で受け取った設定値をクラス内の変数（インスタンス変数）として保存
        self.documents = documents
        self.deacc = deacc
        self.min_len = min_len
        self.use_stemming = use_stemming

        # sklearnライブラリに含まれる一般的な英語のストップワード（"the", "is", "at" などの機能語）を読み込む
        self.stop_word = ENGLISH_STOP_WORDS

        # 初期化の段階で、全ての文書に対して前処理（クリーニング）を実行し、結果を保存しておく
        # preprocess()メソッドを呼び出している
        self.processed_docs = self.preprocess()

    #片方の文書に対して、その文書を１として、それ以外の文書を0としてラベル化する
    # [引数]
    # document_sagi : 詐欺文書（ターゲット）のみが含まれるリスト
    # [戻り値]
    # ラベルのリスト（例: [1, 1, 1, 0, 0, 0]）
    def labels(self, document_sagi):
        return [1] * len(document_sagi) + [0] * (len(self.documents) - len(document_sagi))

    # docは文書の1つ。self.documentのうちの1つ
    def _preprocess_doc(self, doc):
        #以下処理でトークン化、小文字化、アクセント記号の変更、文字数未満の単語の排除が行われる
        #トークン化とは、入力された文章を個々の単語に分割すること
        tokens = gensim.utils.simple_preprocess(doc, deacc=self.deacc, min_len=self.min_len)

        not_stop_word_tokens = [token for token in tokens if token not in self.stop_word]
        # 各単語で構成されている配列を1つの文字列に戻す
        if self.use_stemming:
            processed_tokens = [stem_text(token) for token in not_stop_word_tokens]
        else:
            processed_tokens = not_stop_word_tokens

        return ' '.join(processed_tokens)

    #全ての文書に対して、前処理を実行する
    def preprocess(self):
        #すべての文章に対して、前処理を実行
        return [self._preprocess_doc(doc) for doc in self.documents]

    # ------------------------------------------------------------------
    # TF-IDFベクトルの生成メソッド
    # 前処理済みの文書群から、TF-IDF行列を作成します。
    # ------------------------------------------------------------------
    # [引数]
    # min_df : 文書頻度の最小値（これより少ない文書にしか登場しないレアな単語は無視する）
    # [戻り値]
    # X             : TF-IDF行列（数値データ）
    # feature_names : 行列の各列に対応する単語の名前リスト
    # vectorizer    : 学習済みのVectorizerオブジェクト（後で再利用可能）
    def tf_idf(self, min_df, ngram_range=(1, 1)):
        # 下で定義されているVectorizerメソッドを使い、設定済みのTfidfVectorizerインスタンスを取得
        vectorizer = self._Vectorizer(min_df, ngram_range=ngram_range)

        # fit_transformを実行
        # fit      : どのような単語があるか（語彙）を学習する
        # transform: 各文書をTF-IDF値に基づいたベクトル（数値の列）に変換する
        X = vectorizer.fit_transform(self.processed_docs)

        # ベクトルの各列がどの単語を表しているか（特徴語）を取得する
        feature_names = vectorizer.get_feature_names_out()

        return X,feature_names,vectorizer

    # ------------------------------------------------------------------
    # Vectorizerの設定メソッド
    # TfidfVectorizerのインスタンスを作成・設定するためのヘルパー関数です。
    # ------------------------------------------------------------------
    # [引数]
    # min_df : tf_idfメソッドから渡される、最小文書頻度の設定値
    # [戻り値]
    # 設定済みのTfidfVectorizerオブジェクト
    def _Vectorizer(self, min_df, ngram_range=(1, 1)):
        vectorizer = TfidfVectorizer(
            # ストップワードの除去はすでにpreprocess_docで行っているため、ここではNone（何もしない）にする
            stop_words=None,
            # 指定された頻度以下の単語を無視する設定
            min_df=min_df,
            ngram_range=ngram_range,
        )

        return vectorizer

    # ------------------------------------------------------------------
    # 単語頻度（Term Frequency）のみを計算するメソッド
    # TF-IDFの重み付けを行わず、単純な「単語の出現回数」をベクトル化します。
    # ------------------------------------------------------------------
    # [引数]
    # min_df : デフォルトは1（すべての単語をカウント対象とする）
    def term_frequency(self, min_df=1, ngram_range=(1, 5)):
        # カウントベースの設定でVectorizerを初期化
        vectorizer = TfidfVectorizer(
            stop_words=None,
            min_df=min_df,
            # IDF（逆文書頻度）の計算を行わない -> 単純なTFのみになる
            use_idf=False,
            # 正規化（ベクトルの長さを1にする処理）を行わない -> 生の出現回数がそのまま値になる
            norm=None,
            ngram_range = ngram_range
        )

        # 前処理済みの文書を使って、出現回数行列Xを作成
        X = vectorizer.fit_transform(self.processed_docs)

        # 特徴語（単語リスト）の取得
        feature_names = vectorizer.get_feature_names_out()

        return X, feature_names, vectorizer
