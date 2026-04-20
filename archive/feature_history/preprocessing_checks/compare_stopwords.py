from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stopwords
from gensim.parsing.preprocessing import STOPWORDS as gensim_stopwords

# 1. 単語数を確認する
print(f"Sklearnの単語: {list(sklearn_stopwords)}")
print(f"Gensimの単語: {list(gensim_stopwords)}")
print(f"Sklearnの単語数: {len(sklearn_stopwords)}")
print(f"Gensimの単語数: {len(gensim_stopwords)}")

print("-" * 20)

# 2. Gensimにしか含まれていない単語（一部）
# (gensim_stopwords から sklearn_stopwords を引く)
gensim_only = gensim_stopwords.difference(sklearn_stopwords)
print(f"Gensimのみに含まれる単語 (例): {list(gensim_only)}")
print(f"Gensimのみの単語数: {len(gensim_only)}")


print("-" * 20)

# 3. Sklearnにしか含まれていない単語（一部）
# (sklearn_stopwords から gensim_stopwords を引く)
sklearn_only = sklearn_stopwords.difference(gensim_stopwords)
print(f"Sklearnのみに含まれる単語 (例): {list(sklearn_only)[:10]}")
print(f"Sklearnのみの単語数: {len(sklearn_only)}")