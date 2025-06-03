from gensim.models import Word2Vec
import pandas as pd

document = pd.read_csv("data.csv")
texts = document["News_Tokens"]
model = Word2Vec(sentences=texts, vector_size=100, window=5,min_count=1, sg=0)
model.save('word2vec_5_sgu.model')