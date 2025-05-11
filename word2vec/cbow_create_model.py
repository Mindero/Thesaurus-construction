from gensim.models import Word2Vec
import pandas as pd

document = pd.read_excel("News_SGU_31077_Processed_1.xlsx")
texts = document["News_Tokens"].apply(lambda x: x.split(" "))

model = Word2Vec(sentences=texts, vector_size=100, window=10,min_count=1, sg=0)

model.save('word2vec_cbow10_sgu.model')
