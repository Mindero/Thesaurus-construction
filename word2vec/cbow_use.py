from gensim.models import Word2Vec
import pandas as pd

document = pd.read_excel("News_SGU_31077_Processed_1.xlsx")
texts = document["News_Tokens"].apply(lambda x: x.split(" ")).explode().unique()
model = Word2Vec.load("word2vec_cbow10_sgu.model")

results = []

for word in texts:
    try:
        # Получаем 10 ближайших слов
        similar_words = model.wv.most_similar(word, topn=10)
        results.append([word, similar_words])
    except KeyError:
        # Если слово не в модели, пропускаем его
        print(f"Слово '{word}' не найдено в модели.")
        continue

df_results = pd.DataFrame(results, columns=["Word", "Most_Similar_Word"])

df_results.to_csv("similar_words10.csv", index=False)