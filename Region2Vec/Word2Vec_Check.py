from gensim.models import Word2Vec
from gensim.models import KeyedVectors

loaded_model = Word2Vec.load('word2vec.model')

vocabulary = list(loaded_model.wv.index_to_key)

#어휘에 포함된 단어 수 출력
#print("총 단어 수:", len(vocabulary))

# 처음 몇 개의 단어 출력
#print("총 단어:", vocabulary[:2538])

a = loaded_model.wv['북단']
#print(a)

#print(loaded_model.wv)

for word in vocabulary:
    vector = loaded_model.wv[word]
    print(f"단어: {word}, 벡터: {vector}")