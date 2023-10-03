from gensim.models import Word2Vec

loaded_model = Word2Vec.load('word2vec.model')

import os
import random

folder_path = r'C:\Users\AIMS Lab\Desktop\Region' 

txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

sentences = []

for txt_file in txt_files:
    file_path = os.path.join(folder_path, txt_file)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
        for line in lines:
            line = line.split(': ')[1].strip()
            words = line.split() 
            sentences.append(words)

random.shuffle(sentences)


for sentence in sentences:
    print(sentence)



loaded_model.build_vocab(sentences, update=True)  
loaded_model.train(sentences, total_examples=loaded_model.corpus_count, epochs=1)  

loaded_model.save('word2vec.model')
