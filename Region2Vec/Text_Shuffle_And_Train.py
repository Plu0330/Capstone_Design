import os
import random
from gensim.models import Word2Vec
from tqdm import tqdm

folder_path = r'C:\Users\AIMS Lab\Desktop\sample5' 

txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

sentences = []


for txt_file in txt_files:
    file_path = os.path.join(folder_path, txt_file)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
        for line in lines:
            line = line.strip()
            
            
            words = line.split()  
            sentences.append(words)

random.shuffle(sentences)

# for sentence in sentences:
#    print(sentence)


model = Word2Vec(sentences, vector_size=256, window=3, min_count=1, sg=1, epochs=1)

for epoch in tqdm(range(1), desc="Training Word2Vec", unit="epoch"):
    model.build_vocab(sentences, progress_per=1000)  
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)  


model.save('word2vec.model')


