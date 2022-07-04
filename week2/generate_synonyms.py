import fastext
import sys
import csv

print("User Current Version:-", sys.version)

model = fasttext.load_model('/workspace/datasets/fasttext/title_model.bin')

top_words_path = '/workspace/datasets/fasttext/top_words.txt'

with open(top_words_path, 'r') as f:
    words = f.readlines()

synonyms_list=[]
for word in words:
    # print(word)
    neighbors = model.get_nearest_neighbors(word.strip(), k=100)
    # print(neighbors)
    synonyms = [word.strip()]
    for n in neighbors:
        if n[0]>0.8:
            synonyms.append(n[1])
    if(len(synonyms)>1):
        synonyms_list.append(synonyms)

# print(synonyms_list)



with open("synonyms_data.csv", "w") as f:
    wr = csv.writer(f)
    wr.writerows(synonyms_list)
