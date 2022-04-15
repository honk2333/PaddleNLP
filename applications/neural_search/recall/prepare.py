import numpy as np
import pandas as pd
import os

path = '/home/data_ti6_c/wanghk/work_data/'
corpus_data = pd.read_csv( path + "corpus.tsv", sep="\t", names=["doc", "title"])
test_data = pd.read_csv(path + "dev.query.txt", sep="\t", names=["query", "title"])
train_data = pd.read_csv(path + "train.query.txt", sep="\t", names=["query", "title"])
qrels = pd.read_csv(path + "qrels.train.tsv", sep="\t", names=["query", "doc"])

corpus_data = corpus_data.set_index("doc")
test_data = test_data.set_index("query")
train_data = train_data.set_index("query")
qrels = qrels.set_index("query")

# for index, qrel in qrels.iterrows():
#     print(index,qrel['doc'])
# corpus_data['title'].to_csv('./train_unsupervised.csv',sep='\t',index=False)
l = [(train_data.loc[index]['title'],corpus_data.loc[qrel['doc']]['title']) for index, qrel in qrels.iterrows()]
print(l[:5])
import random
train_num = (int)(len(l)*0.7)
print(type(l),len(l))
random.shuffle(l)
train_dateset = l[:train_num]
dev_dataset = l[train_num:]
print(len(train_dateset),len(dev_dataset))
import csv
with open('./dev.csv','w') as f:
    writer = csv.writer(f,delimiter='\t')
    writer.writerows(dev_dataset)
with open('./train.csv','w') as f:
    writer = csv.writer(f,delimiter='\t')
    writer.writerows(train_dateset)
