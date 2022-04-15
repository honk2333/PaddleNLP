import numpy as np
import pandas as pd
import os

# path = '/home/data_ti6_c/wanghk/work_data/'
#
# qrels = pd.read_csv(path + "qrels.train.tsv", sep="\t", names=["query", "doc"])
# qrels = qrels.set_index("query")
#
# train_qrels = qrels.sample(n=len(qrels)-2000,axis=0)
# print(len(train_qrels))
# print(train_qrels.head())
# dev_qrels = qrels[~qrels.index.isin(train_qrels.index)]
# print(len(dev_qrels))
# print(dev_qrels.head())
#
# train_qrels.to_csv('./train_qrels',sep="\t",header=False)
# dev_qrels.to_csv('./dev_qrels',sep="\t",header=False)


path = '/home/data_ti6_c/wanghk/work_data/'
corpus_data = pd.read_csv( path + "corpus.tsv", sep="\t", names=["doc", "title"])
train_qrels = pd.read_csv('./train_qrels', sep="\t", names=["query", "doc"])
dev_qrels = pd.read_csv('./dev_qrels', sep="\t", names=["query", "doc"])

corpus_data = corpus_data.set_index("doc")
train_qrels = train_qrels.set_index("query")
dev_qrels = dev_qrels.set_index('query')

print(corpus_data)

dev_corpus = {"doc":[],"title":[]}
for index, row in dev_qrels.iterrows():
    dev_corpus["doc"].append(row['doc'])
    dev_corpus["title"].append(corpus_data.loc[row['doc']]['title'])
print(len(dev_corpus['doc']),len(dev_corpus['title']))
tmp = corpus_data[ ~corpus_data.index.isin(dev_corpus['doc']) ]
print(len(tmp))
import random
get_tmp = tmp.sample(100000-len(dev_corpus['doc']))
print(get_tmp)

for index, row in get_tmp.iterrows():
    dev_corpus["doc"].append(index)
    dev_corpus["title"].append(row['title'])
print(len(dev_corpus['doc']), len(dev_corpus['title']))

pd.DataFrame(dev_corpus).to_csv('./dev_corpus',sep="\t",header=False,index=False)

