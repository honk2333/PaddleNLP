import os
import random
from turtle import pos
import torch.utils.data as data
from transformers import AutoTokenizer
import pandas as pd

# tokenizer = AutoTokenizer.from_pretrained(
#             '/home/data_ti6_c/wanghk/bert_model/chinese-roberta-wwm-ext')
# print(tokenizer)

path = '../data/'
corpus_data = pd.read_csv(path + "corpus_cl.tsv", sep="\t", names=["doc", "title"])
query_data = pd.read_csv(path + 'train_query_cl.tsv', sep='\t', names=['query','title'])
test_query_data = pd.read_csv(path + 'test_query_cl.tsv', sep='\t', names=['query','title'])
# train_data = pd.read_csv(path + 'train_qrels', sep='\t', names=['query','doc'])
# train_data = train_data.set_index('query')
query_data = query_data.set_index('query')
corpus_data = corpus_data.set_index("doc")
test_query_data = test_query_data.set_index('query')

# 准备语料
print(len(query_data['title']))
data = pd.concat([query_data['title'], corpus_data['title'], test_query_data['title']])
print(len(data))
pretrain_data = data.sample(n=None, frac=0.8, replace=False, weights=None, random_state=None, axis=None)
pretrain_evaldata = data[~data.index.isin(pretrain_data.index)]
print(len(pretrain_data), len(pretrain_evaldata))
pretrain_data.to_csv(path + 'pretrain_data.txt', sep='\t', index=False, header=False)
pretrain_evaldata.to_csv(path + 'pretrain_evaldata.txt', sep='\t', index=False, header=False)
# from tokenizers import ByteLevelBPETokenizer
#
# # Initialize a tokenizer
# tokenizer = ByteLevelBPETokenizer()
# # Customize training
# tokenizer.train(files=paths, vocab_size=52_000, min_frequency=1)
