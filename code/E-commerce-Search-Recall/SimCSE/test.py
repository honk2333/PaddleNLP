import os
import random
from turtle import pos
import torch.utils.data as data
from transformers import AutoTokenizer
import pandas as pd

# tokenizer = AutoTokenizer.from_pretrained(
#             '/home/data_ti6_c/wanghk/bert_model/chinese-roberta-wwm-ext')
# text = '网易严选N620紫外线除螨吸尘器家用小型强力吸尘器'
# sample = tokenizer(text,
#                   truncation=True,
#                   add_special_tokens=True,
#                   max_length=64,
#                   padding='max_length',
#                   return_tensors='pt')
# print(sample)
# print(type(sample))


# path = '/home/data_ti6_c/wanghk/work_data/'
path = '../../data/'
# train_data = pd.read_csv(path + "train.query.txt", sep="\t", names=["query", "title"])
corpus_data = pd.read_csv( path + "corpus_cl.tsv", sep="\t", names=["doc", "title"])
query_data = pd.read_csv(path+'train_query_cl.tsv', sep='\t', names=['query','title'])
train_data = pd.read_csv(path+'train_qrels', sep='\t', names=['query','doc'])
train_data = train_data.set_index('query')
query_data = query_data.set_index('query')[:20]
corpus_data = corpus_data.set_index("doc")

# tokenizer = AutoTokenizer.from_pretrained(
#             '/home/data_ti6_c/wanghk/bert_model/chinese-roberta-wwm-ext')
for index, row in query_data.iterrows():
    title = row['title']
    query = index
    if query not in train_data.index:
        continue
    doc = train_data.loc[index]['doc']
    print(title, "|" , corpus_data.loc[doc]['title'])

    a = tokenizer.tokenize(title)
    b = tokenizer.tokenize(corpus_data.loc[doc]['title'])
    print(a,"|",b)


# print(corpus_data)
# print(corpus_data.iloc[1001500])
# item = train_data.loc[100]
# print(item)

# print(train_data.item)
# print(item['title'])

# ls = [len(row['title'])for idx, row in corpus_data.iterrows()]
# ls.sort()
# ld = {}
# less48 = 0
# for i in ls:
#     if i in ld.keys():
#         ld[i]+=1
#     else:
#         ld[i] =1
#     if i <= 50:
#         less48 +=1
# print(ld)
# print(less48,less48/len(corpus_data))
# import matplotlib.pyplot as plt
# plt.bar(ld.keys(), ld.values())
# plt.savefig('corpuslen.png')


# l = list(range(0,10))
# print(l.index(5))


# import  torch
# from torch.nn import functional as F
# from torch.utils.data import DataLoader
# import numpy as np
#
# from zhconv import convert
# # 繁体转简体
#
# res_1 = convert('我幹什麼不干你事。', 'zh-cn')

# def sup_loss(y_pred, lamda=0.05):
#     row = torch.arange(0, y_pred.shape[0], 3, device="cuda")
#     col = torch.arange(y_pred.shape[0], device="cuda")
#     col = torch.where(col % 3 != 0)[0].cuda()
#     print(col)
#     y_true = torch.arange(0, len(col), 2, device="cuda")
#     print(y_true)
#     print(y_true.shape)
#     print(y_pred.unsqueeze(1).shape,y_pred.unsqueeze(0).shape)
#     similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
#     print(similarities.shape)
#     similarities = torch.index_select(similarities, 0, row)
#     print(similarities.shape)
#     similarities = torch.index_select(similarities, 1, col)
#     print(similarities.shape)
#
#     similarities = similarities / lamda
#
#     loss = F.cross_entropy(similarities, y_true)
#     return torch.mean(loss)
#
# def unsup_loss(y_pred, lamda=0.05):
#     idxs = torch.arange(0, y_pred.shape[0], device="cuda")
#     y_true = idxs + 1 - idxs % 2 * 2
#     similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
#
#     similarities = similarities - torch.eye(y_pred.shape[0], device="cuda") * 1e12
#
#     similarities = similarities / lamda
#     print(y_true)
#     print(similarities.shape)
#     loss = F.cross_entropy(similarities, y_true)
#     return torch.mean(loss)
#
# y_pred = torch.ones([64*3,128],dtype=torch.float64,device='cuda')
# print(y_pred.shape)
# sup_loss(y_pred)