import os
import random
from turtle import pos
import torch.utils.data as data
from transformers import AutoTokenizer
import pandas as pd
import  csv

class Unsupervised(data.Dataset):
    def __init__(self) -> None:
        super(Unsupervised, self).__init__()
        self.root = '../../data/'
        self.all = os.path.join(self.root, "corpus_cl.tsv")
        self.all_data = pd.read_csv(self.all, sep='\t',names=['doc','title'], quoting=csv.QUOTE_NONE)
        self.all_data = self.all_data.set_index('doc')
        # print(len(self.all_data))
        # print(self.all_data)
        self.tokenizer = AutoTokenizer.from_pretrained(
            '/home/data_ti6_c/wanghk/bert_model/chinese-roberta-wwm-ext')

    def __getitem__(self, index):
        text = self.all_data.iloc[index]['title']
        sample = self.tokenizer([text, text],
                                truncation=True,
                                add_special_tokens=True,
                                max_length=64,
                                padding='max_length',
                                return_tensors='pt')
        return sample

    def __len__(self):
        return len(self.all_data)

class Evaluated(data.Dataset):
    def __init__(self) -> None:
        super(Evaluated, self).__init__()
        self.root = '../../data/'
        self.all = os.path.join(self.root, "dev_corpus_cl.tsv")
        self.train = os.path.join(self.root, "train_query_cl.tsv")
        self.corr = os.path.join(self.root, "dev_qrels")

        self.all_data = pd.read_csv(self.all, sep='\t', names=['doc', 'title'],quoting=csv.QUOTE_NONE)
        self.all_data = self.all_data.set_index('doc')
        self.train_data = pd.read_csv(self.train, sep='\t', names=['query', 'title'])
        self.train_data = self.train_data.set_index('query')
        self.corr_data = pd.read_csv(self.corr, sep='\t', names=['query', 'doc'])
        self.corr_data = self.corr_data.set_index('query')
        self.tokenizer = AutoTokenizer.from_pretrained(
            '/home/data_ti6_c/wanghk/bert_model/chinese-roberta-wwm-ext')


    def __getitem__(self, index):
        item = self.corr_data.iloc[index]
        query = item.name
        doc = item['doc']
        anchor_text = self.train_data.loc[query]['title']
        sample = self.tokenizer(anchor_text,
                                truncation=True,
                                add_special_tokens=True,
                                max_length=64,
                                padding='max_length',
                                return_tensors='pt').to("cuda")
        return sample, doc


    def __len__(self):
        return len(self.corr_data)


class Supervised(data.Dataset):
    def __init__(self) -> None:
        super(Supervised, self).__init__()
        self.root = '../../data/'
        self.all = os.path.join(self.root, "corpus_cl.tsv")
        self.train = os.path.join(self.root, "train_query_cl.tsv")
        self.corr = os.path.join(self.root, "train_qrels")

        self.all_data = pd.read_csv(self.all, sep='\t', names=['doc','title'],quoting=csv.QUOTE_NONE)
        self.all_data = self.all_data.set_index('doc')
        self.train_data = pd.read_csv(self.train, sep='\t', names=['query','title'])
        self.train_data = self.train_data.set_index('query')
        self.corr_data = pd.read_csv(self.corr, sep='\t', names=['query','doc'])
        self.corr_data = self.corr_data.set_index('query')

        self.tokenizer = AutoTokenizer.from_pretrained(
            '/home/data_ti6_c/wanghk/bert_model/chinese-roberta-wwm-ext')


    def __getitem__(self, index):
        item = self.corr_data.iloc[index]
        query = item.name
        doc = item['doc']
        anchor_text, pos_text = self.train_data.loc[query]['title'],self.all_data.loc[doc]['title']
        # print(anchor_text, pos_text)
        tmp = random.randint(0, 1001500-1)
        neg_text = self.all_data.iloc[tmp]['title']
        sample = self.tokenizer([anchor_text, pos_text, neg_text],
                                truncation=True,
                                add_special_tokens=True,
                                max_length=64,
                                padding='max_length',
                                return_tensors='pt').to("cuda")
        return sample


    def __len__(self):
        return len(self.corr_data)


class TESTDATA(data.Dataset):
    def __init__(self, certain="corpus.tsv") -> None:
        super(TESTDATA, self).__init__()
        self.root = '../../data/'
        self.all = os.path.join(self.root, certain)
        self.all_data = pd.read_csv(self.all, sep='\t', names=['a', 'b'])
        self.all_data = self.all_data.set_index('a')
        self.length = 64
        self.tokenizer = AutoTokenizer.from_pretrained(
            '/home/data_ti6_c/wanghk/bert_model/chinese-roberta-wwm-ext')

    def __getitem__(self, index):
        item = self.all_data.iloc[index]
        id_ = item.name
        text = item['b']
        sample = self.tokenizer(text,
                              truncation=True,
                              add_special_tokens=True,
                              max_length=self.length,
                              padding='max_length',
                              return_tensors='pt').to("cuda")
        return id_, sample

    def __len__(self):
        return len(self.all_data)
