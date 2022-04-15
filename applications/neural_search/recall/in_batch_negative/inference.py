from functools import partial
import argparse
import os
import sys
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset, MapDataset, load_dataset
from paddlenlp.utils.log import logger

from base_model import SemanticIndexBase, SemanticIndexBaseStatic
from data import convert_example, create_dataloader
from data import gen_id2corpus, gen_text_file
from ann_util import build_index
from tqdm import tqdm
import pandas as pd

def load_corpus(corpus_path):
    id2corpus = {}
    corpus_data = pd.read_csv( corpus_path, sep="\t", names=["doc", "title"])
    for index, title in enumerate(corpus_data['title']):
        id2corpus[index] = title
    return id2corpus

if __name__ == "__main__":
    device = 'gpu'
    max_seq_length = 64
    output_emb_size = 128
    batch_size = 64
    params_path = 'checkpoints/simcse_inbatch_negative/model_1800/model_state.pdparams'
    corpus_path = '/home/data_ti6_c/wanghk/work_data/corpus.tsv'
    id2corpus = load_corpus(corpus_path)
    print(list(id2corpus.items())[:5])

    paddle.set_device(device)

    tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained('ernie-1.0')
    trans_func = partial(
        convert_example, tokenizer=tokenizer, max_seq_length=max_seq_length)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # text_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # text_segment
    ): [data for data in fn(samples)]

    pretrained_model = ppnlp.transformers.ErnieModel.from_pretrained(
        "ernie-1.0")

    model = SemanticIndexBaseStatic(
        pretrained_model, output_emb_size=output_emb_size)

    # Load pretrained semantic model
    if params_path and os.path.isfile(params_path):
        state_dict = paddle.load(params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % params_path)
    else:
        raise ValueError(
            "Please set --params_path with correct pretrained model file")

    # conver_example function's input must be dict
    corpus_list = [{idx: text} for idx, text in id2corpus.items()]
    corpus_ds = MapDataset(corpus_list)

    corpus_data_loader = create_dataloader(
        corpus_ds,
        mode='predict',
        batch_size=batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    all_embeddings = []
    model.eval()
    with paddle.no_grad():
        for batch_data in tqdm(corpus_data_loader):
            input_ids, token_type_ids = batch_data
            input_ids = paddle.to_tensor(input_ids)
            token_type_ids = paddle.to_tensor(token_type_ids)

            text_embeddings = model(input_ids, token_type_ids)
            all_embeddings.append(text_embeddings)

    with open('doc_embedding', 'w') as up:
        index = 1
        for text_embedding in all_embeddings:
            for embedding in text_embedding:
                up.write('{0}\t{1}\n'.format(index, ','.join([str(x.item())[:6] for x in embedding])))
                index += 1