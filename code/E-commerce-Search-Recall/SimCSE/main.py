import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data import Unsupervised, Supervised, TESTDATA, Evaluated
from model import TextBackbone
from datetime import datetime
from transformers import AdamW, get_linear_schedule_with_warmup
import  fire
import torch.nn as nn
from tqdm import  tqdm
import pickle

def unsup_loss(y_pred, lamda=0.05):
    idxs = torch.arange(0, y_pred.shape[0], device="cuda")
    y_true = idxs + 1 - idxs % 2 * 2
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)

    similarities = similarities - torch.eye(y_pred.shape[0], device="cuda") * 1e12

    similarities = similarities / lamda

    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


def sup_loss(y_pred, lamda=0.05):
    row = torch.arange(0, y_pred.shape[0], 3, device="cuda")
    col = torch.arange(y_pred.shape[0], device="cuda")
    col = torch.where(col % 3 != 0)[0].cuda()
    y_true = torch.arange(0, len(col), 2, device="cuda")
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)

    similarities = torch.index_select(similarities, 0, row)
    similarities = torch.index_select(similarities, 1, col)

    similarities = similarities / lamda

    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


def train(dataloader, model, optimizer, schedular, criterion, mode="unsup"):
    num = 2
    if mode == "sup":
        num = 3
    model.train()
    all_loss = []
    for idx, data in enumerate(tqdm(dataloader)):
        input_ids = data["input_ids"].view(len(data["input_ids"]) * num, -1).cuda()
        attention_mask = (
            data["attention_mask"].view(len(data["attention_mask"]) * num, -1).cuda()
        )
        token_type_ids = (
            data["token_type_ids"].view(len(data["token_type_ids"]) * num, -1).cuda()
        )
        pred = model(input_ids, attention_mask, token_type_ids)
        optimizer.zero_grad()
        loss = criterion(pred)
        all_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        schedular.step()
        if idx % 500 == 499:
            t = sum(all_loss) / len(all_loss)
            info = str(idx) + " == {} == ".format(mode) + str(t) + "\n"
            print(info)
            all_loss = []


# def prepare():
#     os.makedirs("./output", exist_ok=True)
#     now = datetime.now()
#     log_file = now.strftime("%Y_%m_%d_%H_%M_%S") + "_log.txt"
#     return "./output/" + log_file

def fun():
    # torch.cuda.set_device('cuda:0,1')
    device = 'cuda'
    # print(device)
    print(torch.cuda.device_count())
    model = TextBackbone()
    model = nn.DataParallel(model)
    model = model.cuda()
    model.load_state_dict(torch.load("./output/sup_epoch_10.pt"))
    # print(model)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    # print(torch.cuda.current_device())
    # print(model.device)
    evaluate(model)


def evaluate(model=None):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    num = 1
    sum = 0
    devdata = Evaluated()
    devloader = DataLoader(devdata, batch_size=512, shuffle=False)
    devdata_embedding = None
    devdata_label = []
    with torch.no_grad():
        for x, label in tqdm(devloader):
            y = model.predict(x)
            devdata_label += label.numpy().tolist()
            if devdata_embedding == None:
                devdata_embedding = y
            else:
                devdata_embedding = torch.cat([devdata_embedding, y])
    print(devdata_embedding.shape)
    print(len(label))


    devcorpus = TESTDATA(certain="dev_corpus_cl.tsv")
    devcorpusloader = DataLoader(devcorpus, batch_size=512, shuffle=False)
    devcorpus_embedding = None
    devcorpus_label = []
    with torch.no_grad():
            for idx, x  in tqdm(devcorpusloader):
                y = model.predict(x)
                devcorpus_label+=idx.numpy().tolist()
                if devcorpus_embedding == None:
                    devcorpus_embedding = y
                else:
                    devcorpus_embedding = torch.cat([devcorpus_embedding,y])
    print(devcorpus_embedding.shape)
    print(len(devcorpus_label))
    # pickle.dump(devcorpus_embedding, open('devcorpus_embedding.pkl','wb'))
    # pickle.dump(devcorpus_label, open('devcorpus_label.pkl', 'wb'))

    # devcorpus_label = pickle.load(open('devcorpus_label.pkl','rb'))
    # devcorpus_embedding = pickle.load(open('devcorpus_embedding.pkl','rb'))
    for embedding,label in zip(devdata_embedding, devdata_label):
        # print(type(embedding))
        # embedding [1,128]
        embedding = embedding.view(128)
        # print(embedding.shape)
        # sim [1,100000]
        sim = F.cosine_similarity(devcorpus_embedding, embedding, dim=-1)
        # print(sim.shape)
        # print(label)
        id = devcorpus_label.index(label)
        tar_sim = sim[id]
        tar = (sim>tar_sim).nonzero()
        # print(len(tar)+1)
        sum += 1/(len(tar)+1)
    print(sum/len(devdata_embedding))
    return sum/len(devdata_embedding)

def unsupervised_train():
    dataset = Unsupervised()
    dataloader = DataLoader(dataset, batch_size=156, shuffle=True, drop_last=True)
    model = TextBackbone()
    model = nn.DataParallel(model)
    model = model.cuda()
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
    # unspervise train
    epochs = 3
    num_train_steps = int(len(dataloader) * epochs)
    schedular = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.05 * num_train_steps,
        num_training_steps=num_train_steps,
    )

    criterion = unsup_loss
    for epoch in range(1, epochs + 1):
        train(dataloader, model, optimizer, schedular, criterion)
        torch.save(model.state_dict(), "./output/unsup_epoch_{}.pt".format(epoch))
        score = evaluate(model)


def supervised_train():
    dataset = Supervised()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)

    # log_file = prepare()
    model = TextBackbone()
    model = nn.DataParallel(model)
    # print(model)
    # 加载无监督预训练的模型
    model.load_state_dict(torch.load("./output/unsup_epoch_2.pt"))
    model = model.cuda()
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
    epochs = 10
    num_train_steps = int(len(dataloader) * epochs)
    schedular = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.05 * num_train_steps,
        num_training_steps=num_train_steps,
    )

    criterion = sup_loss
    max_score = 0
    for epoch in range(1, epochs + 1):
        train(dataloader, model, optimizer, schedular, criterion, mode="sup")
        score = evaluate(model)
        if max_score<score:
            max_score = score
            print('max_score: ', max_score)
            print('save state_dict: ', epoch)
            torch.save(model.state_dict(), "./output/sup_epoch_{}.pt".format(epoch))


def inference():
    if os.path.exists("doc_embedding"):
        os.remove("doc_embedding")
    if os.path.exists("query_embedding"):
        os.remove("query_embedding")
    model = TextBackbone()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load("./output/sup_epoch_10.pt"))
    model = model.cuda()
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    testdata = TESTDATA(certain="test_query_cl.tsv")
    testloader = DataLoader(testdata, batch_size=1, shuffle=False)
    for idx, x in tqdm(testloader):
        with torch.no_grad():
            y = model.predict(x)[0].detach().cpu().numpy().tolist()
            y = [str(round(float(i), 8)) for i in y]
            info = idx[0] + "\t"
            info = info + ",".join(y)
            with open("query_embedding", "a+") as f:
                f.write(info + "\n")

    testdata = TESTDATA(certain="corpus_cl.tsv")
    testloader = DataLoader(testdata, batch_size=1024, shuffle=False)
    doc_embedding = []
    for idx, x in tqdm(testloader):
        with torch.no_grad():
            y = model.predict(x).detach().cpu().numpy().tolist()
            for x1, y1 in zip(idx, y):
                y1 = [str(round(float(i), 8)) for i in y1]
                info = x1 + "\t"
                info = info + ",".join(y1)
                doc_embedding.append(info)
                with open("doc_embedding", "a+") as f:
                    f.write(info + "\n")


if __name__ == "__main__":
    fire.Fire()
