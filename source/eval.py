import argparse
import timeit
import numpy as np
import random
import json
import torch
import math
import time
import faiss
import pandas as pd
import random
from torch import nn, optim, rand, sum as tsum, reshape, save
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from autofj.datasets import load_data
from datasets import load_dataset
from scipy.stats.stats import pearsonr
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import seed_everything, LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score



def eval_bird(model, dataset, device, batch_size=4):
    text_list = []
    scores = []

    for row in dataset:
        p1, p2, score = row["term1"], row["term2"], row["relatedness score"]
        text_list.append((p1, p2))
        scores.append(score)

    cos_sim = nn.CosineSimilarity(dim=1)
    cos_sim_list = []

    for i in range(0, len(text_list), batch_size):
        batch_text_list = text_list[i:i+batch_size]
        temp1, temp2 = zip(*batch_text_list)
        temp1, temp2 = list(temp1), list(temp2)
        input1 = model(temp1, device)
        input2 = model(temp2, device)
        
        sim = cos_sim(input1, input2)
        sim = (sim + 1) / 2.0
        cos_sim_list.extend(sim.tolist())
    cor, _ = pearsonr(cos_sim_list, scores)
    #print('spearman score of BIRD is {a}'.format(a=cor))
    return cor


def eval_turney(model, dataset, device, batch_size=4):
    
    data_list = list()
    for row in dataset:
        data_list.append(list((row["query"], row["label"], row["candidate_1"], row["candidate_2"], row["candidate_3"], row["candidate_4"])))
    
    num_correct = 0
    for components in data_list:
        emb = encode_in_batch(model, batch_size=batch_size, text_list=components, device=device)
        emb = torch.stack(emb).cpu().detach().numpy()
        query = emb[0, :]
        matrix = emb[1:, :]
        scores = np.dot(matrix, query)
        chosen = np.argmax(scores)

        if chosen == 0:
            num_correct += 1
    accuracy = num_correct / len(data_list)

    return accuracy



def eval_ppdb(model, dataset, device,  batch_size=4):

    
    phrase1_list = [item["phrase_1"] for item in dataset]
    phrase2_list = [item["phrase_2"] for item in dataset]
    label = [item["label"] for item in dataset]
    #print('loaded! size = {a}'.format(a=len(phrase1_list)))

    phrase1_emb_tensor_list = encode_in_batch(model, batch_size, phrase1_list, device)
    phrase2_emb_tensor_list = encode_in_batch(model, batch_size, phrase2_list, device)
    label_list = [1 if e == 'pos' else 0 for e in label]


    combined = list(zip(phrase1_emb_tensor_list, phrase2_emb_tensor_list, label_list))
    random.shuffle(combined)
    phrase1_emb_tensor_list_shuffled, phrase2_emb_tensor_list_shuffled, label_list_shuffled = zip(*combined)
    label_tensor = torch.FloatTensor(label_list_shuffled)

    phrase1_tensor, phrase2_tensor, label = torch.stack(phrase1_emb_tensor_list_shuffled), torch.stack(phrase2_emb_tensor_list_shuffled), label_tensor

    phrase1_tensor.to(device)
    phrase2_tensor.to(device)
    label_tensor.to(device)

    split1 = math.ceil(phrase1_tensor.size()[0] * 0.7)
    split2 = math.ceil(phrase1_tensor.size()[0] * 0.85)

    train_dataset = ParaphraseDataset(phrase1_tensor[:split1, :],
                                      phrase2_tensor[:split1, :],
                                      label_tensor[:split1])
    valid_dataset = ParaphraseDataset(phrase1_tensor[split1:split2, :],
                                      phrase2_tensor[split1:split2, :],
                                      label_tensor[split1:split2])
    test_dataset = ParaphraseDataset(phrase1_tensor[split2:, :],
                                     phrase2_tensor[split2:, :],
                                     label_tensor[split2:])

    early_stop_callback = EarlyStopping(monitor='epoch_val_accuracy', min_delta=0.00, patience=5, verbose=False,
                                        mode='max')
    model = ProbingModel(input_dim=phrase1_tensor.shape[1] * 2,
                         train_dataset=train_dataset,
                         valid_dataset=valid_dataset,
                         test_dataset=test_dataset)
    trainer = Trainer(max_epochs=100, min_epochs=3, callbacks=[early_stop_callback],  gpus=[torch.cuda.current_device()])
    # trainer.tune(model)
    trainer.fit(model)
    result = trainer.test(test_dataloaders=model.test_dataloader())
    # Your statements here

    stop = timeit.default_timer()

    #print('Time: ', stop - start)

    return result[0]['epoch_test_accuracy']


def eval_clustering(model, dataset, device,  batch_size=4, name="conll"):

    label_dict = dict()
    if 'conll' in name:
        label_dict = {'PER': 0, 'LOC': 1, 'ORG': 2}
    elif 'bc5cdr' in name:
        label_dict = {'Chemical': 0, 'Disease': 1}
    num_class = len(label_dict)

    phrases, labels = list(), list()
    for row in dataset:
        entity = row['entity']
        if entity is None:entity="NA"
        label = row['label']
        phrases.append(entity)
        labels.append(label)
    
    #print('loaded! the size of data is {a}'.format(a=len(phrases)))
    phrase_emb_tensor = np.array([t.cpu().detach().numpy() for t in encode_in_batch(model, batch_size, phrases, device)])


    kmeans = KMeans(n_clusters=num_class, random_state=0).fit(phrase_emb_tensor)

    nmi_score = normalized_mutual_info_score(labels, kmeans.labels_)


    return nmi_score


def eval_retrieval(model, kb_dataset, test_dataset, batch_size=16, device='cuda:0'):

    start_time = time.time()
    e_names = [row["entity_name"] for row in kb_dataset]
    #print('entity name = {a}'.format(a=len(e_names)))
    sen_embeddings = np.array([t.cpu().detach().numpy() for t in encode_in_batch(model, batch_size, e_names, device)])
    sen_embeddings = np.array(sen_embeddings, dtype=np.float32)
    #print('entity emb = {a}'.format(a=len(sen_embeddings)))
    shape = np.shape(sen_embeddings)
    end_time = time.time()
    #print("initial --- %s seconds ---" % (round((end_time - start_time), 5)))

    start_time = time.time()
    m = 24  # number of centroid IDs in final compressed vectors
    bits = 8  # number of bits in each centroid
    nlist = 100
    quantizer = faiss.IndexFlatL2(shape[-1])  # we keep the same L2 distance flat index
    emb_index = faiss.IndexIVFPQ(quantizer, shape[-1], nlist, m, bits)
    emb_index.train(sen_embeddings)
    emb_index.add(sen_embeddings)

    end_time = time.time()
    #print("index --- %s seconds ---" % (round((end_time - start_time), 5)))

    start_time = time.time()
    cnt, wrong_cnt = 0, 0
    mentions = [row["query"] for row in test_dataset]
    labels = [row["label"] for row in test_dataset]

    batch_emb = np.array([t.cpu().detach().numpy() for t in encode_in_batch(model, batch_size, mentions, device)])

    D, I = emb_index.search(batch_emb, 1)
    predicts = [e_names[i[0]] for i in I]
    for mention, label, predict in zip(mentions, labels, predicts):
        cnt += 1
        if predict != label:
            wrong_cnt += 1
    acc = (cnt - wrong_cnt) * 1.0 / cnt
    #print('top-1 accuracy of yago = {a}'.format(a=acc))
    end_time = time.time()
    #print("search --- %s seconds ---" % (round((end_time - start_time), 5)))
    return acc


def eval_single_aotufj(dataset, model, device, batch_size):
    cos_sim = nn.CosineSimilarity(dim=1)
    left_table, right_table, gt_table = load_data(dataset)
    left_table = list(left_table.title)
    right_table = list(right_table.title)
    left_label, right_label = list(gt_table.title_l), list(gt_table.title_r)
    gt_label = dict(zip(right_label, left_label))


    all_embs = [t.detach() for t in encode_in_batch(model, batch_size, left_table+right_table, device)]
    all_embs = torch.stack(all_embs)
    left_embs, right_embs = all_embs[:len(left_table)], all_embs[len(left_table):]
    acc_cnt, total = 0, 0

    for index, r_t_emb in enumerate(right_embs):
        r_t = right_table[index]
        if r_t not in gt_label:continue
        g_t = gt_label[r_t]
        r_t_emb = torch.unsqueeze(r_t_emb, dim=0)
        score = cos_sim(r_t_emb, left_embs)
        pred_i = torch.argmax(score).item()
        predicted = left_table[pred_i]
        if predicted == g_t:
            acc_cnt += 1
        total += 1
    acc = acc_cnt * 1.0 / total

    #print('acc = {a}'.format(a=acc))
    return acc


def eval_autofj(model, dataset, device,  batch_size=4):
    table_names  = [row["Dataset"] for row in dataset]
    acc_list = list()
    for t_name in table_names:
        acc = eval_single_aotufj(dataset=t_name, model=model, device=device, batch_size=batch_size)
        print(t_name, acc)
        acc_list.append(acc)
    avg_acc = sum(acc_list) / len(acc_list)
    #print('average acc over 50 datasets = {a}'.format(a=avg_acc))
    return avg_acc


class ParaphraseDataset(Dataset):
    def __init__(self, phrase1_tensor, phrase2_tensor, label_tensor ):
        self.concat_input = torch.cat( (phrase1_tensor, phrase2_tensor), 1 )
        self.label = label_tensor

    def __getitem__(self, index):
        return (self.concat_input[index], self.label[index])

    def __len__(self):
        return self.concat_input.size()[0]

def encode_in_batch(model, batch_size, text_list, device):
    all_emb_tensor_list = []
    for i in range( 0, len(text_list), batch_size ):
        batch_text_list = text_list[i:i+batch_size]
        batch_emb_list = model(batch_text_list, device)
        if len(list(batch_emb_list.size())) < 2: batch_emb_list = torch.unsqueeze(batch_emb_list, dim=0)
        all_emb_tensor_list.extend(batch_emb_list)
    return [t.detach() for t in all_emb_tensor_list]

def load_entity(entity_path):
    e_names = list()
    cnt = 0
    for line in open(entity_path, encoding='utf8'):
        cnt += 1
        #if cnt > 1000:break
        e_name = line.strip()
        e_names.append(e_name)
    return {'mention':e_names, 'entity':e_names}


class ProbingModel(LightningModule):
    def __init__(self, input_dim=1536, train_dataset=None, valid_dataset=None, test_dataset=None):
        super(ProbingModel, self).__init__()
        # Network layers
        self.input_dim = input_dim
        self.linear = nn.Linear(self.input_dim, 256)
        self.linear2 = nn.Linear(256, 1)
        self.output = nn.Sigmoid()

        # Hyper-parameters, that we will auto-tune using lightning!
        self.lr = 0.0001
        self.batch_size = 200

        # datasets
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

    def forward(self, x):
        x1 = self.linear(x)
        x1a = F.relu(x1)
        x2 = self.linear2(x1a)
        output = self.output(x2)
        return reshape(output, (-1,))

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        return loader

    def compute_accuracy(self, y_hat, y):
        with torch.no_grad():
            y_pred = (y_hat >= 0.5)
            y_pred_f = y_pred.float()
            num_correct = tsum(y_pred_f == y)
            denom = float(y.size()[0])
            accuracy = torch.div(num_correct, denom)
        return accuracy

    def training_step(self, batch, batch_nb):
        mode = 'train'
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        accuracy = self.compute_accuracy(y_hat, y)
        #self.log(f'{mode}_loss', loss, on_epoch=True, on_step=True)
        #self.log(f'{mode}_accuracy', accuracy, on_epoch=True, on_step=True)
        return {f'loss': loss, f'{mode}_accuracy': accuracy, 'log': {f'{mode}_loss': loss}}

    def train_epoch_end(self, outputs):
        mode = 'train'
        loss_mean = sum([o[f'loss'] for o in outputs]) / len(outputs)
        accuracy_mean = sum([o[f'{mode}_accuracy'] for o in outputs]) / len(outputs)
        self.log(f'epoch_{mode}_loss', loss_mean, on_epoch=True, on_step=False)
        #print(f'\nThe end of epoch {mode} loss is {loss_mean.item():.4f}')
        self.log(f'epoch_{mode}_accuracy', accuracy_mean, on_epoch=True, on_step=False)
        #print(f'\nThe end of epoch {mode} accuracy is {accuracy_mean.item():.4f}')

    def validation_step(self, batch, batch_nb):
        mode = 'val'
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        accuracy = self.compute_accuracy(y_hat, y)
        self.log(f'{mode}_loss', loss, on_epoch=True, on_step=True)
        self.log(f'{mode}_accuracy', accuracy, on_epoch=True, on_step=True)
        return {f'{mode}_loss': loss, f'{mode}_accuracy': accuracy, 'log': {f'{mode}_loss': loss}}

    def validation_epoch_end(self, outputs):
        mode = 'val'
        loss_mean = sum([o[f'{mode}_loss'] for o in outputs]) / len(outputs)
        accuracy_mean = sum([o[f'{mode}_accuracy'] for o in outputs]) / len(outputs)
        self.log(f'epoch_{mode}_loss', loss_mean, on_epoch=True, on_step=False)
        #print(f'\nThe end of epoch {mode} loss is {loss_mean.item():.4f}')
        self.log(f'epoch_{mode}_accuracy', accuracy_mean, on_epoch=True, on_step=False)
        #print(f'\nThe end of epoch {mode} accuracy is {accuracy_mean.item():.4f}')

    def test_step(self, batch, batch_nb):
        mode = 'test'
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        accuracy = self.compute_accuracy(y_hat, y)
        self.log(f'{mode}_loss', loss, on_epoch=True, on_step=True)
        self.log(f'{mode}_accuracy', accuracy, on_epoch=True, on_step=True)
        return {f'{mode}_loss': loss, f'{mode}_accuracy': accuracy, 'log': {f'{mode}_loss': loss}}

    def test_epoch_end(self, outputs):
        mode = 'test'
        loss_mean = sum([o[f'{mode}_loss'] for o in outputs]) / len(outputs)
        accuracy_mean = sum([o[f'{mode}_accuracy'] for o in outputs]) / len(outputs)
        self.log(f'epoch_{mode}_loss', loss_mean, on_epoch=True, on_step=False)
        # print(f'\nThe end of epoch {mode} loss is {loss_mean.item():.4f}')
        self.log(f'epoch_{mode}_accuracy', accuracy_mean, on_epoch=True, on_step=False)
        # print(f'\nThe end of epoch {mode} accuracy is {accuracy_mean.item():.4f}')


class PearlSmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        model_name = "Lihuchen/pearl_small"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)


    def average_pool(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        

    def forward(self, x, device):
        # Tokenize the input texts
        batch_dict = self.tokenizer(x, max_length=128, padding=True, truncation=True, return_tensors='pt')
        batch_dict = batch_dict.to(device)

        outputs = self.model(**batch_dict)
        phrase_vec = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        return phrase_vec.detach()
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='contrastive learning framework for word vector')
    parser.add_argument('-batch_size', help='the number of samples in one batch', type=int, default=32)
    args = parser.parse_args()
    
    
    model = PearlSmallModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    batch_size = args.batch_size
    
    ppdb_dataset = load_dataset("Lihuchen/pearl_benchmark", "ppdb", split="test")
    ppbd_score = eval_ppdb(model, ppdb_dataset, device=device,  batch_size=batch_size)
    print("ppdb: ", ppbd_score)
    
    ppdb_filtered_dataset = load_dataset("Lihuchen/pearl_benchmark", "ppdb_filtered", split="test")
    ppbd_filtered_score = eval_ppdb(model, ppdb_filtered_dataset, device=device, batch_size=batch_size)
    print("ppdb_filetered: ", ppbd_filtered_score)
    
    turney_dataset = load_dataset("Lihuchen/pearl_benchmark", "turney", split="test")
    turney_score = eval_turney(model, turney_dataset, device=device, batch_size=batch_size)
    print("turney: ", turney_score)
    
    
    bird_dataset = load_dataset("Lihuchen/pearl_benchmark", "bird", split="test")
    bird_score = eval_bird(model, bird_dataset, device=device, batch_size=batch_size)
    print("bird: ", bird_score)
    
    yago_kb_dataset = load_dataset("Lihuchen/pearl_benchmark", "kb", split="yago")
    yago_test_dataset = load_dataset("Lihuchen/pearl_benchmark", "yago", split="test")
    yago_score = eval_retrieval(model, yago_kb_dataset, yago_test_dataset, device=device, batch_size=batch_size)
    print("yago: ", yago_score)
    
    umls_kb_dataset = load_dataset("Lihuchen/pearl_benchmark", "kb", split="umls")
    umls_test_dataset = load_dataset("Lihuchen/pearl_benchmark", "umls", split="test")
    umls_score = eval_retrieval(model, umls_kb_dataset, umls_test_dataset, device=device,  batch_size=batch_size)
    print("umls: ", umls_score)
    
    conll_dataset = load_dataset("Lihuchen/pearl_benchmark", "conll", split="test")
    conll_score = eval_clustering(model, device=device,  batch_size=batch_size, name="conll")
    print("conll: ", conll_score)
    
    bc5cdr_dataset = load_dataset("Lihuchen/pearl_benchmark", "bc5cdr", split="test")
    bc5cdr_score = eval_clustering(model, bc5cdr_dataset, device=device,  batch_size=batch_size, name="bc5cdr")
    print("bc5cdr: ", bc5cdr_score)
    
    autofj_dataset = load_dataset("Lihuchen/pearl_benchmark", "autofj", split="test")
    autofj_score = eval_autofj(model, autofj_dataset, device=device,  batch_size=batch_size)
    print("autofj: ", autofj_score)





