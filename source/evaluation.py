from torch import nn
from scipy.stats.stats import pearsonr
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import seed_everything, LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset
import timeit
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from utils import ProbingModel, load_entity, get_char_input, get_char_batch
import numpy as np
import random
import json
import torch
import math
import time
import faiss
import pandas as pd
from autofj.datasets import load_data
data_path = ''


def eval_bird(model):
    text_list = []
    scores = []

    bird_handler = open(data_path+'../data/bird.txt', "r")
    for line_no, line in enumerate(bird_handler):
        if line_no == 0:
            # skip header
            continue
        words = line.rstrip().split("\t")
        p1, p2, score = words[1], words[2], float(words[-2])
        text_list.append((p1, p2))
        scores.append(score)

    batch_size = 32
    cos_sim = nn.CosineSimilarity(dim=1)
    cos_sim_list = []

    for i in range( 0, len(text_list), batch_size ):
        batch_text_list = text_list[i:i+batch_size]
        temp1, temp2 = zip(*batch_text_list)
        temp1, temp2 = list(temp1), list(temp2)

        input1 = model(temp1, train=False)
        input2 = model(temp2, train=False)
        sim = cos_sim(input1, input2)
        sim = (sim + 1) / 2.0
        cos_sim_list.extend(sim.tolist())

    cor, _ = pearsonr(cos_sim_list, scores)
    print('spearman score of BIRD is {a}'.format(a=cor))
    return cor


def eval_turney(model):
    model.eval()
    with open(data_path+'../data/turney.txt', 'r') as f:
        content = f.readlines()
        data_list = []
        for line in content:
            components = line.strip('\n').split(' | ')
            data_list.append(components[:2]+components[4:])

    num_correct = 0
    for components in data_list:
        emb = model(components, train=False).cpu().detach().numpy()
        query = emb[0, :]
        matrix = emb[1:, :]

        scores = np.dot(matrix, query)
        chosen = np.argmax(scores)

        if chosen == 0:
            num_correct += 1
    accuracy = num_correct / len(data_list)
    print(f'Accuracy on Turney = {accuracy}')

    return accuracy


def encode_in_batch(model, batch_size, text_list):
    all_emb_tensor_list = []
    for i in range( 0, len(text_list), batch_size ):
        batch_text_list = text_list[i:i+batch_size]
        batch_emb_list = model(batch_text_list, train=False)
        if len(list(batch_emb_list.size())) < 2: batch_emb_list = torch.unsqueeze(batch_emb_list, dim=0)
        all_emb_tensor_list.extend(batch_emb_list)
    return all_emb_tensor_list


class ParaphraseDataset(Dataset):
    def __init__(self, phrase1_tensor, phrase2_tensor, label_tensor ):
        self.concat_input = torch.cat( (phrase1_tensor, phrase2_tensor), 1 )
        self.label = label_tensor

    def __getitem__(self, index):
        return (self.concat_input[index], self.label[index])

    def __len__(self):
        return self.concat_input.size()[0]


def eval_ppdb(model, emb_batch_size, path, device='cuda:0'):
    path = data_path + path
    device_num = int(device.split(':')[-1])
    start = timeit.default_timer()

    with open(path, 'r') as f:
        data_list = json.load(f)

    phrase1_list = [item[0] for item in data_list]
    phrase2_list = [item[1] for item in data_list]
    label = [item[2] for item in data_list]
    print('loaded! size = {a}'.format(a=len(phrase1_list)))

    #model.model.to(device)

    phrase1_emb_tensor_list = encode_in_batch(model, emb_batch_size, phrase1_list)
    phrase2_emb_tensor_list = encode_in_batch(model, emb_batch_size, phrase2_list)
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

    early_stop_callback = EarlyStopping(monitor='epoch_val_accuracy', min_delta=0.00, patience=5, verbose=True,
                                        mode='max')
    model = ProbingModel(input_dim=phrase1_tensor.shape[1] * 2,
                         train_dataset=train_dataset,
                         valid_dataset=valid_dataset,
                         test_dataset=test_dataset)
    trainer = Trainer(max_epochs=100, min_epochs=3, callbacks=[early_stop_callback],  gpus=[device_num])
    # trainer.tune(model)
    trainer.fit(model)
    result = trainer.test(dataloaders=model.test_dataloader())
    print(f'\n finished \n')
    output_fname = 'output/ppdb.pt'

    with open(output_fname, 'w') as f:
        json.dump(result, f, indent=4)

    print(result)

    # Your statements here

    stop = timeit.default_timer()

    print('Time: ', stop - start)

    return result[0]['epoch_test_accuracy']


def eval_clustering(model, path='data/conll2003/', batch_size=10, device='cuda:0'):

    base_path = data_path + path
    test_path = base_path + 'test.json'

    label_dict = dict()
    if 'conll2003' in base_path:
        label_dict = {'PER': 0, 'LOC': 1, 'ORG': 2}
    elif 'bc5cdr' in base_path:
        label_dict = {'Chemical': 0, 'Disease': 1}
    num_class = len(label_dict)

    phrases, labels = list(), list()
    for line in open(test_path, encoding='utf8'):
        obj = json.loads(line)
        sentences = obj['sentences'][0]
        spans = obj['ner'][0]

        for span in spans:
            start, end, e_type = span[0], span[1], span[2]
            if e_type not in label_dict:continue
            e_type_id = label_dict[e_type]
            entity = ' '.join(sentences[start:end+1])
            phrases.append(entity)
            labels.append(e_type_id)

    print('loaded! the size of data is {a}'.format(a=len(phrases)))

    phrase_emb_tensor = np.array([t.cpu().detach().numpy() for t in encode_in_batch(model, batch_size, phrases)])

    print('finished embedding')

    kmeans = KMeans(n_clusters=num_class, random_state=0).fit(phrase_emb_tensor)

    nmi_score = normalized_mutual_info_score(labels, kmeans.labels_)

    print('NMI of CONLL is {a}'.format(a=nmi_score))

    return nmi_score


def eval_retrieval(model, kb_path, test_path, batch_size=16, device='cuda:0'):
    test_path = data_path + test_path
    kb_path = data_path + kb_path
    #model.model.to(device)

    start_time = time.time()
    e_names = load_entity(entity_path=kb_path)['entity']
    print('entity name = {a}'.format(a=len(e_names)))
    sen_embeddings = np.array([t.cpu().detach().numpy() for t in encode_in_batch(model, batch_size, e_names)])
    sen_embeddings = np.array(sen_embeddings, dtype=np.float32)
    print('entity emb = {a}'.format(a=len(sen_embeddings)))
    shape = np.shape(sen_embeddings)
    end_time = time.time()
    print("initial --- %s seconds ---" % (round((end_time - start_time), 5)))

    start_time = time.time()
    m = 24  # number of centroid IDs in final compressed vectors
    bits = 8  # number of bits in each centroid
    nlist = 100
    quantizer = faiss.IndexFlatL2(shape[-1])  # we keep the same L2 distance flat index
    emb_index = faiss.IndexIVFPQ(quantizer, shape[-1], nlist, m, bits)
    emb_index.train(sen_embeddings)
    emb_index.add(sen_embeddings)

    end_time = time.time()
    print("index --- %s seconds ---" % (round((end_time - start_time), 5)))

    start_time = time.time()
    cnt, wrong_cnt = 0, 0
    mentions, labels = list(), list()
    for line in open(test_path, encoding='utf8'):
        cnt += 1
        # if cnt >= 10:break
        row = line.strip().split('\t')
        mention, label = row[0], row[1]
        mentions.append(mention)
        labels.append(label)

    batch_emb = np.array([t.cpu().detach().numpy() for t in encode_in_batch(model, batch_size, mentions)])

    D, I = emb_index.search(batch_emb, 1)
    predicts = [e_names[i[0]] for i in I]
    for mention, label, predict in zip(mentions, labels, predicts):
        if predict != label:
            wrong_cnt += 1
            if wrong_cnt < 200:
                print('[wrong], the mentin is [{a}], the predicted is [{b}], the label is [{c}]'.format(a=mention,
                                                                                                        b=predict,
                                                                                              c=label))
    acc = (cnt - wrong_cnt) * 1.0 / cnt
    print('top-1 accuracy of yago = {a}'.format(a=acc))
    end_time = time.time()
    print("search --- %s seconds ---" % (round((end_time - start_time), 5)))
    return acc


def check_aotufj(dataset, model):
    cos_sim = nn.CosineSimilarity(dim=1)
    left_table, right_table, gt_table = load_data(dataset)
    #left_table = [e[0] for e in left_table['title']]
    left_table = list(left_table.title)
    right_table = list(right_table.title)
    #left_table = ['2008 Bell Challenge', '2003 Dubai Tennis Championships and Duty Free Women\'s Open', '2008 Challenge Bell']
    left_label, right_label = list(gt_table.title_l), list(gt_table.title_r)
    gt_label = dict(zip(right_label, left_label))

    #all_embs = model(left_table+right_table)
    all_embs = [t.detach() for t in encode_in_batch(model, 128, left_table+right_table)]
    all_embs = torch.stack(all_embs)
    left_embs, right_embs = all_embs[:len(left_table)], all_embs[len(left_table):]
    acc_cnt, total = 0, 0

    for index, r_t_emb in enumerate(right_embs):
        #if r_t != '2008 Challenge Bell':continue
        r_t = right_table[index]
        if r_t not in gt_label:continue
        g_t = gt_label[r_t]
        score = cos_sim(r_t_emb, left_embs)
        pred_i = torch.argmax(score).item()
        #print(pred_i, score[pred_i])
        predicted = left_table[pred_i]
        #print(r_t, predicted)
        if predicted == g_t:
            acc_cnt += 1
        total += 1
    acc = acc_cnt * 1.0 / total

    print('acc = {a}'.format(a=acc))
    return acc


def run_autofj_eval(model):
    model.eval()
    autofj_table = pd.read_table('../data/autofj.md', sep="|", header=0, skipinitialspace=True).dropna(axis=1,
                                                                                                       how='all').iloc[
                   1:]
    table_names = [e.replace(' ', '') for e in list(autofj_table[autofj_table.keys()[0]])]
    acc_list = list()
    for t_name in table_names:
        print(t_name)
        #if t_name != 'FootballLeagueSeason':continue
        acc = check_aotufj(dataset=t_name, model=model)
        acc_list.append(acc)
    avg_acc = sum(acc_list) / len(acc_list)
    print('average acc over 50 datasets = {a}'.format(a=avg_acc))
    return avg_acc



if __name__ == "__main__":
    eval_clustering(None)

