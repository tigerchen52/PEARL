import torch
import torch.nn.functional as F
import random
from pytorch_lightning import seed_everything, LightningModule, Trainer
from torch import nn, optim, rand, sum as tsum, reshape, save
from torch.utils.data import DataLoader, Dataset
import tokenization
import itertools


def load_dataset(path, lower=True):

    freq_list, ptype_list, phrase_list = list(), list(), list()
    for line in open(path, encoding='utf8'):
        row = line.strip().split('\t')
        freq = int(row[1])
        ptype_phrase = row[0].split('__')
        ptype, phrase = ptype_phrase[0], ptype_phrase[1]
        if lower:phrase = str.lower(phrase)

        freq_list.append(freq)
        ptype_list.append(ptype)
        phrase_list.append(phrase)

    print('loaded! phrase num = {a}'.format(a=len(freq_list)))
    return {'freq': freq_list, 'ptype':ptype_list, 'phrase':phrase_list}


def load_phrase_con(path):
    ptype_dict = dict()
    for line in open(path, encoding='utf8'):
        row = line.strip().split('\t')
        freq = int(row[1])

        ptype_phrase = row[0].split('__')
        ptype, phrase = ptype_phrase[0], ptype_phrase[1]

        ptype_dict[phrase] = ptype
    return ptype_dict


def load_predict_dataset(path):
    origin_words, origin_repre = list(), list()
    for line in open(path, encoding='utf8'):
        word = line.strip()
        origin_repre.append(word)
        origin_words.append(word)
    print('loaded! Word num = {a}'.format(a=len(origin_words)))
    return {'origin_word': origin_words, 'origin_repre':origin_repre}


VOCAB = '../data/vocab.txt'
tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB, do_lower_case=True)


def tokenize_and_getid(word, tokenizer):
    tokens = tokenizer.tokenize(tokenizer.convert_to_unicode(word))
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokens, token_ids


def get_char_input(phrase):
    start = '[CLS]'
    sub = '[SUB]'
    end = '[SEP]'
    char_seq = list(phrase)
    tokens, _ = tokenize_and_getid(phrase, tokenizer)
    repre = [start] + char_seq + [sub] + tokens + [end]
    repre_ids = tokenizer.convert_tokens_to_ids(repre)

    return repre, repre_ids


def get_char_batch(repre_ids, pad=0):
    max_len = max([len(seq) for seq in repre_ids])
    batch_aug_repre_ids = [char + [pad] * (max_len - len(char)) for char in repre_ids]
    batch_aug_repre_ids = torch.LongTensor(batch_aug_repre_ids)
    mask = torch.ne(batch_aug_repre_ids, pad).unsqueeze(2)

    return batch_aug_repre_ids, mask


def load_hard_negative(phrase_path, neg_path, num):
    phrases = list()
    # for line in open(path, encoding='utf8'):
    #     row = line.strip().split('\t')
    #     phrases.append(row[0])

    for line in open(phrase_path, encoding='utf8'):
        row = line.strip().split('\t')
        freq = int(row[1])
        ptype_phrase = row[0].split('__')
        ptype, phrase = ptype_phrase[0], ptype_phrase[1]
        phrases.append(phrase)

    hard_negatives = dict()
    for line in open(neg_path, encoding='utf8'):
        row = line.strip().split('\t')
        phrase = row[0]
        negative = row[1:num+1]
        hard_negatives[phrase] = negative
        # if len(negative) < num:
        #     sampled = random.sample(phrases, num-len(negative))
        #     negative.extend([s for s in sampled if s!=phrase])
        # hard_negatives[phrase] = negative

    for phrase in phrases:
        if phrase not in hard_negatives:
            sampled = random.sample(phrases, num)
            hard_negatives[phrase] = sampled
        else:
            if len(hard_negatives[phrase]) < num:
                sampled = random.sample(phrases, num-len(hard_negatives[phrase]))
                hard_negatives[phrase].extend([s for s in sampled if s!=phrase])

    return hard_negatives


def get_phrase_type():
    constituency_list = ['NP', 'VP', 'PP', 'ADVP', 'ADJP']
    phrase_type, type_list = dict(), list()
    for line in open('../data/phrase_with_etype.txt', encoding='utf8'):
        row = line.strip().split('\t')
        phrase, p_type = row[0], row[1]
        phrase_type[phrase] = p_type
        if p_type not in type_list:
            type_list.append(p_type)
    type_list.append('OTHER')
    labels = list()
    for i in itertools.product(constituency_list, type_list):
        labels.append(i[0]+'-'+i[1])
    return phrase_type, labels


phrase_type_dict, labels = get_phrase_type()
def get_type_id(phrase, con):
    p_label = con+'-'
    p_type = 'OTHER'
    if phrase in phrase_type_dict:
        p_type = phrase_type_dict[phrase]

    p_label += p_type
    return labels.index(p_label)
    #return p_label

class TextData(Dataset):
    def __init__(self, data):
        self.freq = data['freq']
        self.ptype = data['ptype']
        self.phrase = data['phrase']

    def __len__(self):
        return len(self.freq)

    def __getitem__(self, idx):
        return self.freq[idx], self.ptype[idx], self.phrase[idx], get_type_id(self.phrase[idx], self.ptype[idx])


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
        self.log(f'{mode}_loss', loss, on_epoch=True, on_step=True)
        self.log(f'{mode}_accuracy', accuracy, on_epoch=True, on_step=True)
        return {f'loss': loss, f'{mode}_accuracy': accuracy, 'log': {f'{mode}_loss': loss}}

    def train_epoch_end(self, outputs):
        mode = 'train'
        loss_mean = sum([o[f'loss'] for o in outputs]) / len(outputs)
        accuracy_mean = sum([o[f'{mode}_accuracy'] for o in outputs]) / len(outputs)
        self.log(f'epoch_{mode}_loss', loss_mean, on_epoch=True, on_step=False)
        print(f'\nThe end of epoch {mode} loss is {loss_mean.item():.4f}')
        self.log(f'epoch_{mode}_accuracy', accuracy_mean, on_epoch=True, on_step=False)
        print(f'\nThe end of epoch {mode} accuracy is {accuracy_mean.item():.4f}')

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
        print(f'\nThe end of epoch {mode} loss is {loss_mean.item():.4f}')
        self.log(f'epoch_{mode}_accuracy', accuracy_mean, on_epoch=True, on_step=False)
        print(f'\nThe end of epoch {mode} accuracy is {accuracy_mean.item():.4f}')

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
        print(f'\nThe end of epoch {mode} loss is {loss_mean.item():.4f}')
        self.log(f'epoch_{mode}_accuracy', accuracy_mean, on_epoch=True, on_step=False)
        print(f'\nThe end of epoch {mode} accuracy is {accuracy_mean.item():.4f}')


def load_entity(entity_path):
    e_names = list()
    cnt = 0
    for line in open(entity_path, encoding='utf8'):
        cnt += 1
        #if cnt > 1000:break
        e_name = line.strip()
        e_names.append(e_name)
    return {'mention':e_names, 'entity':e_names}

def read_ner_data(path):
    pass

if __name__ == '__main__':
    load_hard_negative(path='data/hard_negative.txt',num= 3)


