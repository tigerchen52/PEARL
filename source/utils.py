import torch
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, Dataset
import itertools



def load_dataset(path, lower=True):

    freq_list, ptype_list, phrase_list = list(), list(), list()
    for index, line in enumerate(open(path, encoding='utf8')):
        if index > 1000:break
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


def tokenize_and_getid(word, tokenizer):
    tokens = tokenizer.tokenize(tokenizer.convert_to_unicode(word))
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokens, token_ids


def load_hard_negative(phrase_path, neg_path, num):
    phrases = list()


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
    for line in open('train_data/phrase_with_etype.txt', encoding='utf8'):
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



if __name__ == '__main__':
    load_hard_negative(path='data/hard_negative.txt',num= 3)


