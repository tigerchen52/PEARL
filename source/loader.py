from registry import register
from functools import partial
import torch
import random
from torch.utils.data import DataLoader
from utils import load_dataset, TextData, load_hard_negative, get_type_id, load_phrase_con
from augmentation import get_random_aug
registry = {}
register = partial(register, registry=registry)


@register('phrase_loader')
class PhraseLoader():
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.shuffle = args.shuffle
        self.lowercase = args.lowercase
        self.dataset = args.dataset
        self.hard_neg_numbers = args.hard_neg_numbers
        self.hard_neg_path = args.hard_neg_path
        self.neg_samples = load_hard_negative(self.dataset, self.hard_neg_path, num=12)
        print('loaded neg_samples = {a} '.format(a=len(self.neg_samples)))
        self.phrase_con = load_phrase_con(args.dataset)

    def collate_fn(self, batch_data):
        freq_list, ptype_list, phrase_list, p_type_label = list(zip(*batch_data))
        batch_origin_phrase, batch_aug_phrase, batch_origin_input, batch_aug_input = list(), list(), list(), list()

        phrase_with_negs, p_type_label_with_negs = list(), list()
        for index in range(len(phrase_list)):
            phrase = phrase_list[index]
            t_id = p_type_label[index]
            phrase_with_negs.append(phrase)
            p_type_label_with_negs.append(t_id)

            negatives = random.sample(self.neg_samples[phrase], self.hard_neg_numbers)
            type_ids = [get_type_id(neg, self.phrase_con[neg]) for neg in negatives]

            phrase_with_negs.extend(negatives)
            p_type_label_with_negs.extend(type_ids)

        for index in range(len(phrase_with_negs)):
            origin_phrase = phrase_with_negs[index]
            aug_phrase = get_random_aug(origin_phrase)

            batch_origin_phrase.append(origin_phrase)
            batch_aug_phrase.append(aug_phrase)

        # ptype
        batch_type = torch.tensor(p_type_label_with_negs)

        return batch_origin_phrase, batch_aug_phrase, batch_type

    def __call__(self, data_path):
        dataset = load_dataset(path=data_path, lower=self.lowercase)
        dataset = TextData(dataset)
        train_iterator = DataLoader(dataset=dataset, batch_size=self.batch_size // (2 * (self.hard_neg_numbers+1)), shuffle=self.shuffle,
                                    collate_fn=self.collate_fn)

        return train_iterator