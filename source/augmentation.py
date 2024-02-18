import numpy as np
import nlpaug.augmenter.char as nac
import json
import random
import nltk


def load_synonym():
    synonyms = dict()
    empty_cnt, total_cnt = 0, 0
    for line in open('train_data/token_aug.jsonl', encoding='utf8'):
        total_cnt += 1
        obj = json.loads(line)
        phrase, all_syn = obj['phrase'], obj['synonyms']
        if len(all_syn) == 0:
            empty_cnt += 1
            continue
        synonyms[phrase] = all_syn

    print('loaded synonyms! total: {a}, missing: {b}'.format(a=total_cnt, b=empty_cnt))
    return synonyms


phrase_synonyms = load_synonym()


def load_parrot():
    synonyms = dict()
    empty_cnt, total_cnt = 0, 0
    for line in open('train_data/phrase_aug.jsonl', encoding='utf8'):
        total_cnt += 1
        obj = json.loads(line)
        phrase, aug = obj['phrase'], list(set(obj['aug']))
        if len(aug) == 0:
            empty_cnt += 1
            continue
        synonyms[phrase] = aug

    print('loaded parrot paraphrase! total: {a}, missing: {b}'.format(a=total_cnt, b=empty_cnt))
    return synonyms

parrot_paraphrase = load_parrot()


def keyboard(word, max_char=1):
    aug = nac.KeyboardAug(include_upper_case=True, aug_char_max=max_char)
    auged_word = aug.augment(word)
    return auged_word[0]


def insert_aug(word, max_char=1):
    aug = nac.RandomCharAug(action="insert", aug_char_max=max_char)
    auged_word = aug.augment(word)
    return auged_word[0]


def swap_aug(word, max_char=1):
    aug = nac.RandomCharAug(action="swap", aug_char_max=max_char)
    auged_word = aug.augment(word)
    #print(word, auged_word)
    return auged_word[0]


def delete_aug(word, max_char=1):
    aug = nac.RandomCharAug(action="delete", aug_char_max=max_char)
    auged_word = aug.augment(word)
    #print(word, auged_word)
    return auged_word[0]


def substitute_aug(word, max_char=1):
    aug = nac.RandomCharAug(action="substitute", aug_char_max=max_char)
    auged_word = aug.augment(word)
    #print(word, auged_word)
    return auged_word[0]


def ocr_aug(word, max_char=1):
    aug = nac.OcrAug(aug_char_max=max_char)
    return aug.augment(word)[0]


def character_aug(phrase):
    attack_type = ['swap', 'delete', 'insert', 'keyboard', 'substitute', 'ocr']
    probs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    attack_probs = np.array(probs)
    attack_probs = attack_probs / sum(attack_probs)
    attack = np.random.choice(attack_type, 1, p=attack_probs)[0]
    #print(attack)
    for _ in range(5):
        if attack == 'swap':
            return swap_aug(phrase)
        if attack == 'delete':
            return delete_aug(phrase)
        if attack == 'insert':
            return insert_aug(phrase)
        if attack == 'keyboard':
            return keyboard(phrase)
        if attack == 'substitute':
            return substitute_aug(phrase)
        if attack == 'ocr':
            return ocr_aug(phrase)
    return phrase


def synonym_aug(phrase):
    if phrase in phrase_synonyms:return random.choice(phrase_synonyms[phrase])
    return phrase


def change_word_order(phrase):
    seq = list(nltk.wordpunct_tokenize(phrase))
    idx = range(len(seq))

    i1, i2 = random.sample(idx, 2)

    seq[i1], seq[i2] = seq[i2], seq[i1]
    return ' '.join(seq)


def parrot_aug(phrase):
    if phrase in parrot_paraphrase:return random.choice(parrot_paraphrase[phrase])
    return phrase


def get_random_aug(phrase, probs=[0.10, 0.30, 0.50, 0.05, 0.05]):
    attack_type = ['character', 'synonym', 'parrot', 'swap', 'unchange']
    attack_probs = np.array(probs)
    attack_probs = attack_probs / sum(attack_probs)
    attack = np.random.choice(attack_type, 1, p=attack_probs)[0]
    #print(phrase, attack)
    for _ in range(5):
        if attack == 'character':
            return character_aug(phrase)
        if attack == 'synonym':
            return synonym_aug(phrase)
        if attack == 'swap':
            return change_word_order(phrase)
        if attack == 'parrot':
            return parrot_aug(phrase)
    return phrase


if __name__ == '__main__':
    phrase = "Toshiki Kadomatsu"
    print(character_aug(phrase))