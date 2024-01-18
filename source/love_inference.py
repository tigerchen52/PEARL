import torch
import sys
from nltk import wordpunct_tokenize
import argparse
import tokenization


#hyper-parameters
parser = argparse.ArgumentParser(description='contrastive learning framework for word vector')
parser.add_argument('-dataset', help='the file of target vectors', type=str, default='data/wiki_100.vec')
parser.add_argument('-batch_size', help='the number of samples in one batch', type=int, default=32)
parser.add_argument('-epochs', help='the number of epochs to train the model', type=int, default=20)
parser.add_argument('-shuffle', help='whether shuffle the samples', type=bool, default=True)
parser.add_argument('-lowercase', help='if only use lower case', type=bool, default=True)
parser.add_argument('-model_type', help='sum, rnn, cnn, attention, pam', type=str, default='pam')
parser.add_argument('-encoder_layer', help='the number of layer of the encoder', type=int, default=1)
parser.add_argument('-merge', help='merge pam and attention layer', type=bool, default=False)
parser.add_argument('-att_head_num', help='the number of attentional head for the pam encoder', type=int, default=1)
parser.add_argument('-loader_type', help='simple, aug, hard', type=str, default='hard')
parser.add_argument('-loss_type', help='mse, ntx, align_uniform', type=str, default='ntx')
parser.add_argument('-input_type', help='mixed, char, sub', type=str, default='mixed')
parser.add_argument('-learning_rate', help='learning rate for training', type=float, default=2e-3)
parser.add_argument('-drop_rate', help='the rate for dropout', type=float, default=0.1)
parser.add_argument('-gamma', help='decay rate', type=float, default=0.97)
parser.add_argument('-emb_dim', help='the dimension of target embeddings (FastText:300; BERT:768)', type=int, default=300)
parser.add_argument('-vocab_path', help='the vocabulary used for training and inference', type=str, default='../data/vocab.txt')
parser.add_argument('-hard_neg_numbers', help='the number of hard negatives in each mini-batch', type=int, default=3)
parser.add_argument('-hard_neg_path', help='the file path of hard negative samples ', type=str, default='data/hard_neg_samples.txt')
parser.add_argument('-vocab_size', help='the size of the vocabulart', type=int, default=0)

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)


tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_path, do_lower_case=args.lowercase)
vocab_size = len(tokenizer.vocab)
args.vocab_size = vocab_size


def tokenize_and_getid(word):
    tokens = tokenizer.tokenize(tokenizer.convert_to_unicode(word))
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokens, token_ids


def repre_word(word, rtype='mixed'):
    start = '[CLS]'
    sub = '[SUB]'
    end = '[SEP]'
    char_seq = list(word)
    tokens, _ = tokenize_and_getid(word)

    if rtype == 'mixed':
        repre = [start] + char_seq + [sub] + tokens + [end]
    elif rtype == 'char':
        repre = [start] + char_seq + [end]
    else:
        repre = [start] + tokens + [end]
    repre_ids = tokenizer.convert_tokens_to_ids(repre)

    return repre, repre_ids


def collate_fn_predict(batch_data, rtype='mixed', pad=0):
    batch_words = [list(wordpunct_tokenize(phrase)) for phrase in batch_data]
    max_words_len = max([len(seq) for seq in batch_words])
    batch_data = [words + ['[PAD]'] * (max_words_len - len(words)) for words in batch_words]
    batch_repre_ids = list()
    for words in batch_data:
        temp_repre_id = list()
        for word in words:
            repre, repre_id = repre_word(word, rtype=rtype)
            temp_repre_id.append(repre_id)
        batch_repre_ids.append(temp_repre_id)

    max_len = max([len(chars) for seq in batch_repre_ids for chars in seq])
    for index, word_seq in enumerate(batch_repre_ids):
        batch_repre_ids[index] = [char + [pad]*(max_len - len(char)) for char in word_seq]

    batch_repre_ids = torch.LongTensor(batch_repre_ids)
    mask = torch.ne(batch_repre_ids, pad).unsqueeze(3)
    return batch_repre_ids, mask



