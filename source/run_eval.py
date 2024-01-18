from loader import registry as loader
from model import registry as encoder
from loss import registry as loss_func
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from evaluation import eval_bird, eval_turney, encode_in_batch, eval_ppdb, eval_clustering, eval_retrieval, run_autofj_eval
import argparse


def run_eval():
    parser = argparse.ArgumentParser(description='contrastive learning framework for word vector')
    parser.add_argument('-lowercase', help='if only use lower case', type=bool, default=False)
    parser.add_argument('-batch_size', help='the number of samples in one batch', type=int, default=200)
    parser.add_argument('-encoder', help='model type', type=str, default='pearl_small')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    args.device = device

    model = encoder[args.encoder](args)
    model.load_state_dict(torch.load('../output/pearl_small.pt'))

    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('the number of trainable parameters = {a}'.format(a=trainable_num))


    print('start!')
    bird_score = eval_bird(model)
    turney_score = eval_turney(model)
    conll_nmi = eval_clustering(model, path='../data/conll2003/')
    bc5cdr_nmi = eval_clustering(model, path='../data/bc5cdr/')
    ppdb_acc = eval_ppdb(model, emb_batch_size=8, path='../data/ppdb.json')
    ppdb_filtered_acc = eval_ppdb(model, emb_batch_size=8, path='../data/ppdb_filtered.json')
    autofj_acc = run_autofj_eval(model)
    yago_acc = eval_retrieval(model, kb_path='../data/yago/yago_kb.txt', test_path='data/yago/yago_test.txt')
    umls_acc = eval_retrieval(model, kb_path='../data/umls/umls_kb.txt', test_path='data/umls/umls_test.txt')

    print('--------------------------------------------------')

    print('bird_score = {a}'.format(a=bird_score))
    print('turney_score = {a}'.format(a=turney_score))
    print('conll_nmi = {a}'.format(a=conll_nmi))
    print('bc5cdr_nmi = {a}'.format(a=bc5cdr_nmi))
    print('ppdb_acc = {a}'.format(a=ppdb_acc))
    print('ppdb_filtered_acc = {a}'.format(a=ppdb_filtered_acc))
    print('autofj_acc = {a}'.format(a=autofj_acc))
    print('yago_acc = {a}'.format(a=yago_acc))
    print('umls_acc = {a}'.format(a=umls_acc))

    print('--------------------------------------------------')



if __name__ == '__main__':
    run_eval()


