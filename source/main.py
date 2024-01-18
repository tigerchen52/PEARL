import random
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
from loader import registry as loader
from model import registry as encoder
from loss import registry as loss_func



def train_and_eval():
    parser = argparse.ArgumentParser(description='contrastive learning framework for word vector')
    parser.add_argument('-dataset', help='the file of target vectors', type=str, default='../data/freq_phrase.txt')
    parser.add_argument('-lowercase', help='if only use lower case', type=bool, default=False)
    parser.add_argument('-shuffle', help='whether shuffle the samples', type=bool, default=True)
    parser.add_argument('-batch_size', help='the number of samples in one batch', type=int, default=32)
    parser.add_argument('-encoder', help='model type', type=str, default='pearl_small')
    parser.add_argument('-loss_type', help='mse, ntx, align_uniform', type=str, default='ntx')
    parser.add_argument('-learning_rate', help='learning rate for training', type=float, default=3e-5)
    parser.add_argument('-gamma', help='decay rate', type=float, default=0.98)
    parser.add_argument('-epochs', help='the number of epochs to train the model', type=int, default=2)
    parser.add_argument('-hard_neg_numbers', help='the number of hard negatives in each mini-batch', type=int, default=2)
    parser.add_argument('-hard_neg_path', help='the file path of hard negative samples ', type=str, default='../data/hard_negative_test.txt')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    args.device = device

    model = encoder[args.encoder](args)

    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('the number of trainable parameters = {a}'.format(a=trainable_num))

    criterion = loss_func[args.loss_type]()
    ptype_criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    model.train()
    for e in range(args.epochs):
        epoch_loss = 0
        batch_num = 0

        data_loader = loader['phrase_loader'](args)
        train_iterator = data_loader(data_path=args.dataset)

        for ori_phrase, aug_phrase, _, _, _, _, p_type in train_iterator:
            p_type = p_type.to(device)
            optimizer.zero_grad()
            batch_num += 1

            if batch_num % 100 == 0:
                print('sample = {b}, loss = {a}'.format(a=epoch_loss / batch_num, b=batch_num * args.batch_size))

            if batch_num % 2000 == 0:
                scheduler.step()

            origin_outputs, origin_type = model(ori_phrase)
            aug_outputs, aug_type = model(aug_phrase)

            # calculate loss
            phrase_loss = criterion(origin_outputs, aug_outputs)

            p_type_loss_origin, p_type_loss_aug = ptype_criterion(origin_type, p_type), ptype_criterion(aug_type, p_type)

            loss = phrase_loss + p_type_loss_aug + p_type_loss_origin

            # backward
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print('[ lr rate] = {a}'.format(a=optimizer.state_dict()['param_groups'][0]['lr']))
        print('----------------------')
        print('this is the {a} epoch, loss = {b}'.format(a=e + 1, b=epoch_loss / len(train_iterator)))

    torch.save(model.state_dict(), './output/model.pt')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    setup_seed(42)
    train_and_eval()

