from registry import register
from functools import partial
import torch.nn as nn
import torch
import math
import fasttext
from love_model import registry as Producer
import numpy as np
from torch import Tensor
from torch.autograd import Variable
import torch.nn.functional as F
from uctopic import UCTopicTokenizer, UCTopic, UCTopicTool
from transformers import AutoTokenizer, BertModel, AutoModel
from sentence_transformers import SentenceTransformer
from love_inference import args as love_args, collate_fn_predict

registry = {}
register = partial(register, registry=registry)


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def gen_love_emb(love_model, x, mask):
    batch_repre_ids = torch.transpose(x, 1, 0)
    mask = torch.transpose(mask, 1, 0)
    love_embs = list()
    for i in range(batch_repre_ids.size()[0]):
        temp = batch_repre_ids[i]
        temp_mask = mask[i]
        emb = love_model(temp, temp_mask)
        love_embs.append(emb)
    love_embs = torch.stack(love_embs)
    love_embs = torch.mean(love_embs, dim=0)
    return love_embs


@register('pearl_small')
class PhraseBERT(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_name = 'intfloat/e5-small-v2'
        dim = 384
        self.device = args.device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)

        self.love_model = Producer[love_args.model_type](love_args)
        self.love_model.load_state_dict(torch.load('../data/love_model.pt'))
        self.love_model.to(self.device)

        self.fusion = nn.Sequential(
            nn.Linear(300, 96),
        )
        self.fusion = self.fusion.to(self.device)

        self.type_fc = nn.Linear(dim+96, 95)
        self.type_fc = self.type_fc.to(self.device)

    def forward(self, x, train=True):
        # Tokenize the input texts
        batch_dict = self.tokenizer(x, max_length=128, padding=True, truncation=True, return_tensors='pt')
        batch_dict = batch_dict.to(self.device)

        batch_repre_ids, mask = collate_fn_predict(x)
        batch_repre_ids = batch_repre_ids.to(self.device)
        mask = mask.to(self.device)
        #love_emb = self.love_model(batch_repre_ids, mask)
        love_emb = gen_love_emb(self.love_model, batch_repre_ids, mask)

        outputs = self.model(**batch_dict)
        phrase_vec = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        phrase_vec = torch.cat([phrase_vec, self.fusion(love_emb)], dim=-1)

        type_vec = self.type_fc(phrase_vec)
        if train:
            return phrase_vec, type_vec
        else:
            return phrase_vec.detach()


@register('pearl_base')
class PhraseBERT(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_name = 'intfloat/e5-base-v2'
        dim = 768
        self.device = args.device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)

        self.love_model = Producer[love_args.model_type](love_args)
        self.love_model.load_state_dict(torch.load('./data/love_model.pt'))
        self.love_model.to(self.device)

        self.projection = nn.Sequential(
            nn.Linear(300, 288),
        )
        self.projection = self.projection.to(self.device)

        self.type_fc = nn.Linear(dim+288, 95)
        self.type_fc = self.type_fc.to(self.device)

    def forward(self, x, train=True):
        # Tokenize the input texts
        batch_dict = self.tokenizer(x, max_length=128, padding=True, truncation=True, return_tensors='pt')
        batch_dict = batch_dict.to(self.device)

        batch_repre_ids, mask = collate_fn_predict(x)
        batch_repre_ids = batch_repre_ids.to(self.device)
        mask = mask.to(self.device)
        #love_emb = self.love_model(batch_repre_ids, mask)
        love_emb = gen_love_emb(self.love_model, batch_repre_ids, mask)

        outputs = self.model(**batch_dict)
        phrase_vec = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        phrase_vec = torch.cat([phrase_vec, self.projection(love_emb)], dim=-1)

        type_vec = self.type_fc(phrase_vec)
        if train:
            return phrase_vec, type_vec
        else:
            return phrase_vec.detach()


@register('love')
class PhraseBERT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.love_model = Producer[love_args.model_type](love_args)
        self.love_model.load_state_dict(torch.load('./data/love_model.pt'))
        self.love_model.to(self.device)


    def forward(self, x, train=True):


        batch_repre_ids, mask = collate_fn_predict(x)
        batch_repre_ids = batch_repre_ids.to(self.device)
        mask = mask.to(self.device)
        #love_emb = self.love_model(batch_repre_ids, mask)
        love_emb = gen_love_emb(self.love_model, batch_repre_ids, mask)

        return love_emb


@register('fasttext')
class PhraseBERT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.word_embedding_model = fasttext.load_model('wiki.en.bin')

    def forward(self, x, train=True):

        import nltk
        temp = list()
        for e in x:
            embedding = np.array([self.word_embedding_model.get_word_vector(token)  for token in nltk.wordpunct_tokenize(e)])
            embedding = np.mean(embedding, axis=0)
            #print(np.shape(embedding))
            temp.append(embedding)
        phrase_vec = np.array(temp)
        return torch.from_numpy(phrase_vec)


@register('e5_base')
class PhraseBERT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
        self.model = AutoModel.from_pretrained('intfloat/e5-base-v2')
        self.model.to(self.device)


    def forward(self, x, train=True):
        # Tokenize the input texts
        batch_dict = self.tokenizer(x, max_length=128, padding=True, truncation=True, return_tensors='pt')
        batch_dict = batch_dict.to(self.device)

        outputs = self.model(**batch_dict)
        phrase_vec = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        if train:
            return phrase_vec
        else:
            return phrase_vec.detach()


@register('phrase_bert')
class PhraseBERT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = SentenceTransformer('whaleloops/phrase-bert')

    def forward(self, x, train=True):
        phrase_embs = self.model.encode(x)
        return torch.tensor(phrase_embs)



def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


@register('sentence_bert')
class PhraseBERT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        model_name = 'sentence-transformers/bert-large-nli-mean-tokens'
        self.model = SentenceTransformer(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, x, train=True):
        encoded_input = self.tokenizer(x, padding=True, truncation=True, return_tensors='pt')


        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        #batch_dict = self.tokenizer(x, max_length=512, padding=True, truncation=True, return_tensors='pt')
        # batch_dict = batch_dict.to(self.device)
        # outputs = self.model(**batch_dict)
        # phrase_embs = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        return sentence_embeddings


@register('uctopic')
class UCTopicModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        #self.tokenizer = UCTopicTokenizer.from_pretrained('JiachengLi/uctopic-base')
        #self.model = UCTopicTool('JiachengLi/uctopic-base', device='cuda:0')
        self.device = args.device
        self.tokenizer = UCTopicTokenizer.from_pretrained('JiachengLi/uctopic-base')
        self.model = UCTopic.from_pretrained('JiachengLi/uctopic-base')

    def forward(self, x, train=True):
        #phrases = [[name, (0, len(name))] for name in x]
        entity_spans = [[(0, len(name))] for name in x]
        inputs = self.tokenizer(x, entity_spans=entity_spans, add_prefix_space=True, return_tensors="pt", padding=True)
        #inputs.to(device=self.device)
        outputs, phrase_embs = self.model(**inputs)
        #phrase_embs = self.model.encode(phrases)
        if train:
            return phrase_embs
        else:
            return phrase_embs.detach()


class PAMCharEcoder(nn.Module):
    def __init__(self):
        super(PAMCharEcoder, self).__init__()
        self.dim = 72
        self.encoder_layer = 1
        self.head = 1
        self.drop_rate = 0.1
        self.embedding = nn.Embedding(21257, self.dim, padding_idx=0)
        self.embedding.weight.requires_grad = True

        self.encoders = nn.ModuleList([Pamelaformer() for _ in range(self.encoder_layer)])
        self.sublayer = SublayerConnection(self.drop_rate, self.dim)

    def forward(self, x, mask):
        x = self.embedding(x)
        shape = list(x.size())
        position = PositionalEncoding(shape[-1], shape[-2])
        pos_att = position(x)

        for i, encoder in enumerate(self.encoders):
            x = self.sublayer(x, lambda x: encoder(x, mask, pos_att))

        x = x.masked_fill_(~mask, 0).sum(dim=1)
        return l2norm(x)


def l2norm(x):
    return x / x.norm(p=2, dim=1, keepdim=True)


class Pamelaformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_attention = PAM()
        dim = 72
        proj_dim = 72
        self.projection = nn.Sequential(
            nn.Linear(proj_dim, dim),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, mask, position):
        pos = self.pos_attention(x, mask, position)
        c = self.projection(pos)
        return c


class PAM(nn.Module):
    def __init__(self):
        super().__init__()
        dim = 72
        self.head = 72
        self.projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, mask, pos):
        q = pos
        k = pos
        v = x
        q, k, v = (split_last(a, (self.head, -1)).transpose(1, 2) for a in [q, k, v])
        scores = torch.matmul(q, k.transpose(2, 3)) / (k.size(-1) ** 0.25)
        mask = torch.matmul(mask.float(), mask.transpose(1, 2).float()).bool()
        mask = mask.unsqueeze(1)
        mask = mask.repeat([1, self.head, 1, 1])
        scores.masked_fill_(~mask, -1e7)

        scores = F.softmax(scores, dim=2)
        scores = scores.transpose(2, 3)
        v_ = torch.matmul(scores, v)
        v_ = v_.transpose(1, 2).contiguous()
        v_ = merge_last(v_, 2)
        v_ = self.projection(v_)
        return v_


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, dropout, dim):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(self.norm(sublayer(x)))


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionalAttCached(nn.Module):
    def __init__(self, d_model, pos_attns, max_len=5000):
        super(PositionalAttCached, self).__init__()
        # Compute the positional encodings once in log space.
        self.d_model = d_model
        self.pos_attns = pos_attns
        self.max_len = max_len

    def forward(self, x):
        shape = list(x.size())
        pos_attn = self.pos_attns[shape[1]]
        p_e = Variable(pos_attn, requires_grad=False).cuda()
        p_e = p_e.repeat([shape[0], 1, 1])
        return p_e


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        position = position * 1
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        shape = list(x.size())
        p_e = Variable(self.pe[:, :x.size(1)], requires_grad=False).cuda()
        p_e = p_e.repeat([shape[0], 1, 1])
        return p_e
