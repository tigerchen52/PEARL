from registry import register
from functools import partial
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch.autograd import Variable
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


registry = {}
register = partial(register, registry=registry)


@register('pearl')
class PEARL(nn.Module):
    def __init__(self, args):
        super().__init__()
        target_model = args.target_model
        dim = args.dim
        self.max_length = args.max_length
        self.device = args.device
        self.tokenizer = AutoTokenizer.from_pretrained(target_model)
        self.model = AutoModel.from_pretrained(target_model)
        self.model.to(self.device)
        self.type_fc = nn.Linear(dim, args.label_num)
        self.type_fc = self.type_fc.to(self.device)

    def average_pool(self, last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def forward(self, x, train=True):
        # Tokenize the input texts
        batch_dict = self.tokenizer(x, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt')
        batch_dict = batch_dict.to(self.device)
        outputs = self.model(**batch_dict)
        phrase_vec = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        type_vec = self.type_fc(phrase_vec)
        if train:
            return phrase_vec, type_vec
        else:
            return phrase_vec.detach()



