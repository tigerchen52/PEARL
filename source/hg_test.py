import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def encode_text(model, input_texts):
    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    
    return embeddings


query_texts = ["The New York Times"]
doc_texts = [ "nytimes.com", "The NYT", "NYTimes", "New York Post", "New York"]
input_texts = query_texts + doc_texts

tokenizer = AutoTokenizer.from_pretrained('Lihuchen/pearl_small')
model = AutoModel.from_pretrained('Lihuchen/pearl_small')

# encode
embeddings = encode_text(model, input_texts)

# calculate similarity
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:1] @ embeddings[1:].T) * 100
print(scores.tolist())

# expected outputs
# [[90.56318664550781, 79.65763854980469, 75.52054595947266]]
