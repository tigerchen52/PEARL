# PEARL (Learning High-Quality and General-Purpose Phrase Representations)
:book: [paper](https://arxiv.org/pdf/2401.10407.pdf)  🤗 [PEARL-small](https://huggingface.co/Lihuchen/pearl_small) 🤗 [PEARL-base](Lihuchen/pearl_base)

Our PEARL is a framework to learn phrase-level representations. <br>
If you require semantic similarity computation for strings, our PEARL model might be a helpful tool. <br>
It offers powerful embeddings suitable for tasks like string matching, entity retrieval, entity clustering, and fuzzy join. 

| Model |Size| PPDB | PPDB filtered |Turney|BIRD|YAGO|UMLS|CoNLL|BC5CDR|AutoFJ|Avg
|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| FastText  |-|  94.4  | 61.2  |  59.6  | 58.9  |16.9|14.5|3.0|0.2| 53.6|40.3|
| Sentence-BERT  |110M| 94.6  | 66.8  | 50.4  | 62.6  | 21.6|23.6|25.5|48.4| 57.2| 50.1|
| Phrase-BERT  |110M|  96.8  |  68.7  | 57.2  |  68.8  |23.7|26.1|35.4| 59.5|66.9| 54.5|
| E5-small  |34M|  96.0| 56.8|55.9| 63.1|43.3| 42.0|27.6| 53.7|74.8|57.0|
|E5-base|110M|  95.4|65.6|59.4|66.3| 47.3|44.0|32.0| 69.3|76.1|61.1|
|PEARL-small|34M|  97.0|70.2|57.9|68.1| 48.1|44.5|42.4|59.3|75.2|62.5|
|PEARL-base|110M|97.3|72.2|59.7|72.6|50.7|45.8|39.3|69.4|77.1|64.8|

## Usage
Check out our model on Huggingface: 🤗 [PEARL-small](https://huggingface.co/Lihuchen/pearl_small) 🤗 [PEARL-base](Lihuchen/pearl_base)

```python
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
doc_texts = [ "NYTimes", "New York Post", "New York"]
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
```
## Evaluation
We evaluate phrase embeddings on a benchmark that contains 9 datasets of 5 different tasks. :inbox_tray: [Download Benchmark](https://zenodo.org/records/10676475/files/eval_data.zip?download=1)
| - | PPDB | PPDB filtered |Turney|BIRD|YAGO|UMLS|CoNLL|BC5CDR|AutoFJ|
|-|-|-|-|-|-|-|-|-|-|
|Task|Paraphrase Classification|Paraphrase Classification|Phrase Similarity|Phrase Similarity|Entity Retrieval|Entity Retrieval|Entity Clustering|Entity Clustering|Fuzzy Join|
|Metric|Acc|Acc|Acc|Pearson|Top-1 Acc|Top-1 Acc|NMI|NMI|Acc|

Put the downloaded `eval_data/` into `evaluation/` dicrectory and run the script `Evaluation/eval.py` to get scores in our paper.
```python
python eval.py -batch_size 8
```

**Evaluate your custom model** <br>
You need to implement a `Module` class to generate embeddings given a list of texts, and then reuse the `eval.py`.
```python
class PearlSmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        model_name = "Lihuchen/pearl_small"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)


    def average_pool(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        

    def forward(self, x, device):
        # Tokenize the input texts
        batch_dict = self.tokenizer(x, max_length=128, padding=True, truncation=True, return_tensors='pt')
        batch_dict = batch_dict.to(device)

        outputs = self.model(**batch_dict)
        phrase_vec = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        return phrase_vec
```


The repo structure is shown below. <br>
The `data` directory contains all data needed for training and evaluation. [data](https://www.dropbox.com/scl/fi/49c87s9tm8jgf3gwmcz0e/data.zip?rlkey=g47iv7oy5fgonj6obe2d8kiq1&dl=1) <br>
The `output` directory has our model (PEARL-small), and you can use it to reproduce the results reported in Table 1. [PEARL-small](https://www.dropbox.com/scl/fi/96nui29fj6wlj7roy6pl4/output.zip?rlkey=ra0lngk9afyokpqv9xcrptjyz&dl=1)
<br>

## Training
First, you can use `-help` to show the arguments
```python
python main.py -help
```
Once completing the data preparation and environment setup, we can train the model via `main.py`.
We have also provided sample datasets, you can just run the mode without downloading.
```python
python main.py -encoder pearl_small -dataset '../data/freq_phrase.txt'
```

## Citation
If you find our paper and code useful, please give us a citation :blush:
```
@article{chen2024learning,
  title={Learning High-Quality and General-Purpose Phrase Representations},
  author={Chen, Lihu and Varoquaux, Ga{\"e}l and Suchanek, Fabian M},
  journal={arXiv preprint arXiv:2401.10407},
  year={2024}
}
```

