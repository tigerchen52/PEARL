# PEARL (Learning High-Quality and General-Purpose Phrase Representations)
| **[ :book: paper](https://arxiv.org/pdf/2401.10407.pdf)** |  **[🤗 PEARL-small](https://huggingface.co/Lihuchen/pearl_small)** |  **[🤗 PEARL-base](https://huggingface.co/Lihuchen/pearl_base)** | 🤗 **[PEARL-Benchmark](https://huggingface.co/datasets/Lihuchen/pearl_benchmark)** |
  **[:floppy_disk: data](https://zenodo.org/records/10676475)** |

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

Cost comparison of FastText and PEARL. The estimated memory is calculated by the number of parameters (float16). The unit of inference speed is `*ms/512 samples`. The FastText model here is `crawl-300d-2M-subword.bin`.
| Model |Avg Score| Estimated Memory |Speed GPU | Speed CPU |
|-|-|-|-|-|
|FastText|40.3|1200MB|-|57ms|
|PEARL-small|62.5|68MB|42ms|446ms|
|PEARL-base|64.8|220MB|89ms|1394ms|


## Usage
Check out our model on Huggingface: 🤗 [PEARL-small](https://huggingface.co/Lihuchen/pearl_small) 🤗 [PEARL-base](https://huggingface.co/Lihuchen/pearl_base)

```python
from sentence_transformers import SentenceTransformer, util

query_texts = ["The New York Times"]
doc_texts = [ "NYTimes", "New York Post", "New York"]
input_texts = query_texts + doc_texts

model = SentenceTransformer("Lihuchen/pearl_small")
embeddings = model.encode(input_texts)
scores = util.cos_sim(embeddings[0], embeddings[1:]) * 100
print(scores.tolist())
# [[90.56318664550781, 79.65763854980469, 75.52056121826172]]
```
## Evaluation
We evaluate phrase embeddings on a benchmark that contains 9 datasets of 5 different tasks. 🤗 [PEARL-Benchmark](https://huggingface.co/datasets/Lihuchen/pearl_benchmark) 
* **Paraphrase Classification**: PPDB and PPDBfiltered ([Wang et al., 2021](https://aclanthology.org/2021.emnlp-main.846/))
* **Phrase Similarity**: Turney ([Turney, 2012](https://arxiv.org/pdf/1309.4035.pdf)) and BIRD ([Asaadi et al., 2019](https://aclanthology.org/N19-1050/))
* **Entity Retrieval**: We constructed two datasets based on Yago ([Pellissier Tanon et al., 2020](https://hal-lara.archives-ouvertes.fr/DIG/hal-03108570v1)) and UMLS ([Bodenreider, 2004](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC308795/))
* **Entity Clustering**: CoNLL 03 ([Tjong Kim Sang, 2002](https://aclanthology.org/W02-2024/)) and BC5CDR ([Li et al., 2016](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4860626/))
* **Fuzzy Join**: AutoFJ benchmark ([Li et al., 2021](https://arxiv.org/abs/2103.04489)), which contains 50 diverse fuzzy-join datasets 

| - | PPDB | PPDB filtered |Turney|BIRD|YAGO|UMLS|CoNLL|BC5CDR|AutoFJ|
|-|-|-|-|-|-|-|-|-|-|
|Task|Paraphrase Classification|Paraphrase Classification|Phrase Similarity|Phrase Similarity|Entity Retrieval|Entity Retrieval|Entity Clustering|Entity Clustering|Fuzzy Join|
|Samples|23.4k|15.5k|2.2k|3.4k|10k|10k|5.0k|9.7k|50 subsets|
|Averaged Length|2.5|2.0|1.2|1.7|3.3|4.1|1.5|1.4|3.8|
|Metric|Acc|Acc|Acc|Pearson|Top-1 Acc|Top-1 Acc|NMI|NMI|Acc|

### Use our script to evaluate your model on PEARL benchmark
```python
python eval.py -batch_size 8
```


## Training
Download all needed training files: :inbox_tray: [Download Training Files](https://zenodo.org/records/10676475/files/train_data.zip?download=1) <br>
There are five files in total:
* `freq_phrase.txt` has more than 3M phrases
* `phrase_with_etype.txt` has the entity label for the Phrase Type Classification
* `token_aug.jsonl` has token-level augmentations
* `phrase_aug.jsonl` has phrase-level augmentations
* `hard_negative.txt` has pre-defined hard negatives

Put the downloaded files into `source/train_data`.

```python
python main.py -help
```
Once completing the data preparation and environment setup, we can train the model via `main.py`.
```python
python main.py -target_model intfloat/e5-small-v2 -dim 384
```

## Citation
If you find our paper and code useful, please give us a citation :blush:
```bibtex
@article{chen2024learning,
  title={Learning High-Quality and General-Purpose Phrase Representations},
  author={Chen, Lihu and Varoquaux, Ga{\"e}l and Suchanek, Fabian M},
  journal={arXiv preprint arXiv:2401.10407},
  year={2024}
}
```

