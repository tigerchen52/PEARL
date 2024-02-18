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

| - | PPDB | PPDB filtered |Turney|BIRD|YAGO|UMLS|CoNLL|BC5CDR|AutoFJ|
|-|-|-|-|-|-|-|-|-|-|
|Task|Paraphrase Classification|Paraphrase Classification|Phrase Similarity|Phrase Similarity|Entity Retrieval|Entity Retrieval|Entity Clustering|Entity Clustering|Fuzzy Join|
|Metric|Acc|Acc|Acc|Pearson|Top-1 Acc|Top-1 Acc|NMI|NMI|Acc|

The repo structure is shown below. <br>
The `data` directory contains all data needed for training and evaluation. [data](https://www.dropbox.com/scl/fi/49c87s9tm8jgf3gwmcz0e/data.zip?rlkey=g47iv7oy5fgonj6obe2d8kiq1&dl=1) <br>
The `output` directory has our model (PEARL-small), and you can use it to reproduce the results reported in Table 1. [PEARL-small](https://www.dropbox.com/scl/fi/96nui29fj6wlj7roy6pl4/output.zip?rlkey=ra0lngk9afyokpqv9xcrptjyz&dl=1)
<br>
The `source` directory includes all the source code for our framework.
```
├───data
│   │   autofj.md
│   │   bird.txt
│   │   freq_phrase.txt
│   │   hard_negative_test.txt
│   │   love_model.pt
│   │   phrase_aug_test.jsonl
│   │   phrase_with_etype.txt
│   │   ppdb.json
│   │   ppdb_filtered.json
│   │   token_aug_test.jsonl
│   │   turney.txt
│   │   vocab.txt
│   │
│   ├───bc5cdr
│   │       test.json
│   │
│   ├───conll2003
│   │       test.json
│   │
│   ├───umls
│   │       umls_kb.txt
│   │       umls_test.txt
│   │
│   └───yago
│           yago_kb.txt
│           yago_test.txt
│
├───output
└───source
    │   augmentation.py
    │   clean.py
    │   evaluation.py
    │   loader.py
    │   loss.py
    │   love_inference.py
    │   love_model.py
    │   main.py
    │   model.py
    │   registry.py
    │   tokenization.py
    │   utils.py
    │   __init__.py
```
## Data Preparation
* All dataset and training corpus: [download](https://www.dropbox.com/scl/fi/49c87s9tm8jgf3gwmcz0e/data.zip?rlkey=g47iv7oy5fgonj6obe2d8kiq1&dl=1)
* [PEARL-small](https://www.dropbox.com/scl/fi/96nui29fj6wlj7roy6pl4/output.zip?rlkey=ra0lngk9afyokpqv9xcrptjyz&dl=1)

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
## Reproduction
To show the experimental results in Table 1, you can use the following command and 
we have provided the trained model we used in our paper. 

```python
python run_eval.py

## expected output
model parameters：~40M
bird_score = 0.6983
turney_score = 0.5623
conll_nmi = 0.4829
bc5cdr_nmi = 0.6240
ppdb_acc = 0.9745
ppdb_filtered_acc = 0.6876
autofj_acc = 0.7471
yago_acc = 0.4813
umls_acc = 0.4347

```


