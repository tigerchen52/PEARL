# PEARL
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
* All dataset and training corpus: [downlaod](https://www.dropbox.com/scl/fi/49c87s9tm8jgf3gwmcz0e/data.zip?rlkey=g47iv7oy5fgonj6obe2d8kiq1&dl=1)
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
