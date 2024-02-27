from transformers import AutoTokenizer, AutoModel
import torch

# tokenizer = AutoTokenizer.from_pretrained("Lihuchen/pearl_base")
# model = AutoModel.from_pretrained("Lihuchen/pearl_base")

# inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")

# with torch.no_grad():
#     logits = model(**inputs).logits

# # retrieve index of [MASK]
# mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

# predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
# a = tokenizer.decode(predicted_token_id)
# print(a)


from sentence_transformers import SentenceTransformer, util

query_texts = ["The New York Times"]
doc_texts = [ "nytimes.com", "The NYT", "NYTimes", "New York Post", "New York"]
input_texts = query_texts + doc_texts

model = SentenceTransformer("Lihuchen/pearl_small")
embeddings = model.encode(input_texts)
scores = util.cos_sim(embeddings[0], embeddings[1:]) * 100
print(scores.tolist())
# [[90.56318664550781, 79.65763854980469, 75.52056121826172]]
