from os import walk
import re
from nltk.tokenize import sent_tokenize
import glob
from infer_sent import InferSent
import torch
import numpy as np
import pickle

data_dir = "DUC2006_Summarization_Documents/duc2006_docs"
article_set = "D0601"
filenames = next(walk(f"{data_dir}/{article_set}A"), (None, None, []))[2]

print(filenames)

articles = []

for article in sorted(filenames):
    with open(f"{data_dir}/{article_set}A/{article}", 'r') as f:
        articles.append({"filename": article, "article": f.read()})

article_sentences = []
for article in articles:
    body = re.split(r"<TEXT>", article["article"])

    paras = re.split(r"<P>", body[1])

    all_paras = [para.replace("</P>", "") for para in paras[:-1]]

    sentences = [sentence.replace("\n", " ") for para in all_paras for sentence in sent_tokenize(para)]
    if len(sentences) > 0:
        article_sentences.append({"filename": article["filename"], "sentences": sentences})

with open(f"{article_set}_article_sentences.pkl", 'wb') as file:
    pickle.dump(article_sentences, file)

summaries = []
for name in glob.glob(f"DUC2006_Summarization_Documents/NISTeval/ROUGE/models/{article_set}*"):
    with open(name,'r') as f:
        summary_lines = f.readlines()
        summaries += summary_lines

print(len(summaries))

with open(f"{article_set}_summary_sentences.pkl", 'wb') as file:
    pickle.dump(summaries, file)

V = 2
MODEL_PATH = 'sentence_encoder/infersent%s.pkl' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))

W2V_PATH = '../../NLP/glove.6B/glove.6B.300d.txt'
model.set_w2v_path(W2V_PATH)

all_sentences = []
for sentence in article_sentences:
    all_sentences += sentence["sentences"]

model.build_vocab(all_sentences, tokenize=True)

article_sentence_embeddings = []
for i, article in enumerate(article_sentences):
    print(f"article {i}, # sentences {len(article['sentences'])}")
    if len(article['sentences']) == 0:
        print("found 0")
    article_sentence_embeddings.append({"filename": article["filename"], "embeddings": [model.encode(article[
                                                                                                       'sentences'])]})
summary_sentence_embeddings = model.encode(summaries)

with open(f"{article_set}_article_embeddings", 'wb') as file:
    pickle.dump(article_sentence_embeddings, file)

with open(f"{article_set}_summary_embeddings.pkl", 'wb') as file:
    pickle.dump(summary_sentence_embeddings, file)


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


similarities = [[[[] for _ in range(len(summary_sentence_embeddings))] for _ in range(len(
    article_sentence_embeddings[i]["embeddings"]))] for i
 in range(len(article_sentence_embeddings))]

for i, article in enumerate(article_sentence_embeddings):
    print(article["filename"])
    for j, sentence in enumerate(article["embeddings"]):
        for k, summary in enumerate(summary_sentence_embeddings):
            similarities[i][j][k] = cosine(sentence, summary)
        if i % 100 == 0:
            print(f"completed {i} articles")

with open(f"{article_set}_summary_similarities.pkl", 'wb') as file:
    pickle.dump(similarities, file)