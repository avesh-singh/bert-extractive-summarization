from os import walk
import re
from nltk.tokenize import sent_tokenize
import glob
from infer_sent import InferSent
import torch
import numpy as np
import pickle
import sys

prepared_dir = "prepared"


def read_articles(article_set="D0601A", data_dir="DUC2006_Summarization_Documents/duc2006_docs", write=True):
    filenames = next(walk(f"{data_dir}/{article_set}"), (None, None, []))[2]
    print(filenames)

    articles = []

    for article in sorted(filenames):
        with open(f"{data_dir}/{article_set}/{article}", 'r') as f:
            articles.append({"filename": article, "article": f.read()})

    article_sentences = []
    for article in articles:
        body = re.split(r"<TEXT>", article["article"])

        paras = re.split(r"<P>", body[1])

        all_paras = [para.replace("</P>", "") for para in paras[:-1]]

        sentences = [sentence.replace("\n", " ").strip() for para in all_paras for sentence in sent_tokenize(para)]
        if len(sentences) > 0:
            article_sentences.append({"filename": article["filename"], "sentences": sentences})
    if write:
        with open(f"{prepared_dir}/{article_set}_article_sentences.pkl", 'wb') as file:
            pickle.dump(article_sentences, file)

    return article_sentences


def read_summaries(article_set="D0601A"):
    summaries = []
    for name in glob.glob(f"DUC2006_Summarization_Documents/NISTeval/ROUGE/models/{article_set[:-1]}*"):
        with open(name, 'r') as f:
            summary_lines = f.readlines()
            summaries += [line.strip() for line in summary_lines]
    return summaries


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def calculate_similarities(article_set, article_sentences, summaries):
    V = 2
    MODEL_PATH = 'sentence_encoder/infersent%s.pkl' % V
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    model = InferSent(params_model)
    model.load_state_dict(torch.load(MODEL_PATH))

    W2V_PATH = '/media/avesh/New Volume/Study/NLP/glove.6B/glove.6B.300d.txt'
    model.set_w2v_path(W2V_PATH)

    all_sentences = []
    for sentence in article_sentences:
        all_sentences += sentence["sentences"]

    model.build_vocab(all_sentences, tokenize=True)

    article_sentence_embeddings = []
    for i, article in enumerate(article_sentences):
        print(f"article {article['filename']}, # sentences {len(article['sentences'])}")
        if len(article['sentences']) == 0:
            print("found 0")
        article_sentence_embeddings.append({"filename": article["filename"], "embeddings": [model.encode(article[
                                                                                                           'sentences'])]})
    summary_sentence_embeddings = model.encode(summaries)
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

    with open(f"{prepared_dir}/{article_set}_summary_similarities.pkl", 'wb') as file:
        pickle.dump(similarities, file)


if __name__ == "__main__":
    args = sys.argv
    if len(args) == 1:
        topics = [
            # "D0601A",
            # "D0603C",
            #     "D0605E",
                # "D0607G",
                # "D0609I",
                # "D0611B",
                # "D0613D",
                # "D0615F",
                # "D0617H",
                # "D0619A",
                # "D0621C",
                # "D0623E",
                # "D0625G",
                # "D0627I",
                # "D0629B",
                # "D0631D",
                # "D0633F",
                # "D0635H",
                # "D0637A",
                # "D0639C",
                # "D0641E",
                # "D0643G",
                # "D0645I",
                "D0647B",
                "D0649D",
                "D0602B",
                "D0604D",
                "D0606F",
                "D0608H",
                "D0610A",
                "D0612C",
                "D0614E",
                "D0616G",
                "D0618I",
                "D0620B",
                "D0622D",
                "D0624F",
                "D0626H",
                "D0628A",
                    "D0630C",
                    "D0632E",
                    "D0634G",
                    "D0636I",
                    "D0638B",
                    "D0640D",
                    "D0642F",
                    "D0644H",
                    "D0646A",
                    "D0648C",
                    "D0650E"]
    else:
        topics = [args[1]]
    for topic in topics:
        articles = read_articles(topic)
        summaries = read_summaries(topic)
        calculate_similarities(topic, articles, summaries)
