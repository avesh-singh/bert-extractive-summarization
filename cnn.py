import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from infer_sent import InferSent
import torch
import nltk
import numpy as np
from ext_sum import load_text, preprocess


def clean(document, join=True):
    sents = [" ".join([word for word in word_tokenize(sentence) if word.isalnum()]) for sentence in
                    sent_tokenize(
        document)]
    if join:
        return ". ".join(sents)
    else:
        return sents


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def cnn_training_batch(filename, max_pos=512, items=None):
    df = pd.read_csv(f"raw_data/cnn_dailymail/{filename}.processed.csv")
    if items:
        df = df.iloc[items]
    batch = []
    y = df.labels.apply(lambda x: list(map(int, x[1:-1].split(", "))))
    for i, article in df.iterrows():
        processed = preprocess(clean(article['processed_article'], join=False))
        input_data = load_text(processed, max_pos, device="cpu")
        batch.append([input_data, y[i]])
    return batch


if __name__ == "__main__":
    stopwords = stopwords.words('english')
    filename = "input5001"
    df = pd.read_csv(f"raw_data/cnn_dailymail/{filename}.csv", header=None, names=["id", "article", 'highlights'])
    df = df.iloc[:100]
    print(df.columns)

    df["processed_article"] = df.article.apply(clean)
    df["processed_summary"] = df.highlights.apply(clean)

    V = 2
    MODEL_PATH = 'sentence_encoder/infersent%s.pkl' % V
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    model = InferSent(params_model)
    model.load_state_dict(torch.load(MODEL_PATH))

    W2V_PATH = '/media/avesh/New Volume/Study/NLP/glove.6B/glove.6B.300d.txt'
    model.set_w2v_path(W2V_PATH)

    all_sentences = []
    for index, row in df.iterrows():
        all_sentences += sent_tokenize(row['article'])
    model.build_vocab(all_sentences, tokenize=True)
    total_rows = df.shape[0]
    all_labels = []
    for i in range(df.shape[0]):
        article_sentences = sent_tokenize(df.iloc[i].processed_article)
        article_embeds = model.encode(article_sentences)
        summary_sentences = sent_tokenize(df.iloc[i].processed_summary)
        summary_embeds = model.encode(summary_sentences)

        similarities = np.array([[i, j, cosine(article, summary)] for i, article in enumerate(article_embeds) for j,
                                                                                                                  summary in enumerate(summary_embeds)])
        sorted_idx = np.argsort(similarities[:, -1])[-2:]
        labels = np.array([1 if i in similarities[sorted_idx][:1] else 0 for i in range(len(article_sentences))])
        all_labels.append(labels.T)
        if i % 100 == 0:
            print(f"on {i}/{total_rows} ")
    labels = pd.DataFrame({'labels': all_labels})
    new_df = pd.concat([df, labels], axis=1)
    new_df.to_csv(f"raw_data/cnn_dailymail/{filename}.processed.csv")
