import pickle
import numpy as np
from ext_sum import preprocess, load_text

def load_file(path):
    file = open(path, 'rb')
    contents = pickle.load(file)
    return contents


similarities = load_file("D0601_summary_similarities.pkl")
article_sentences = load_file("D0601_article_sentences.pkl")

y = []

for articles in similarities:
    for article in articles[0]:
        best_matches = [sentences[np.argsort(sentences)[-1]] for i, sentences in enumerate(articles[0])]
        sorted_idx = np.argsort(best_matches)
        article_sents_in_summary = sorted_idx[-3:]

        labels = [1 if i in sorted_idx[-3:] else 0 for i in range(len(sorted_idx))]
        y.append(labels)


def training_batch(max_pos=512):
    batch = []
    for i, article in enumerate(article_sentences):
        processed = preprocess(article)
        input_data = load_text(processed, max_pos, device="cpu")
        batch.append([input_data, y[i]])
    return batch
