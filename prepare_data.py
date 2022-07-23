import sys
import pickle
import numpy as np
from ext_sum import preprocess, load_text

def load_file(path):
    file = open(path, 'rb')
    contents = pickle.load(file)
    return contents


def label_summaries(topic):
    similarities = load_file(f"prepared/{topic}_summary_similarities.pkl")

    y = []

    for articles in similarities:
        for article in articles[0]:
            best_matches = [sentences[np.argsort(sentences)[-1]] for i, sentences in enumerate(articles[0])]
            sorted_idx = np.argsort(best_matches)
            article_sents_in_summary = sorted_idx[-3:]
            label_size = 50
            labels = [1 if i in sorted_idx[-3:] else 0 for i in range(label_size)]
            y.append(labels)
    return y


def training_batch(topic, max_pos=512):
    article_sentences = load_file(f"prepared/{topic}_article_sentences.pkl")
    batch = []
    y = label_summaries(topic)
    for i, article in enumerate(article_sentences):
        processed = preprocess(article['sentences'])
        input_data = load_text(processed, max_pos, device="cpu")
        batch.append([input_data, y[i]])
    return batch


if __name__ == "__main__":
    args = sys.argv
    training_batch(args[1])