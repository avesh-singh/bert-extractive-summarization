import sys
import pickle
import numpy as np
from ext_sum import preprocess, load_text


def load_file(path):
    try:
        file = open(path, 'rb')
    except FileNotFoundError:
        return False, "file not found"
    contents = pickle.load(file)
    return True, contents


def label_summaries(topic):
    """
    this method reads similarity matrix for given topic
    and generates classification labels for each article
    Args:
        topic: DUC topic id

    Returns: list of labels of each article for all articles in the topic

    """
    found, response = load_file(f"prepared/{topic}_summary_similarities.pkl")
    if not found:
        print(response)
        return []
    else:
        similarities = response
    y = []

    for articles in similarities:
        best_matches = [sentences[np.argsort(sentences)[-1]] for i, sentences in enumerate(articles[0])]
        sorted_idx = np.argsort(best_matches)
        label_size = 50
        labels = [1 if i in sorted_idx[-3:] else 0 for i in range(label_size)]
        y.append(labels)
    return y


def training_batch(topic, max_pos=512):
    """
    this method prepares training batches for DUC dataset
    Args:
        topic: DUC topic id
        max_pos: max number of tokens to be read from each article

    Returns: a list of [example, label]

    """
    found, response = load_file(f"prepared/{topic}_article_sentences.pkl")
    if not found:
        print(response)
        return []
    else:
        article_sentences = response
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
