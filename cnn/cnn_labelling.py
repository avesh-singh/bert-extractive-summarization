from nltk.tokenize import sent_tokenize
import pandas as pd
from rouge_score import rouge_scorer
import operator


def main(file):
    """
    this method labels cnn article sentences on whether
    each sentence should come in summary or not. To determine
    these sentences, article sentences are picked one by one,
    and gain in ROUGE value is calculated. Article sentence which
    improves this value the most is labelled as 1 meaning 'in the
    summary'. Article sentences are not picked once either of the
    two conditions are satisfied
     - total 5 sentences are already picked from the article
     - adding a new sentence does not improve the ROUGE score
    Args:
        file: filename to be processed

    Returns:
        DataFrame: processed CNN/DailyMail articles with
        labels
    """
    df = pd.read_csv(f"raw_data/cnn_dailymail/{file}.csv", header=None, names=["id", "article", 'highlights'])
    scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
    labels = []
    all_article_sentences = []
    for i, row in df.iterrows():
        article_sentences = sent_tokenize(row.article)
        all_article_sentences.append(article_sentences)
        final_rouge = 0
        best_summary = ""
        improver_sentences = []
        while len(improver_sentences) <= 5:
            j = 0
            marginal_rouge = []
            for k in range(j, len(article_sentences)):
                test_summary = best_summary
                test_summary += "\n" + article_sentences[k]
                score = scorer.score(row.highlights, test_summary)
                marginal_rouge.append(score['rouge2'].fmeasure)
            index, value = max(enumerate(marginal_rouge), key=operator.itemgetter(1))
            if final_rouge >= value:
                break
            else:
                improver_sentences.append(index)
                best_summary += "\n" + article_sentences[index]
                final_rouge = value
                j += 1
        if i % 100 == 0:
            print(f"processed article {i}")
        labels.append([1 if i in improver_sentences else 0 for i in range(len(article_sentences))])
    df['processed_article'] = all_article_sentences
    df['processed_summary'] = df.highlights.apply(sent_tokenize)
    df['labels'] = pd.Series(labels)
    df.to_csv(f"raw_data/cnn_dailymail/{file}.rouge.processed.csv")
    return df


if __name__ == '__main__':
    filename = "input1"
    main(filename)
