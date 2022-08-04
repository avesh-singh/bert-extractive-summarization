import pandas as pd
from ext_sum import load_text, preprocess
from cnn.cnn import clean
from ext_sum import summarize
from rouge_score import rouge_scorer
import torch
from models.model_builder import ExtSummarizer
from functools import reduce

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


def get_model(is_train):
    if is_train:
        checkpoint = torch.load(f'checkpoints/gru-bert/model_checkpoints_1659608782.tar', map_location='cpu')
        return ExtSummarizer(device='cpu', checkpoint=checkpoint['model_state_dict'], gru_layers=2)
    else:
        checkpoint = torch.load(f'gru_cnn_model.pt', map_location='cpu')
        return ExtSummarizer(device='cpu', checkpoint=checkpoint, gru_layers=2)


def read_articles_and_human_summaries(filename, portion, max_pos=512):
    df = pd.read_csv(f"raw_data/cnn_dailymail/{filename}")
    if portion:
        df = df.iloc[portion]
    batch = []
    for i, article in df.iterrows():
        processed = preprocess(clean(article['processed_article'], join=False))
        input_data = load_text(processed, max_pos, device="cpu")
        batch.append([input_data, article['highlights']])
    return batch


def compute_summary(article, model, filename=''):
    result_fp = f"results/{filename}.txt"
    summary = summarize(article, result_fp, model, max_length=4, write_summary=False)
    return summary


if __name__ == '__main__':
    articles_and_summaries = read_articles_and_human_summaries('input5001.100.processed.csv', slice(-200, None,
                                                                                                        None))
    article_scores = []
    articles = []
    for i in range(len(articles_and_summaries)):
        articles.append({'sentences': articles_and_summaries[i][0][-1][0], 'summary': articles_and_summaries[i][1]})
    model = get_model(is_train=True)
    for i in range(len(articles)):
        model_summary = compute_summary(articles[i], model)
        score = scorer.score(articles[i]['summary'], model_summary)
        article_scores.append(score)
    print(article_scores)
    print(reduce(lambda y, x: y+x, map(lambda x: x['rouge1'].fmeasure, article_scores)) / len(article_scores))
