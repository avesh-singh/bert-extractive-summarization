import torch
from models.model_builder import ExtSummarizer
from ext_sum import summarize
from os import walk
from scrape import read_articles, read_summaries
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
# Load model
checkpoint = torch.load(f'gru_model_1_layer.pt', map_location='cpu')
model = ExtSummarizer(device='cpu', checkpoint=checkpoint, gru_layers=1)

data_directory = "DUC2006_Summarization_Documents/duc2006_docs"
summary_directory = "DUC2006_Summarization_Documents/NISTeval/ROUGE"

topics = [
    "D0601A",
    "D0603C",
    "D0605E",
    "D0607G",
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
    # # --------
    # "D0647B",
    # "D0649D",
    # "D0602B",
    # "D0604D",
    # "D0606F",
    # "D0608H",
    # "D0610A",
    # "D0612C",
    # "D0614E",
    # "D0616G",
    # "D0618I",
    # "D0620B",
    # "D0622D",
    # "D0624F",
    # "D0626H",
    # "D0628A",
    # "D0630C",
    # "D0632E",
    # "D0634G",
    # "D0636I",
    # "D0638B",
    # "D0640D",
    # "D0642F",
    # "D0644H",
    # "D0646A",
    # "D0648C",
    # "D0650E"
    ]

topic_wise_scores = {}
for topic in topics:
    articles = read_articles(data_dir=data_directory, article_set=topic, write=False)
    topic_summaries = []
    for article in articles:
        result_fp = f"results/summary_{article['filename']}.txt"
        summary = summarize(article, result_fp, model, max_length=1)
        topic_summaries.append(summary)
    summaries = read_summaries(topic, False)
    human_summaries = "\n".join(summaries)
    model_summaries = "\n".join(topic_summaries)
    scores = scorer.score(human_summaries, model_summaries)
    topic_wise_scores[topic] = scores
    print(scores)

print(topic_wise_scores)
