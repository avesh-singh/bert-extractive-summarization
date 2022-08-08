import torch
from models.model_builder import ExtSummarizer
from ext_sum import summarize
from scrape import read_articles, read_summaries
from rouge_score import rouge_scorer
from functools import reduce
from rouge_metric import PyRouge
import pickle


scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge = PyRouge(rouge_w=True, rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)
# Load model
checkpoint = torch.load(f'gru_model_1_layer.pt', map_location='cpu')
model = ExtSummarizer(device='cpu', checkpoint=checkpoint, gru_layers=1)

# data_directory = "DUC2006_Summarization_Documents/duc2006_docs"
# summary_directory = "DUC2006_Summarization_Documents/NISTeval/ROUGE"

data_directory = "DUC2007_Summarization_Documents/duc2007_testdocs/main"
summary_directory = "DUC2007_Summarization_Documents/mainEval/ROUGE"

# topics = [
#     "D0601A",
#     # "D0603C",
#     # "D0605E",
#     # "D0607G",
#     # "D0609I",
#     # "D0611B",
#     # "D0613D",
#     # "D0615F",
#     # "D0617H",
#     # "D0619A",
#     # "D0621C",
#     # "D0623E",
#     # "D0625G",
#     # "D0627I",
#     # "D0629B",
#     # "D0631D",
#     # "D0633F",
#     # "D0635H",
#     # "D0637A",
#     # "D0639C",
#     # "D0641E",
#     # "D0643G",
#     # "D0645I",
#     # # --------
#     # "D0647B",
#     # "D0649D",
#     # "D0602B",
#     # "D0604D",
#     # "D0606F",
#     # "D0608H",
#     # "D0610A",
#     # "D0612C",
#     # "D0614E",
#     # "D0616G",
#     # "D0618I",
#     # "D0620B",
#     # "D0622D",
#     # "D0624F",
#     # "D0626H",
#     # "D0628A",
#     # "D0630C",
#     # "D0632E",
#     # "D0634G",
#     # "D0636I",
#     # "D0638B",
#     # "D0640D",
#     # "D0642F",
#     # "D0644H",
#     # "D0646A",
#     # "D0648C",
#     # "D0650E"
# ]

topics = [
    "D0701A",
    "D0706B",
    "D0711C",
    "D0716D",
    "D0721E",
    "D0726F",
    "D0731G",
    "D0736H",
    "D0741I",
    "D0702A",
    "D0707B",
    "D0712C",
    "D0717D",
    "D0722E",
    "D0727G",
    "D0732H",
    "D0737I",
    "D0742J",
    "D0703A",
    "D0708B",
    # "D0713C", --
    "D0718D",
    "D0723F",
    "D0728G",
    "D0733H",
    "D0738I",
    "D0743J",
    "D0704A",
    "D0709B",
    "D0714D",
    "D0719E",
    "D0724F",
    "D0729G",
    "D0734H",
    "D0739I",
    "D0744J",
    "D0705A",
    "D0710C",
    "D0715D",
    "D0720E",
    "D0725F",
    "D0730G",
    "D0735H",
    # "D0740I", ---
    "D0745J"
]

topic_wise_scores = {}


def convert_to_json(score):
    json = {}
    for key, value in score.items():
        json[key] = {
            "precision": value.precision,
            "recall": value.recall,
            "fmeasure": value.fmeasure,
        }
    return json


def average_rouge(scores, rouge='rouge1', measure='fmeasure'):
    return reduce(lambda total, key: total + scores[key][rouge][measure], scores, 0) / len(scores)

model_summaries = {}
human_summaries = {}
for topic in topics:
    articles = read_articles(data_dir=data_directory, article_set=topic, write=False)
    topic_summaries = []
    for article in articles:
        result_fp = f"results/summary_{article['filename']}.txt"
        summary = summarize(article, result_fp, model, max_length=1)
        topic_summaries.append(summary)
    summaries = read_summaries(article_set=topic, creator=f"{summary_directory}/models")
    human_summaries[topic] = ["\n".join(topic_summary) for topic_summary in summaries]
    model_summaries[topic] = "\n".join(topic_summaries)
    scores = scorer.score(human_summaries[topic][0], model_summaries[topic])

    topic_wise_scores[topic] = scores
with open('results/2007/generated_topic_summaries.pkl', 'wb') as f:
    pickle.dump(model_summaries, f)

with open('results/2007/model_topic_summaries.pkl', 'wb') as f:
    pickle.dump(human_summaries, f)

print(rouge.evaluate(list(model_summaries.values()), list(human_summaries.values())))
scores = {k: convert_to_json(v) for k, v in topic_wise_scores.items()}

print(f"average rouge1 recall on {len(topics)} topics: {average_rouge(scores, rouge='rouge1', measure='recall')}")
print(f"average rouge1 precision on {len(topics)} topics: {average_rouge(scores, rouge='rouge1', measure='precision')}")
print(f"average rouge1 fmeasure on {len(topics)} topics: {average_rouge(scores, rouge='rouge1', measure='fmeasure')}")
print("")
print(f"average rouge2 recall on {len(topics)} topics: {average_rouge(scores, rouge='rouge2', measure='recall')}")
print(f"average rouge2 precision on {len(topics)} topics: {average_rouge(scores, rouge='rouge2', measure='precision')}")
print(f"average rouge2 fmeasure on {len(topics)} topics: {average_rouge(scores, rouge='rouge2', measure='fmeasure')}")
print("")
print(f"average rougeL recall on {len(topics)} topics: {average_rouge(scores, rouge='rougeL', measure='recall')}")
print(f"average rougeL precision on {len(topics)} topics: {average_rouge(scores, rouge='rougeL', measure='precision')}")
print(f"average rougeL fmeasure on {len(topics)} topics: {average_rouge(scores, rouge='rougeL', measure='fmeasure')}")
