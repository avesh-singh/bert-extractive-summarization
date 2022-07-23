from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
data_directory = "DUCDatasets-20220708T144932Z-001/DUCDatasets/DUC2007_Summarization_Documents"
summary_directory = data_directory + "/mainEval/ROUGE"

model_summary_file = summary_directory + '/models/D0701.M.250.A.I'
peer_summary_file = "results/summary.txt"


with open(peer_summary_file, 'r') as file:
    summaries = file.read()

with open(model_summary_file, 'r') as file:
    model_summary = file.read()

print(summaries)
scores = scorer.score(model_summary, summaries)
print(scores)