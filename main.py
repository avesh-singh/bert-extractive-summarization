import torch
from models.model_builder import ExtSummarizer
from ext_sum import summarize, summarize_v1
from os import walk
from scrape import read_articles
# Load model
checkpoint = torch.load(f'gru_model.pt', map_location='cpu')
model = ExtSummarizer(device='cpu')

# Run summarization
root_directory = "/home/avesh/Documents/summarization/bert-extractive-summarization/DUCDatasets-20220708T144932Z-001" \
                 "/DUCDatasets/DUC2007_Summarization_Documents/duc2007_testdocs/main"
article_set = "D0701A"
# root_directory = "/home/avesh/Documents/summarization/bert-extractive-summarization/DUC2006_Summarization_Documents/" \
#                  "duc2006_docs"
# article_set = "D0601A"
filenames = next(walk(f"{root_directory}/{article_set}"), (None, None, []))[2]
print(filenames)
articles = read_articles(article_set, root_directory, False)
all_summaries = []
for article in articles:
    # input_fp = f"{root_directory}/{article_set}/{article}"
    result_fp = f"results/summary_{article['filename']}.txt"
    summary = summarize_v1(article, result_fp, model, max_length=1)
    all_summaries.append(summary)
    print(summary)

with open("results/summary.txt", 'w') as file:
    file.write("\n".join(all_summaries))
