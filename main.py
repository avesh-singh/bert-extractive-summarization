import torch
from models.model_builder import ExtSummarizer
from ext_sum import summarize
from os import walk
from scrape import read_articles
# Load model
checkpoint = torch.load(f'checkpoints/gru-bert/model_checkpoints_1659519102.tar', map_location='cpu')
model = ExtSummarizer(device='cpu', checkpoint=checkpoint['model_state_dict'])
# checkpoint = torch.load(f'gru_model_1_layer.pt', map_location='cpu')
# model = ExtSummarizer(device='cpu', checkpoint=checkpoint)

# Run summarization
root_directory = "DUC2006_Summarization_Documents/duc2006_docs"
article_set = "D0601A"
filenames = next(walk(f"{root_directory}/{article_set}"), (None, None, []))[2]
print(filenames)
articles = read_articles(article_set, root_directory, False)
all_summaries = []
for article in articles:
    result_fp = f"results/summary_{article['filename']}.txt"
    summary = summarize(article, result_fp, model, max_length=1)
    all_summaries.append(summary)
    print(summary)

with open("results/summary.txt", 'w') as file:
    file.write("\n".join(all_summaries))
