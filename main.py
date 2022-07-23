import torch
from models.model_builder import ExtSummarizer
from ext_sum import summarize
from os import walk
from scrape import read_articles
# Load model
model_type = 'bertbase' #@param ['bertbase', 'distilbert', 'mobilebert']
checkpoint = torch.load(f'checkpoints/{model_type}_ext.pt', map_location='cpu')
model = ExtSummarizer(checkpoint=checkpoint, bert_type=model_type, device='cpu')

# Run summarization
root_directory = "DUCDatasets-20220708T144932Z-001/DUCDatasets/DUC2007_Summarization_Documents/duc2007_testdocs/main"
article_set = "D0701A"
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