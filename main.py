import torch
from models.model_builder import ExtSummarizer
from ext_sum import summarize

# Load model
checkpoint = torch.load(f'gru_model.pt', map_location='cpu')
model = ExtSummarizer(device='cpu')

# Run summarization
input_fp = "D0601_article_sentences.pkl"
result_fp = 'results/summary_1.txt'
summary = summarize(input_fp, result_fp, model, max_length=3)
print(summary)