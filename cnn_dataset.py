from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer
import torch
from ext_sum import load_text
from nltk.tokenize import sent_tokenize

class CNNDataset(Dataset):
    def __init__(self):
        self.train = pd.read_csv("raw_data/cnn_dailymail/train_10000.csvaa.csv")
        self.prepared_x = self.train.apply(lambda x: load_text("[CLS] [SEP]".join(sent_tokenize(x['article'])), 512,
                                                         "cpu"), 1)


    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        row = self.train.iloc[idx]
        return torch.tensor(row[1]), torch.tensor(row[2])


from torch.utils.data import DataLoader

train_dataloader = DataLoader(CNNDataset(), batch_size=64, shuffle=True)
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
