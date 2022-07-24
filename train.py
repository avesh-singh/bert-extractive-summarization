from ext_sum import load_text
from prepare_data import training_batch
import torch.nn as nn
from torch.optim import Adam
from models.model_builder import ExtSummarizer
import torch
from cnn import cnn_training_batch

topics = [
    "D0601A",
    "D0603C",
    "D0605E",
    "D0607G",
    "D0609I",
    "D0611B",
    "D0613D",
    "D0615F",
    "D0617H",
    "D0619A",
    "D0621C",
    "D0623E",
    "D0625G",
    "D0627I",
    "D0629B",
    "D0631D",
    "D0633F",
    "D0635H",
    "D0637A",
    "D0639C",
    "D0641E",
    "D0643G",
    "D0645I",
    # --------
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

loss_fn = nn.BCEWithLogitsLoss()
batch_size = 1
learning_rate = 0.002
epochs = 2
model = ExtSummarizer(device='cpu')
optimizer = Adam(model.parameters(), lr=learning_rate)
batches = []
cnn_topics = ["train_10000.csvaa"]
for topic in cnn_topics:
    batch = cnn_training_batch(topic, items=5)
    size = len(batch)
    batches += [[size, batch]]
for epoch in range(epochs):
    print(f"epoch: {epoch + 1}")
    for j, [size, batch] in enumerate(batches):
        print(f"batch: {j}")
        for i, example in enumerate(batch):
            src, mask, segs, clss, mask_cls, src_str = example[0]
            sent_scores, mask = model(src, segs, clss, mask, mask_cls)
            sent_scores = sent_scores + mask.float()
            sent_scores = sent_scores.cpu()
            optimizer.zero_grad()
            loss = loss_fn(sent_scores, torch.tensor(example[1][:sent_scores.shape[1]], dtype=torch.float).unsqueeze(0))
            loss.backward()
            optimizer.step()

            loss = loss.item()
            print(f"loss: {loss:>7f}  [{i:>5d}/{size:>5d}]")

torch.save(model.state_dict(), 'gru_model.pt')
