from ext_sum import load_text
from prepare_data import training_batch
import torch.nn as nn
from torch.optim import SGD
from models.model_builder import ExtSummarizer
import torch

batch = training_batch()
size = len(batch)
loss_fn = nn.CrossEntropyLoss()
batch_size = 1
learning_rate = 0.001
epochs = 3
model = ExtSummarizer(device='cpu')
optimizer = SGD(model.parameters(), lr=learning_rate)
for epoch in range(epochs):
    print(f"epoch: {epoch + 1}")
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