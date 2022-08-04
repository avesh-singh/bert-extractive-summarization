import torch.nn as nn
from torch.optim import Adam
from models.model_builder import ExtSummarizer
import torch
from cnn.cnn import cnn_training_batch
import glob
import time

checkpoints_dir = "checkpoints/gru-bert"

loss_fn = nn.BCEWithLogitsLoss()
batch_size = 1
learning_rate = 0.002
epochs = 3
checkpoint_epoch = 0
model = ExtSummarizer(device='cpu', gru_layers=2)
optimizer = Adam(model.parameters(), lr=learning_rate)

checkpoints = sorted(glob.glob(f"{checkpoints_dir}/model_checkpoint*"))
print(checkpoints)
checkpoint_file = None
if checkpoints:
    checkpoint_file = checkpoints[-1]

if checkpoint_file:
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    checkpoint_epoch = checkpoint['epoch']


if checkpoint_epoch:
    epochs = epochs - checkpoint_epoch
batch = []
cnn_topics = ["input1.rouge"]
for epoch in range(epochs):
    print(f"epoch: {epoch + 1}")
    for j in range(2):
        if j == 0:
            batch = cnn_training_batch("input1.rouge", items=slice(0, 2500, None))
        elif j == 1:
            batch = cnn_training_batch("input1.rouge", items=slice(2500, None, None))
        print(f"batch: {j}")
        for i, example in enumerate(batch):
            src, mask, segs, clss, mask_cls, src_str = example[0]
            sent_scores, mask = model(src, segs, clss, mask, mask_cls)
            sent_scores = sent_scores + mask.float()
            sent_scores = sent_scores.cpu()
            optimizer.zero_grad()
            loss = loss_fn(sent_scores[:, :len(example[1])], torch.tensor(example[1][:sent_scores.shape[1]],
                           dtype=torch.float).unsqueeze(0))
            loss.backward()
            optimizer.step()

            loss = loss.item()
            print(f"loss: {loss:>7f}  [{i:>5d}/{len(batch):>5d}]")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
        }, f"checkpoints/gru-bert/model_checkpoints_{int(time.time())}.tar")
