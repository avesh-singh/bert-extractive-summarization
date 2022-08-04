from ext_sum import load_text
from prepare_data import training_batch
import torch.nn as nn
from torch.optim import Adam
from models.model_builder import ExtSummarizer
import torch
import glob
import time

checkpoints_dir = "checkpoints/gru-bert-duc"

loss_fn = nn.BCEWithLogitsLoss()
batch_size = 1
learning_rate = 0.002
epochs = 3
checkpoint_epoch = 0
model = ExtSummarizer(device='cpu')
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
batches = []
cnn_topics = ["input1"]
for topic in cnn_topics:
    batch = training_batch(topic)
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
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }, f"{checkpoints_dir}/model_checkpoints_{int(time.time())}.tar")
