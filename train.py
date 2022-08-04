from prepare_data import training_batch
import torch.nn as nn
from torch.optim import Adam
from models.model_builder import ExtSummarizer
import torch
import glob
import time

checkpoints_dir = "checkpoints/gru-bert-duc"
topics = [
    # "D0601A",
    # "D0603C",
    # "D0605E",
    # "D0607G",
    # "D0609I",
    # "D0611B",
    # "D0613D",
    # "D0615F",
    # "D0617H",
    # "D0619A",
    # "D0621C",
    # "D0623E",
    # "D0625G",
    # "D0627I",
    # "D0629B",
    # "D0631D",
    # "D0633F",
    # "D0635H",
    # "D0637A",
    # "D0639C",
    # "D0641E",
    # "D0643G",
    # "D0645I",
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
    "D0650E"
]
loss_fn = nn.BCEWithLogitsLoss()
batch_size = 1
learning_rate = 0.002
epochs = 3
checkpoint_epoch = 0
model = ExtSummarizer(device='cpu', gru_layers=1)
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


def get_all_training_data():
    batches = []
    for topic in topics:
        batch = training_batch(topic)
        size = len(batch)
        if size == 0:
            print("no training examples found")
            break
        batches += [[size, batch]]
    return batches


def train():
    training_data = get_all_training_data()
    if len(training_data) == 0:
        print("training data not available")
        return
    for epoch in range(epochs):
        print(f"epoch: {epoch + 1}")
        for j, [size, batch] in enumerate(training_data):
            print(f"batch: {j}")
            for i, example in enumerate(batch):
                # for each batch, tokenized article and its labels are passed to the model
                src, mask, segs, clss, mask_cls, src_str = example[0]
                sent_scores, mask = model(src, segs, clss, mask, mask_cls)
                sent_scores = sent_scores + mask.float()
                sent_scores = sent_scores.cpu()
                # this is done so that optimizer does not accumulate previous step's values
                optimizer.zero_grad()
                # based on the prediction by the model, loss function calculates the magnitude
                # of error incurred by the model
                loss = loss_fn(sent_scores,
                               torch.tensor(example[1][:sent_scores.shape[1]], dtype=torch.float).unsqueeze(0))
                # backpropagation is done
                loss.backward()
                optimizer.step()

                loss = loss.item()
                print(f"loss: {loss:>7f}  [{i:>5d}/{size:>5d}]")
        # model checkpoint is saved to speed up future training
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
        }, f"{checkpoints_dir}/model_checkpoints_{int(time.time())}.tar")


if __name__ == "__main__":
    train()
