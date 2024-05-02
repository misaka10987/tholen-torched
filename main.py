#!/usr/bin/python
import torch.optim
from torch import nn
from torch.utils.data import random_split, DataLoader

from init import init
from loader import Database
from parser import Asteroid, SpecT, MyData

init()

db = Database()

data = MyData([Asteroid.from_row(row) for row in db.fetch()])

train_size = int(len(data) * 0.8)
test_size = len(data) - train_size

model = nn.Sequential(
    nn.Linear(17, 17),
    nn.ReLU(),
    nn.Linear(17, 17),
    nn.ReLU(),
    nn.Linear(17, 17),
    nn.ReLU(),
    nn.Linear(17, 3),
    nn.LogSoftmax(dim=0),
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

global_correct = 0

for i in range(50):
    train, test = random_split(data, [train_size, test_size])

    # training
    for a in train:
        output = model(a.to_tensor())

        loss = loss_fn(output, a.to_label())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    # testing
    with torch.no_grad():
        correct = 0
        freq = {SpecT.X: 0, SpecT.C: 0, SpecT.S: 0}
        for a in test:
            output = torch.argmax(model(a.to_tensor()))
            freq[a.spec_t] += 1
            if output == a.to_label():
                correct += 1
                global_correct += 1
        print(f"[{i}]: {correct}/{len(test)} correct, rate={round(100 * global_correct / ((i + 1) * test_size), 1)}%")
