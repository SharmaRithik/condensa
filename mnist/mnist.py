import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import condensa
import util
from condensa.schemes import Compose, Prune, Quantize
import torchvision.datasets as datasets
import logging

assert torch.cuda.is_available()

print("Reading the data...")
data = pd.read_csv('data/train.csv', sep=",")
test_data = pd.read_csv('data/test.csv', sep=",")

print("Reshaping the data...")
dataFinal = data.drop('label', axis=1)
labels = data['label']

dataNp = dataFinal.to_numpy()
labelsNp = labels.to_numpy()
test_dataNp = test_data.to_numpy()
dataNpFlattened = dataNp.reshape(dataNp.shape[0], -1)

print("Data is ready")

x = torch.FloatTensor(dataNp.tolist())
y = torch.LongTensor(labelsNp.tolist())

# Hyperparameters
input_size = 784
output_size = 10
hidden_size = 200

epochs = 20
batch_size = 50
learning_rate = 0.00005

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l3(x)
        return F.log_softmax(x, dim=1)

net = Network()
print(net)

optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

loss_log = []

for e in range(epochs):
    for i in range(0, x.shape[0], batch_size):
        x_mini = x[i:i + batch_size]
        y_mini = y[i:i + batch_size]

        optimizer.zero_grad()
        net_out = net(x_mini)

        loss = loss_func(net_out, y_mini)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            loss_log.append(loss.item())

    print('Epoch: {} - Loss: {:.6f}'.format(e, loss.item()))

plt.figure(figsize=(10, 8))
plt.plot(loss_log)

test = torch.FloatTensor(test_dataNp.tolist())
net_out = net(test)

predictions = torch.max(net_out.data, 1)[1].numpy()

plt.figure(figsize=(14, 12))
output = np.column_stack((np.arange(1, predictions.size + 1), predictions))
np.savetxt("out.csv", output, fmt='%d', delimiter=',', header='ImageId,Label')
