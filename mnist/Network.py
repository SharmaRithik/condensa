# Network.py
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.l1 = nn.Linear(784, 200)  # assuming input size is 784 and hidden size is 200
        self.relu = nn.ReLU()
        self.l3 = nn.Linear(200, 10)  # assuming hidden size is 200 and output size is 10

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l3(x)
        return F.log_softmax(x, dim=1)

if __name__ == "__main__":
    net = Network()
    print(net)

