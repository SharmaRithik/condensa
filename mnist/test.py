import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import condensa
from condensa.schemes import Compose, Prune, Quantize

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load data from CSV
print("Reading the data...")
data = pd.read_csv('data/train.csv', sep=",")
test_data = pd.read_csv('data/test.csv', sep=",")

# Preprocess and reshape data
print("Reshaping the data...")
dataFinal = data.drop('label', axis=1)
labels = data['label']

x = torch.FloatTensor(dataFinal.to_numpy()) / 255.0
y = torch.LongTensor(labels.to_numpy())
test = torch.FloatTensor(test_data.to_numpy()) / 255.0
print("Data is ready")

# Hyperparameters
input_size = 784
output_size = 10
hidden_size = 200
epochs = 20
batch_size = 50
learning_rate = 0.00005

# Neural Network Definition
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
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.85)
loss_func = nn.CrossEntropyLoss()

loss_log = []

# Training loop
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
    scheduler.step()

torch.save(net.state_dict(), 'original_model.pth')

# Predict on test data
net_out = net(test)
predictions = torch.max(net_out.data, 1)[1].numpy()

# Save predictions
output = np.column_stack((np.arange(1, predictions.size + 1), predictions))
np.savetxt("out.csv", output, fmt='%d', delimiter=',', header='ImageId,Label')

# Model Compression with Condensa
# remove quantize ~ maybe not now
# test prune values ~ 90ish atleast ~ experiments. loop~

#MEM = Compose([Prune(0.02), Quantize(condensa.float16)])
MEM = Prune(0.98)

train_dataset = torch.utils.data.TensorDataset(x, y)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)

# Compression setup
lc = condensa.opt.LC(steps=35,
                     l_optimizer=condensa.opt.lc.SGD,
                     l_optimizer_params={'momentum': 0.95},
                     lr=0.01,
                     lr_end=1e-4,
                     mb_iterations_per_l=3000,
                     mb_iterations_first_l=30000,
                     mu_init=1e-3,
                     mu_multiplier=1.1,
                     mu_cap=10000,
                     debugging_flags={'custom_model_statistics': condensa.util.cnn_statistics})

compressor_MEM = condensa.Compressor(lc,
                                     MEM,
                                     net,
                                     trainloader,
                                     testloader,
                                     valloader=None,
                                     criterion=loss_func)

w_MEM = compressor_MEM.run()

# Save compressed model
torch.save(w_MEM.state_dict(), 'compressed_model.pth')

