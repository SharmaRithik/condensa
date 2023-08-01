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

# Ensure CUDA is available
assert torch.cuda.is_available()

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
MEM = Compose([Prune(0.02), Quantize(condensa.float16)])

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

# ... [previous code]

# Split the dataset into training and validation sets
validation_split = 0.2  # Use 20% of the data for validation
dataset_size = len(x)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.shuffle(indices)  # Make sure to shuffle the data before splitting

train_indices, val_indices = indices[split:], indices[:split]

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

train_dataset = torch.utils.data.TensorDataset(x, y)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
validloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

# ... [previous code]

# Training loop
for e in range(epochs):
    net.train()
    for i, (x_mini, y_mini) in enumerate(trainloader):  # Use DataLoader's batching capability

        optimizer.zero_grad()
        net_out = net(x_mini)

        loss = loss_func(net_out, y_mini)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            loss_log.append(loss.item())

    # Evaluate on validation set
    net.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Turn off gradient computation
        validation_loss = 0
        correct = 0
        for x_val, y_val in validloader:
            outputs = net(x_val)
            validation_loss += F.cross_entropy(outputs, y_val, reduction='sum').item()
            preds = outputs.argmax(dim=1, keepdim=True)
            correct += preds.eq(y_val.view_as(preds)).sum().item()

        validation_loss /= len(validloader.dataset)
        validation_accuracy = 100. * correct / len(validloader.dataset)
        print('Validation Loss: {:.6f}, Validation Accuracy: {:.2f}%'.format(validation_loss, validation_accuracy))

    print('Epoch: {} - Training Loss: {:.6f}'.format(e, loss.item()))
    scheduler.step()

# ... [rest of the code]

