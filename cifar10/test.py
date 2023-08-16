import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import condensa
from condensa.schemes import Prune

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load training data
train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Load test data
test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

# Define the network architecture
class FullyConnectedNetwork(nn.Module):
    def __init__(self):
        super(FullyConnectedNetwork, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Initialize the network
model = FullyConnectedNetwork().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(20):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Clear the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(images)
        loss = criterion(output, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

# Test the model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Test Accuracy: %d %%' % (100 * correct / total))

# Compression with Condensa
MEM = Prune(0.98)

# Convert data and labels to tensors
train_data_tensor = torch.tensor(train_data.data, dtype=torch.float32).permute(0, 3, 1, 2)  # Change shape to (50000, 3, 32, 32)
train_targets_tensor = torch.tensor(train_data.targets, dtype=torch.int64)  # Convert labels to tensor of shape (50000,)

# Create a TensorDataset
train_dataset = torch.utils.data.TensorDataset(train_data_tensor, train_targets_tensor)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

# Create a TensorDataset for test data
test_data_tensor = torch.tensor(test_data.data, dtype=torch.float32).permute(0, 3, 1, 2)
test_targets_tensor = torch.tensor(test_data.targets, dtype=torch.int64)
test_dataset = torch.utils.data.TensorDataset(test_data_tensor, test_targets_tensor)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)


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
                                     model,
                                     trainloader,
                                     testloader,
                                     valloader=None,
                                     criterion=criterion)

w_MEM = compressor_MEM.run()

# Save compressed model
torch.save(w_MEM.state_dict(), 'compressed_model_cifar10.pth')

