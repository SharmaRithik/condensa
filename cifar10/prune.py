import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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
        self.fc1 = nn.Linear(32*32*3, 512)  # 32*32*3 = 3072 is the input size
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)  # 10 is the number of classes

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten the input
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

# ... [your previous imports here] ...

import condensa
from condensa.schemes import Compose, Prune
from condensa.opt import LC
from condensa.util import cnn_statistics

# ... [your dataset loading and network architecture here] ...

# Save the original model first
torch.save(model.state_dict(), 'original_model.pth')

# For Condensa, define the compression scheme
scheme = Prune(0.98)  # Prune 98% of the weights (as per earlier example, but this is aggressive)

# Convert the training data and labels to tensors
train_images = torch.tensor(train_data.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0  # Normalizing the images
train_labels = torch.tensor(train_data.targets, dtype=torch.long)

# Convert the test data and labels to tensors
test_images = torch.tensor(test_data.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
test_labels = torch.tensor(test_data.targets, dtype=torch.long)

# Create TensorDatasets
train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)

# Then, you can create your DataLoaders like before
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)


# Initialize the LC optimizer for Condensa
lc = LC(steps=35,
        l_optimizer=condensa.opt.lc.Adam,  # Note: we're using Adam optimizer which you used for training
        lr=0.001,
        mb_iterations_per_l=3000,
        mb_iterations_first_l=30000,
        mu_init=1e-3,
        mu_multiplier=1.1,
        mu_cap=10000,
        debugging_flags={'custom_model_statistics': cnn_statistics})

# Setup the Compressor
compressor = condensa.Compressor(lc,
                                 scheme,
                                 model,
                                 trainloader,
                                 testloader,
                                 valloader=None,  # We don't have a validation loader as of now
                                 criterion=criterion)

# Run the compression
compressed_model = compressor.run()

# Save the compressed model
torch.save(compressed_model.state_dict(), 'compressed_model.pth')

# Test with the compressed model (same as before)
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        output = compressed_model(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Test Accuracy with Compressed Model: %d %%' % (100 * correct / total))

