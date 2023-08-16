import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from network import FullyConnectedNetwork

# Define a function to test the model accuracy
def test_accuracy(model, test_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load test data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

# Load original model
original_model = FullyConnectedNetwork().to(device)
original_model.load_state_dict(torch.load('original_model_cifar10.pth'))

# Load compressed model
compressed_model = FullyConnectedNetwork().to(device)
compressed_model.load_state_dict(torch.load('compressed_model_cifar10.pth'))

# Test the original model
original_accuracy = test_accuracy(original_model, test_loader, device)
print('Test Accuracy of Original Model: %.2f %%' % original_accuracy)

# Test the compressed model
compressed_accuracy = test_accuracy(compressed_model, test_loader, device)
print('Test Accuracy of Compressed Model: %.2f %%' % compressed_accuracy)

