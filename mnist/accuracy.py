# accuracy.py
import numpy as np
import torch
from sklearn.model_selection import train_test_split

def load_data():
    # Load data from CSV
    print("Reading the data...")
    data = pd.read_csv('data/train.csv', sep=",")
    
    # Preprocess and reshape data
    print("Reshaping the data...")
    dataFinal = data.drop('label', axis=1)
    labels = data['label']

    x = torch.FloatTensor(dataFinal.to_numpy()) / 255.0
    y = torch.LongTensor(labels.to_numpy())

    return x, y

def split_data(x, y):
    # Split data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_val, y_train, y_val

# check this accuracy function
# manually checks for images, vector of probab
# Check if loading the right data
# differnt data maybe
# 
def accuracy(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in data_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

if __name__ == "__main__":
    import pandas as pd
    from Network import Network  # Assuming the neural network class is saved in a separate file called Network.py

    x, y = load_data()
    x_train, x_val, y_train, y_val = split_data(x, y)

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Load original model and check its accuracy
    net = Network()
    net.load_state_dict(torch.load('compressed_model.pth'))
    #net.load_state_dict(torch.load('original_model.pth'))
    original_model_accuracy = accuracy(net, valloader)
    print(f"Original Model Accuracy: {original_model_accuracy:.4f}%")

    # Load the compressed model and check its accuracy
    model_compressed = Network()
    model_compressed.load_state_dict(torch.load('original_model.pth'))
    #model_compressed.load_state_dict(torch.load('compressed_model.pth'))
    compressed_model_accuracy = accuracy(model_compressed, valloader)
    print(f"Compressed Model Accuracy: {compressed_model_accuracy:.4f}%")

