import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# Hyperparameters and other constants
input_size = 784
output_size = 10
hidden_size = 200
epochs = 20
batch_size = 50
learning_rate = 0.00005


# Define the Network class
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

# Load the compressed model
model_compressed = Network()
model_compressed.load_state_dict(torch.load('compressed_model.pth'))
model_compressed.eval()

# Extract weights from the compressed model
weight1 = model_compressed.l1.weight.data.cpu().numpy()
print("Shape of Weight1 =", weight1.shape)
weight3 = model_compressed.l3.weight.data.cpu().numpy()
print("Shape of Weight2 =", weight3.shape)

# Convert weight matrices to CSR format
weight1_csr = sp.csr_matrix(weight1)
weight3_csr = sp.csr_matrix(weight3)

# Convert weight matrices to BSR format
weight1_bsr = sp.bsr_matrix(weight1)
weight3_bsr = sp.bsr_matrix(weight3)
# Load data from CSV
data = pd.read_csv('data/train.csv', sep=",")
dataFinal = data.drop('label', axis=1)
x = torch.FloatTensor(dataFinal.to_numpy()) / 255.0
sample_input_tensor = x[0]  # This gets the first image/data point from your dataset.
input_vector = sample_input_tensor.cpu().numpy()

# For multiplication using CSR
result1_csr = weight1_csr.dot(input_vector)
# Apply the ReLU activation
result1_csr_activated = np.maximum(0, result1_csr)
result3_csr = weight3_csr.dot(result1_csr_activated)

# Similarly, for multiplication using BSR
result1_bsr = weight1_bsr.dot(input_vector)
# Apply the ReLU activation
result1_bsr_activated = np.maximum(0, result1_bsr)
result3_bsr = weight3_bsr.dot(result1_bsr_activated)
