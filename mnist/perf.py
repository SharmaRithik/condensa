import os

original_size = os.path.getsize('original_model.pth')
compressed_size = os.path.getsize('compressed_model.pth')
reduction_percentage = ((original_size - compressed_size) / original_size) * 100
print(f"Model size reduction: {reduction_percentage:.2f}%")

