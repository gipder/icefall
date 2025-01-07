"""
import torch

# Example tensor with shape [1, 10, 5, 4]
batch_size, time, label, num_classes = 1, 10, 5, 4
logits = torch.randn(batch_size, time, label, num_classes)  # Simulated tensor

# Define the diagonal indices for `time` and `label`
# For example, time index [0, 1, 2, 3, 4] aligns with label index [0, 1, 2, 3, 4]
time_indices = torch.arange(min(time, label))
label_indices = torch.arange(min(time, label))
class_indices = torch.tensor([1, 2, 3, 3, 0])  # Example class indices for diagonal

# Use advanced indexing to extract the logits
diagonal_logits = logits[0, time_indices, label_indices, class_indices]

# Compute Cross-Entropy loss
target = torch.tensor(class_indices)  # Targets must be the same length as the diagonal
loss = torch.nn.CrossEntropyLoss()(diagonal_logits.unsqueeze(0), target.unsqueeze(0))

print(f"Diagonal logits: {diagonal_logits}")
print(f"Loss: {loss.item()}")
"""

import torch

# Example tensor with shape [1, 10, 5, 4]
batch_size, time, label, num_classes = 1, 10, 5, 4
logits = torch.randn(batch_size, time, label, num_classes)  # Simulated tensor

# Define the diagonal indices for `time` and `label`
time_indices = torch.arange(min(time, label))
label_indices = torch.arange(min(time, label))
class_indices = torch.tensor([1, 2, 3, 3, 0])  # Example class indices for diagonal

# Use advanced indexing to extract the logits
diagonal_logits = logits[0, time_indices, label_indices]  # Shape: [5, 4]

# Compute Cross-Entropy loss
# diagonal_logits should have shape [batch_size, num_classes], so unsqueeze batch dimension
loss = torch.nn.CrossEntropyLoss()(diagonal_logits, class_indices)

print(f"Logits: {logits}")
print(f"Diagonal logits: {diagonal_logits}")
print(f"Loss: {loss.item()}")

