import torch

# Example tensor with shape [2, 10, 5, 4]
batch_size, time, label, num_classes = 2, 10, 5, 4
logits = torch.randn(batch_size, time, label, num_classes)  # Simulated tensor

# Define the diagonal indices for `time` and `label`
time_indices = torch.arange(min(time, label))
label_indices = torch.arange(min(time, label))

# Example class indices for each batch (2 batches)
class_indices_batch1 = torch.tensor([1, 2, 3, 3, 0])  # Batch 1 diagonal class indices
class_indices_batch2 = torch.tensor([0, 1, 2, 3, 1])  # Batch 2 diagonal class indices

# Combine the class indices for both batches
class_indices = torch.stack([class_indices_batch1, class_indices_batch2])  # Shape: [2, 5]

# Extract logits for diagonal elements for both batches
diagonal_logits = logits[:, time_indices, label_indices]  # Shape: [2, 5, 4]

# Permute logits to (minibatch, C, d1)
diagonal_logits_permuted = diagonal_logits.permute(0, 2, 1)  # Shape: [2, 4, 5]

# Compute Cross-Entropy Loss directly without reshaping
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(diagonal_logits_permuted, class_indices)

# Output intermediate and final results
print("Logits:\n", logits)
print("\nDiagonal logits:\n", diagonal_logits)
print("\nPermuted logits (minibatch, C, d1):\n", diagonal_logits_permuted)
print("\nClass indices:\n", class_indices)
print("\nCross-Entropy Loss:\n", loss.item())

