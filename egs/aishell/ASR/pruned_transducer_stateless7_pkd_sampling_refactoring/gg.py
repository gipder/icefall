import torch

# 랜덤한 (1, 10, 5) 텐서 생성
tensor = torch.rand((2, 4, 2, 3))

# 기준값 설정
threshold = 0.5

# dim=2의 첫 번째 요소값을 기준으로 특정값보다 작은 경우 dim=1 축소하여 새로운 텐서 생성
dim2_values = tensor[:, :, :, 0]  # dim=2의 첫 번째 요소 선택
mask = dim2_values < threshold  # 기준값보다 작은 원소들에 대한 마스크 생성
print(f"{tensor=}")
print(f"{mask=}")
reduced_tensor = torch.masked_select(tensor, mask.unsqueeze(-1)).reshape(tensor.shape[0], -1, tensor.shape[-2], tensor.shape[-1])

print("원본 텐서:\n", tensor)
print("dim=2의 첫 번째 요소값을 기준으로 특정값보다 작은 경우 dim=1을 축소한 텐서:\n", reduced_tensor)

import sys
sys.exit(0)
import numpy as np

# 예시 데이터 생성
batch_size = 2
rows = 10
new_dim = 3
cols = 5
num_values = 8

"""
# numpy
a = np.random.randint(0, num_values, size=(batch_size, rows, cols))
y = np.arange(batch_size * num_values).reshape(batch_size, num_values)

# 최적화된 함수 정의
def optimized_operation(a, y):
    batch_size, rows, cols = a.shape
    idx = a.reshape(batch_size, -1)
    result = y[np.arange(batch_size).reshape(-1, 1), idx]
    result = result.reshape(batch_size, rows, cols)
    return result

# 최적화된 함수를 호출하여 결과 얻기
result = optimized_operation(a, y)
"""
B = 2
T = 5
R = 2
K = 10

a = torch.tensor([[[0, 1], [0, 1], [1, 2], [2, 3], [3, 4]],
                  [[0, 1], [0, 1], [0, 1], [1, 2], [1, 2]]])
#y = torch.arange(batch_size * num_values).view(batch_size, num_values)
y = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]])

# 최적화된 연산 함수 정의
def optimized_operation(a, y):
    idx = a.view(batch_size, -1)
    result = y[torch.arange(batch_size).view(-1, 1), idx]
    result = result.view(batch_size, rows, cols)
    return result

# dim 이 4인 경우,
def optimized_operation2(a, y):
    idx = a.view(B, T, -1)
    result = y[torch.arange(B).view(-1, 1, 1), idx]
    return result

# 최적화된 함수를 호출하여 결과 얻기
result = optimized_operation2(a, y)

print("Original a:")
print(a)
print("\nOriginal y:")
print(y)
print("\nResult:")
print(result)
