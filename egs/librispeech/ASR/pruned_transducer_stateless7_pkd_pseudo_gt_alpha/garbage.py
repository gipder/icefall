import torch

# 각 batch의 x 길이와 y 길이가 다른 3차원 텐서 생성
batch_size = 3
x_lengths = [4, 3, 5]  # 각 batch의 x 길이
y_lengths = [5, 4, 2]  # 각 batch의 y 길이

# 각 batch에 대한 랜덤한 2차원 텐서 생성
tensor_list = [torch.rand(x_len, y_len) for x_len, y_len in zip(x_lengths, y_lengths)]

# 각 배치의 텐서를 리스트에 저장
tensors = []

# 각 배치에 대해 경계 처리 함수를 적용하고 리스트에 추가
for i in range(batch_size):
    x_len, y_len = tensor_list[i].size()
    tensor = tensor_list[i]

    # 각 batch의 2차원 텐서의 경계 영역을 무한대(inf)로 설정
    tensor[0, :] = float('inf')            # 위쪽 행
    tensor[x_len - 1, :] = float('inf')   # 아래쪽 행
    tensor[:, 0] = float('inf')            # 왼쪽 열
    tensor[:, y_len - 1] = float('inf')   # 오른쪽 열

    tensors.append(tensor)

# 리스트에 저장된 각 배치의 텐서를 하나로 연결하여 3차원 텐서 생성
final_tensor = tensors
#final_tensor = torch.stack(tensors, dim=0)

# 경계가 무한대로 설정된 3차원 텐서 출력
print("경계가 무한대로 설정된 3차원 텐서:")
print(final_tensor)

