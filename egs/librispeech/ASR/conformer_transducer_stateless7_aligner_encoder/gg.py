import torch
import torch.nn as nn

# 입력 텐서 정의 (N: batch, T: time, F: feature dimension)
N, T, F = 16, 50, 128  # Batch size: 16, Time steps: 50, Feature dim: 128
input_tensor = torch.rand(N, T, F)  # (N, T, F)

# Conv2d 기반 Subsampling 모듈
class Subsampling(nn.Module):
    def __init__(self, in_channels=1, in_features=128, out_features=32, kernel_size=3, stride=2):
        super(Subsampling, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=(1, kernel_size),  # 시간 축(T)에 영향을 주지 않기 위해 (1, kernel_size) 사용
            stride=(1, stride),           # 시간 축(T)에 영향을 주지 않기 위해 (1, stride) 사용
            padding=(0, (kernel_size - 1) // 2)  # Feature 축에서만 padding 적용
        )

    def forward(self, x):
        # (N, T, F) -> (N, 1, T, F) for Conv2d
        x = x.unsqueeze(1)  # Add channel dimension: (N, 1, T, F)
        x = self.conv2d(x)  # Apply Conv2d: (N, 1, T, F')
        x = x.squeeze(1)    # Remove channel dimension: (N, T, F')
        return x

# Subsampling 예제
subsampling = Subsampling(in_features=128, out_features=32, kernel_size=3, stride=2)
output_tensor = subsampling(input_tensor)  # (N, T, F') where F' = 32

# 결과 출력
print("Input shape:", input_tensor.shape)   # (16, 50, 128)
print("Output shape:", output_tensor.shape) # (16, 50, 32)

