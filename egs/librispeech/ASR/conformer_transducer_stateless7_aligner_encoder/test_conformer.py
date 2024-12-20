import torch
from conformer import Conformer

def test_conformer():
    num_features = 40
    num_classes = 87
    d_model = 256
    model = Conformer(num_features=num_features, num_classes=num_classes, d_model=d_model)
    N = 31

    for T in range(17, 18):
        x = torch.rand(N, T, num_features)
        x_lens = torch.full((N,), T, dtype=torch.int32)
        #print(f"{x=}")
        #my, _ = model.run_encoder(x)
        encoded_x, encoded_x_lens = model(x, x_lens)
        print(f"{encoded_x.shape=}")
        print(f"{encoded_x_lens=}")
        assert encoded_x.shape == (N, (((T - 1) // 2) - 1) // 2, d_model)

def main():
    test_conformer()

if __name__ == "__main__":
    main()
