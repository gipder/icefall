import torch
from zipformer import Zipformer

def test_zipformer():
    num_features = 40
    num_classes = 87
    model = Zipformer(num_features=num_features, output_downsampling_factor=4)

    N = 31

    for T in range(37, 38):
        x = torch.rand(N, T, num_features)
        # length
        x_lens = torch.randint(14, T, (N,))
        print(f"{x_lens=}")
        #print(f"{x=}")
        my, _ = model(x, x_lens)
        #y, y1, y2 = model(x)
        #print(f"{y=}")
        #print(f"{y1=}")
        print(f"{my=}")
        print(f"{x.shape=}")
        #print(f"{y.shape=}")
        #print(f"{y1.shape=}")
        print(f"{my.shape=}")
        assert y.shape == (N, (((T - 1) // 2) - 1) // 2, num_classes)


def main():
    test_zipformer()

if __name__ == "__main__":
    main()
