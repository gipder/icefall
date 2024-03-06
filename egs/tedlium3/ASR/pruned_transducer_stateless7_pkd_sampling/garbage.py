import torch
import k2

def gg(
    sampling_y: list = None,
    num: int = 1,
):
    for i in range(num):
        print(sampling_y[i])
    return 0

a = k2.RaggedTensor('[[1 2] [3 4 5]]')
b = k2.RaggedTensor('[[3 4 5] [3 4 5]]')
alist = list()
alist.append(a)
alist.append(b)

gg(alist, len(alist))

