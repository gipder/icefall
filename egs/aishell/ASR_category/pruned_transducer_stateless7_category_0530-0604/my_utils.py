import torch
import k2


def map_to_table(y, N, num_vocabs=4336, blank_id=0, unk_id=2 ):
    """
    y: k2.RaggedTensor, values in range 0~4335 # list
    N: total number of mapping buckets (including 0)
    blank_id
    Returns: mapped LongTensor of shape (B, U)
    """
    mapped = list()
    for x in y.tolist():
        x = torch.tensor(x, dtype=torch.long)
        x_mapped = torch.zeros_like(x)

        # Case 1: 0~2 → 0
        mask_zero = (x <= unk_id)
        x_mapped = x_mapped.masked_fill(mask_zero, 0)

        # Case 2: 3~4335 → scaled to 1 ~ N-1
        above_unk = unk_id + 1
        mask_scale = (x >= above_unk)
        x_clipped = torch.clamp(x, min=above_unk, max=num_vocabs-1)
        x_scaled = (x_clipped - above_unk).float() / (num_vocabs-1-above_unk) * (N - unk_id) + 1
        x_scaled = x_scaled.round().long()
        x_mapped = x_mapped.masked_scatter(mask_scale, x_scaled[mask_scale])
        mapped.append(x_mapped.tolist())

    return mapped
