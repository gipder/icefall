import torch
import torch.nn.functional as F


def garbage():
    print("Garbage")

def find_best_path(logits: torch.Tensor,
                   x_lens: torch.Tensor,
                   y: torch.Tensor,
                   y_lens: torch.Tensor,
                   blank: int = 0):
    """
    Find the best path from the logits
    Args:
      logits: 4D tensor [batch_size, time, label, num_classes]
              The two elements of the last dimension are blank and non-blank
      x_lens: 1D tensor [batch_size]
              The lengths of the input feature sequences
      y: 2D tensor [batch_size, label]
         The label sequences that don't align with the logits
      y_lens: 1D tensor [batch_size]
              The lengths of the label sequences

    Returns:
      mono: 2D tensor [batch_size, time], monotonically aligned label sequences
      alignment: 2D tensor [batch_size, time], alignment according to the mono

    """
    B, T, U, V = logits.shape
    device = logits.device

    # log_probs: [B, T, U, V]
    log_probs = F.log_softmax(logits, dim=-1)

    # prepend the blank label in front of the label sequence
    y_expanded = torch.concat([
        y,
        torch.full((B, 1), blank, dtype=torch.int64, device=device)
    ], dim=-1)
    y_expanded = y_expanded.unsqueeze(1).expand(B, T, U)

    b_log_probs = log_probs[:, :, :, blank].squeeze(-1)
    y_log_probs = torch.gather(log_probs,
                               dim=-1,
                               index=y_expanded.unsqueeze(-1)).squeeze(-1)

    # initialize
    dp = torch.full((B, T, U), float('-inf'), device=device)
    path = torch.full((B, T, U, 2), -1, dtype=torch.int64, device=device)

    dp[:, 0, 0] = b_log_probs[:, 0, 0]

    assert b_log_probs.shape == y_log_probs.shape
    assert b_log_probs.shape == (B, T, U)

    # T-axis
    for t in range(1, T):
        dp[:, t, 0] = dp[:, t-1, 0] + b_log_probs[:, t-1, 0]
        path[:, t, 0] = torch.stack([torch.full((B,), t-1, dtype=torch.int64),
                                     torch.full((B,), 0, dtype=torch.int64)], dim=-1)

    # U-axis
    for u in range(1, U):
        dp[:, u, u] = dp[:, u-1, u-1] + y_log_probs[:, u-1, u-1]
        path[:, u, u] = torch.stack([torch.full((B,), u-1, dtype=torch.int64),
                                     torch.full((B,), u-1, dtype=torch.int64)], dim=-1)

    # DP
    for t in range(1, T):
        for u in range(1, U):
            if t <= u:
                continue

            candidates = torch.stack([
                dp[:, t-1, u] + b_log_probs[:, t-1, u],
                dp[:, t-1, u-1] + y_log_probs[:, t-1, u-1]
            ], dim=-1)

            dp[:, t, u], max_idx = candidates.max(dim=-1)
            prev_t = torch.full((B,), t-1, dtype=torch.int64, device=device)
            prev_u = torch.where(
                max_idx == torch.full((B,), 0, dtype=torch.int64, device=device),
                torch.tensor(u, dtype=torch.int64, device=device),
                torch.tensor(u-1, dtype=torch.int64, device=device)
            )
            path[:, t, u] = torch.stack([prev_t, prev_u], dim=-1)

    # backtrace
    mono = torch.full((B, T), blank, dtype=torch.int64, device=device)
    best_paths = []
    for b in range(B):
        hop = T - x_lens[b].item()
        t, u = x_lens[b].item()-1, y_lens[b].item()-1
        best_path = [(t, u)]
        mono[b, t] = u
        while (t, u) != (0, 0):
            t, u = path[b, t, u].tolist()
            mono[b, t] = u
            best_path.append((t, u))
        best_paths.append(best_path[::-1])

    # transform the best paths to tensor
    mask = mono[:, 1:] == mono[:, :-1]
    mono_index = torch.cumsum(~mask, dim=1)
    mono_index = torch.where(~mask, mono_index, 0)
    mono_index = torch.concat(
        [mono_index, torch.zeros((B, 1), dtype=torch.int64, device=device)],
        dim=-1
    )

    index = mono_index - 1
    mask = mono_index != 0
    alignment = torch.zeros_like(mono_index)

    row_indices = torch.arange(B).unsqueeze(-1).expand(B, T)
    valid_indices = index[mask]
    valid_row_indices = row_indices[mask]

    alignment[mask] = y[valid_row_indices, valid_indices]

    # masking
    mask = torch.arange(T, device=device).unsqueeze(0) >= x_lens.unsqueeze(-1)
    mono[mask] = blank
    alignment[mask] = -100

    return mono, alignment

def find_best_path_optimized(logits: torch.Tensor,
                             x_lens: torch.Tensor,
                             y: torch.Tensor,
                             y_lens: torch.Tensor,
                             blank: int = 0):
    """
    Find the best path from the logits
    Args:
      logits: 4D tensor [batch_size, time, label, num_classes]
              The two elements of the last dimension are blank and non-blank
      x_lens: 1D tensor [batch_size]
              The lengths of the input feature sequences
      y: 2D tensor [batch_size, label]
         The label sequences that don't align with the logits
      y_lens: 1D tensor [batch_size]
              The lengths of the label sequences

    Returns:
      mono: 2D tensor [batch_size, time], monotonically aligned label sequences
      alignment: 2D tensor [batch_size, time], alignment according to the mono

    """
    B, T, U, V = logits.shape
    device = logits.device

    # log_probs: [B, T, U, V]
    log_probs = F.log_softmax(logits, dim=-1)

    # prepend the blank label in front of the label sequence
    y_expanded = torch.concat([
        y,
        torch.full((B, 1), blank, dtype=torch.int64, device=device)
    ], dim=-1)
    y_expanded = y_expanded.unsqueeze(1).expand(B, T, U)

    b_log_probs = log_probs[:, :, :, blank].squeeze(-1)
    y_log_probs = torch.gather(log_probs,
                               dim=-1,
                               index=y_expanded.unsqueeze(-1)).squeeze(-1)

    # initialize
    dp = torch.full((B, T, U), float('-inf'), device=device)
    path = torch.full((B, T, U, 2), -1, dtype=torch.int64, device=device)

    dp[:, 0, 0] = b_log_probs[:, 0, 0]

    # T-axis
    dp[:, 1:, 0] = dp[:, :1, 0] + torch.cumsum(b_log_probs[:, :-1, 0], dim=1)
    t_indices = torch.arange(0, T-1, device=dp.device)
    t_indices_expanded = t_indices.unsqueeze(0).expand(B, -1)
    zeros = torch.zeros_like(t_indices_expanded, dtype=torch.int64)
    path[:, 1:, 0] = torch.stack([t_indices_expanded, zeros], dim=-1)

    assert b_log_probs.shape == y_log_probs.shape
    assert b_log_probs.shape == (B, T, U)

    # U-axis
    diag_y = y_log_probs.diagonal(dim1=1, dim2=2)
    cumsum_diag = torch.cat([
        torch.zeros(B, 1, device=dp.device, dtype=diag_y.dtype),
        torch.cumsum(diag_y[:, :-1], dim=1)
    ], dim=1)  # shape: (B, U)

    new_diag = dp[:, 0, 0].unsqueeze(1) + cumsum_diag  # shape: (B, U)
    dp.diagonal(dim1=1, dim2=2).copy_(new_diag)
    if U > 1:
        # u = 1 ~ U-1 인덱스 생성 (shape: (U-1,))
        u_indices = torch.arange(1, U, device=path.device)
        # 각 u에 대해 (u-1, u-1) 값을 만듭니다. (shape: (U-1, 2))
        new_path_val = torch.stack([u_indices - 1, u_indices - 1], dim=-1)
        # 배치 차원 확장: (B, U-1, 2)
        new_path_val = new_path_val.unsqueeze(0).expand(B, -1, -1)

        # 각 배치별, u 인덱스에 해당하는 대각선 위치에 값을 할당합니다.
        batch_idx = torch.arange(B, device=path.device).unsqueeze(1)  # shape: (B, 1)
        u_idx = u_indices.unsqueeze(0).expand(B, -1)  # shape: (B, U-1)

        path[batch_idx, u_idx, u_idx] = new_path_val

    for t in range(1, T):
        # t에서 유효한 u 인덱스: u = 1, 2, ..., min(t, U)-1
        # (t가 U보다 작으면 u는 1 ~ t-1, t가 U 이상이면 u는 1 ~ U-1)
        u_max = min(t, U)
        if u_max <= 1:  # 계산할 u가 없다면 continue
            continue

        # u 인덱스를 벡터화 (shape: (u_max-1,))
        u_idx = torch.arange(1, u_max, device=dp.device)

        # 두 후보 값 계산 (각각 shape: (B, u_max-1))
        cand0 = dp[:, t-1, u_idx] + b_log_probs[:, t-1, u_idx]
        cand1 = dp[:, t-1, u_idx - 1] + y_log_probs[:, t-1, u_idx - 1]

        # 후보를 쌓아서 (B, u_max-1, 2) shape로 만듭니다.
        candidate = torch.stack([cand0, cand1], dim=-1)

        # 두 후보 중 최댓값과 그에 해당하는 인덱스를 선택합니다.
        dp[:, t, u_idx], best_idx = candidate.max(dim=-1)

        # 이전 t는 항상 t-1, 그리고 이전 u는 best_idx에 따라 u_idx 또는 u_idx-1을 선택합니다.
        prev_t = torch.full((B, candidate.shape[1]), t-1, dtype=torch.int64, device=dp.device)
        prev_u = torch.where(
            best_idx == 0,
            u_idx.unsqueeze(0).expand(B, -1),
            (u_idx - 1).unsqueeze(0).expand(B, -1)

        )

        # 계산된 prev_t와 prev_u를 스택해서 path에 할당합니다.
        path[:, t, u_idx] = torch.stack([prev_t, prev_u], dim=-1)

    # backtrace
    mono = torch.full((B, T), blank, dtype=torch.int64, device=device)
    best_paths = []
    for b in range(B):
        hop = T - x_lens[b].item()
        t, u = x_lens[b].item()-1, y_lens[b].item()-1
        best_path = [(t, u)]
        mono[b, t] = u
        while (t, u) != (0, 0):
            t, u = path[b, t, u].tolist()
            mono[b, t] = u
            best_path.append((t, u))
        best_paths.append(best_path[::-1])

    # transform the best paths to tensor
    mask = mono[:, 1:] == mono[:, :-1]
    mono_index = torch.cumsum(~mask, dim=1)
    mono_index = torch.where(~mask, mono_index, 0)
    mono_index = torch.concat(
        [mono_index, torch.zeros((B, 1), dtype=torch.int64, device=device)],
        dim=-1
    )

    index = mono_index - 1
    mask = mono_index != 0
    alignment = torch.zeros_like(mono_index)

    row_indices = torch.arange(B).unsqueeze(-1).expand(B, T)
    valid_indices = index[mask]
    valid_row_indices = row_indices[mask]

    alignment[mask] = y[valid_row_indices, valid_indices]

    # masking
    mask = torch.arange(T, device=device).unsqueeze(0) >= x_lens.unsqueeze(-1)
    mono[mask] = blank
    alignment[mask] = -100

    return mono, alignment
