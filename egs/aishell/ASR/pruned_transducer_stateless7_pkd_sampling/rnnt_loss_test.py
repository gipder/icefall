import k2
import torch
from torch import Tensor
from typing import Optional, Tuple, Union

def main():
# Starting
    B=1
    T=10
    U=8
    K=5
    s_range=3
    logits = torch.rand(B, T, s_range, K)
    # when logits are in log space, the function get_rnnt_logprobs_pruned
    # will not affect the result
    logits = torch.softmax(logits, dim=-1).log()
    s_range = torch.tensor([[[0, 1, 2],
                            [0, 1, 2],
                            [1, 2, 3],
                            [1, 2, 3],
                            [2, 3, 4],
                            [3, 4, 5],
                            [4, 5, 6],
                            [4, 5, 6],
                            [5, 6, 7],
                            [5, 6, 7]]], dtype=torch.int64)

    y = [[4, 3, 2, 1, 2, 1, 2, 3]]
    y = k2.RaggedTensor(y)
    y_padded = y.pad(mode="constant", padding_value=0).to(torch.int64)
    row_splits = y.shape.row_splits(1)
    y_lens = row_splits[1:] - row_splits[:-1]
    boundary = torch.zeros((logits.shape[0], 4), dtype=torch.int64)
    boundary[:, 2] = y_lens
    boundary[:, 3] = T
    print(f"{logits.shape=}")
    print(f"{s_range.shape=}")
    print(f"{y.shape=}")
    print(f"{boundary=}")
    print(f"{y_padded=}")
    px, py = my_get_rnnt_logprobs_pruned(
            logits=logits,
            symbols=y_padded,
            ranges=s_range,
            termination_symbol=0,
            boundary=boundary,
        )
    # remove the last frame of px
    px = px[:, :, :-1]
    # remove the last symbol of py
    py = py[:, :-1, :]
    print(f"{px.shape=}")
    print(f"{py.shape=}")
    print(f"{logits[0, 0:2]=}")
    print(f"{px=}")
    print(f"{py=}")

    # remove inf in px and py
    # px = px[px != float("-inf")]

    # instead of above, we can use the following code
    # we will use s_range and gather the corresponding px and py
    # then we will remove all -inf in px and py
    px = px.permute(0, 2, 1)
    py = py.permute(0, 2, 1)
    px = torch.gather(px, 2, s_range).exp()
    py = torch.gather(py, 2, s_range).exp()
    print(f"{px=}")
    print(f"{py=}")


def fix_for_boundary(px: Tensor, boundary: Optional[Tensor] = None) -> Tensor:
    if boundary is None:
        return px
    B, S, T1 = px.shape
    boundary = boundary[:, 3].reshape(B, 1, 1).expand(B, S, T1)
    return px.scatter_(dim=2, index=boundary, value=float("-inf"))

def my_get_rnnt_logprobs_pruned(
    logits: Tensor,
    symbols: Tensor,
    ranges: Tensor,
    termination_symbol: int,
    boundary: Tensor,
    rnnt_type: str = "regular",
) -> Tuple[Tensor, Tensor]:
    """Construct px, py for mutual_information_recursion with pruned output.

    Args:
      logits:
        The pruned output of joiner network, with shape (B, T, s_range, C)
      symbols:
        The symbol sequences, a LongTensor of shape [B][S], and elements in
        {0..C-1}.
      ranges:
        A tensor containing the symbol ids for each frame that we want to keep.
        It is a LongTensor of shape ``[B][T][s_range]``, where ``ranges[b,t,0]``
        contains the begin symbol ``0 <= s <= S - s_range + 1``, such that
        ``logits[b,t,:,:]`` represents the logits with positions
        ``s, s + 1, ... s + s_range - 1``.
        See docs in :func:`get_rnnt_prune_ranges` for more details of what
        ranges contains.
      termination_symbol:
        the termination symbol, with 0 <= termination_symbol < C
      boundary:
        a optional LongTensor of shape [B, 4] with elements interpreted as
        [begin_symbol, begin_frame, end_symbol, end_frame] that is treated as
        [0, 0, S, T]
        if boundary is not supplied.
        Most likely you will want begin_symbol and begin_frame to be zero.
      rnnt_type:
        Specifies the type of rnnt paths: `regular`, `modified` or `constrained`.
        `regular`: The regular rnnt that taking you to the next frame only if
                   emitting a blank (i.e., emitting a symbol does not take you
                   to the next frame).
        `modified`: A modified version of rnnt that will take you to the next
                    frame whether emitting a blank or a non-blank symbol.
        `constrained`: A version likes the modified one that will go to the next
                       frame when you emit a non-blank symbol, but this is done
                       by "forcing" you to take the blank transition from the
                       *next* context on the *current* frame, e.g. if we emit
                       c given "a b" context, we are forced to emit "blank"
                       given "b c" context on the current frame.
    Returns:
      (px, py) (the names are quite arbitrary)::

          px: logprobs, of shape [B][S][T+1] if rnnt_type is regular,
                                 [B][S][T] if rnnt_type is not regular.
          py: logprobs, of shape [B][S+1][T]

      in the recursion::

         p[b,0,0] = 0.0
         if rnnt_type == "regular":
            p[b,s,t] = log_add(p[b,s-1,t] + px[b,s-1,t],
                               p[b,s,t-1] + py[b,s,t-1])
         if rnnt_type != "regular":
            p[b,s,t] = log_add(p[b,s-1,t-1] + px[b,s-1,t-1],
                               p[b,s,t-1] + py[b,s,t-1])

      .. where p[b][s][t] is the "joint score" of the pair of subsequences of
      length s and t respectively.  px[b][s][t] represents the probability of
      extending the subsequences of length (s,t) by one in the s direction,
      given the particular symbol, and py[b][s][t] represents the probability
      of extending the subsequences of length (s,t) by one in the t direction,
      i.e. of emitting the termination/next-frame symbol.

      if `rnnt_type == "regular"`, px[:,:,T] equals -infinity, meaning on the
      "one-past-the-last" frame we cannot emit any symbols.
      This is simply a way of incorporating
      the probability of the termination symbol on the last frame.
    """
    # logits (B, T, s_range, C)
    # symbols (B, S)
    # ranges (B, T, s_range)
    assert logits.ndim == 4, logits.shape
    (B, T, s_range, C) = logits.shape
    assert ranges.shape == (B, T, s_range), (ranges.shape, B, T, s_range)
    (B, S) = symbols.shape
    assert S >= 0, S
    assert (
        rnnt_type != "modified" or T >= S
    ), f"Modified transducer requires T >= S, but got T={T} and S={S}"
    assert rnnt_type in ["regular", "modified", "constrained"], rnnt_type

    normalizers = torch.logsumexp(logits, dim=3)
    symbols_with_terminal = torch.cat(
        (
            symbols,
            torch.tensor(
                [termination_symbol] * B,
                dtype=torch.int64,
                device=symbols.device,
            ).reshape((B, 1)),
        ),
        dim=1,
    )

    # (B, T, s_range)
    pruned_symbols = torch.gather(
        symbols_with_terminal.unsqueeze(1).expand((B, T, S + 1)),
        dim=2,
        index=ranges,
    )

    # (B, T, s_range)
    px = torch.gather(
        logits, dim=3, index=pruned_symbols.reshape(B, T, s_range, 1)
    ).squeeze(-1)
    px = px - normalizers

    # (B, T, S) with index larger than s_range in dim 2 fill with -inf
    px = torch.cat(
        (
            px,
            torch.full(
                (B, T, S + 1 - s_range),
                float("-inf"),
                device=px.device,
                dtype=px.dtype,
            ),
        ),
        dim=2,
    )

    # (B, T, S) with index out of s_range in dim 2 fill with -inf
    px = k2._roll_by_shifts(px, ranges[:, :, 0])[:, :, :S]

    px = px.permute((0, 2, 1))

    if rnnt_type == "regular":
        px = torch.cat(
            (
                px,
                torch.full(
                    (B, S, 1), float("-inf"), device=px.device, dtype=px.dtype
                ),
            ),
            dim=2,
        )  # now: [B][S][T+1], index [:,:,T] has -inf..

    py = logits[:, :, :, termination_symbol].clone()  # (B, T, s_range)
    py = py - normalizers

    # (B, T, S + 1) with index larger than s_range in dim 2 filled with -inf
    py = torch.cat(
        (
            py,
            torch.full(
                (B, T, S + 1 - s_range),
                float("-inf"),
                device=py.device,
                dtype=py.dtype,
            ),
        ),
        dim=2,
    )

    # (B, T, S + 1) with index out of s_range in dim 2 fill with -inf
    py = k2._roll_by_shifts(py, ranges[:, :, 0])
    # (B, S + 1, T)
    py = py.permute((0, 2, 1))
    #print(f"{px.shape=}")
    #print(f"{py.shape=}")
    #print(f"{logits.shape=}")
    #print(f"{normalizers.shape=}")
    #print(f"{logits=}")
    #tt = logits - torch.logsumexp(logits, dim=-1).unsqueeze(-1)
    #print(f"{px[0, 0]=}")
    #print(f"{py[0, 0]=}")
    #print(f"{tt[0, 0, 0]=}")
    #print(f"{tt[0, 0, 0, 0]=}")
    #print(f"{tt[0, 0, 0, 427]=}")
    ## this logsumexp is the same with the normalizers
    ##print(f"{torch.logsumexp(logits, dim=-1)=}")
    #print(f"{normalizers=}")

    if rnnt_type == "regular":
        px = fix_for_boundary(px, boundary)
    elif rnnt_type == "constrained":
        px += py[:, 1:, :]

    return (px, py)


main()
