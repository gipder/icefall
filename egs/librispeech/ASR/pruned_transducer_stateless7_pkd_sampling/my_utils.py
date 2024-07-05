import torch
from typing import Any, Dict, Optional, Tuple, Union
import k2
from icefall.utils import (
    AttributeDict,
    setup_logger,
)

from icefall.utils import add_sos

import logging
import numpy as np
import sys
import os
import math

def make_nbest_alignment(batch: dict,
                         hyp_cache: dict,
                         params: AttributeDict,
                         topk: int = 1,
                         device: torch.device = torch.device("cpu"),
                         ) -> Tuple[Union[list, torch.tensor], Union[list, k2.RaggedTensor]]:
    """
    Make n-best alignments from the given batch and hypotheses dictionary.
    The n-best alignments contains N alignments for each utterance in the batch.
    The type of the n-best alignments is a list of Tensor, which is a tensor of Torch.

    Args:
        batch (dict):
            The batch of icefall.
        hyp_cache (dict):
            The serialized dictionary contains N-best beam search alignments
            from pretrained ASR Models. {key: [[1-best alignment], [2-best alignment],
            ..., [N-best alignment]]}.
        params (AttributeDict):
            The parameters of icefall.
        topk (int):
            The number of alignments to return (N).
        device (torch.device):
            The device to run the model.
    Returns:
        nbest_beam_search_alignment (list of torch.Tensor):
            The list of n-best alignments.
        nbest_pseudo_y (list of k2.RaggedTensor):
            The lis t of n-best scores.
    """

    ids = list()
    # get ids from batch
    for i in range(len(batch["supervisions"]["cut"])):
        ids.append(batch["supervisions"]["cut"][i].id)

    feature_lens = batch["supervisions"]["num_frames"]

    assert hyp_cache is not None, "hyp_cache is None"

    assert not (params.use_1best and params.use_nbest)
    if params.use_1best:
        topk = 1
        logging.info("topk is set to 1")

    with torch.no_grad():
        nbest_beam_search_alignment = list()
        nbest_pseudo_y = list()
        is_all_in_cache = True
        for i in range(len(ids)):
            if hyp_cache.get(ids[i]) is None:
                is_all_in_cache = False
                break
        if not is_all_in_cache:
            logging.error("Not all ids are in hyp_cache")
            logging.error(f"{ids=}")
            assert False

        for n in range(topk):
            # loop for the number of elements in a batch
            for i in range(feature_lens.size()[0]):
                # get encoder output lens
                encoder_len = match.ceil((feature_lens[i]-8)/4)
                # when the range is larger than the number of N-best alignments
                if n >= len(hyp_cache[ids[i]]):
                    n = len(hyp_cache[ids[i]]) - 1
                hyp_len = len(hyp_cache[ids[i]][n])
                if hyp_len != encoder_len:
                    tmp_hyp = hyp_cache[ids[i]][n]
                    # case 1. hyp_cache[ids[i]] is shorter than encoder_lens
                    if hyp_len < encoder_len:
                        hyp_cache[ids[i]][n] = tmp_hyp.extend([0]*(encoder_len-hyp_len))
                    # case 2. hyp_cache[ids[i]] is longer than encoder_lens
                    if hyp_len > encoder_len:
                        hyp_cache[ids[i]][n] = tmp_hyp[:encoder_len]

            # loading hyp_tokens from cache
            hyp_tokens = list()
            for i in range(len(ids)):
                hyp_tokens.append(hyp_cache[ids[i]][n])

            # make pseudo labels
            pseudo_labels = list()
            for i in range(len(hyp_tokens)):
                tmp_pseudo_label = [ t for t in hyp_tokens[i] if t != 0 ]
                pseudo_labels.append(tmp_pseudo_label)

            # convert list(list()) to k2.RaggedTensor
            pseudo_y = k2.RaggedTensor(pseudo_labels).to(device)

            # first, check if the first token is blank
            for i in range(len(hyp_tokens)):
                if hyp_tokens[i][0] != 0:
                    hyp_tokens[i][1:] = hyp_tokens[i][:-1] # shift to right
                    hyp_tokens[i][0] = 0
            # get max length of hyp_tokens
            max_len = 0
            for i in range(len(hyp_tokens)):
                if max_len < len(hyp_tokens[i]):
                    max_len = len(hyp_tokens[i])
            alignment = torch.zeros((len(hyp_tokens), max_len), dtype=torch.int32)

            # get pseudo_y_sequence
            pseudo_y_sequence = alignment.clone().to(device)
            for i in range(len(hyp_tokens)):
                pseudo_y_sequence[i, :len(hyp_tokens[i])] = torch.tensor(hyp_tokens[i], dtype=torch.int32)

            # remove duplicated labels in hyp_tokens
            # and convert the sequence to monotonic increasing sequence
            for i in range(len(hyp_tokens)):
                mask = (torch.tensor(hyp_tokens[i], dtype=torch.bool) != 0)
                alignment[i, :len(hyp_tokens[i])] = torch.cumsum(mask.int(), dim=-1, dtype=torch.int32)
            beam_search_alignment = alignment.to(device)

            # add the element to the list
            nbest_beam_search_alignment.append(beam_search_alignment)
            nbest_pseudo_y.append(pseudo_y)

    return nbest_beam_search_alignment, nbest_pseudo_y


