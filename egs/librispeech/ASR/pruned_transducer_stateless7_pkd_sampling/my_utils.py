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
import sentencepiece as spm
from llm_gen import LLMGenDict, LLMGenDB

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
                encoder_len = math.ceil((feature_lens[i]-8)/4)
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

def make_llm_gen_label(batch: dict,
                       llm_gen_db: LLMGenDB,
                       params: AttributeDict,
                       sp: spm.SentencePieceProcessor,
                       device: torch.device = torch.device("cpu"),
                       ) -> k2.RaggedTensor:
    """
    Make n-best alignments from the given batch and hypotheses dictionary.
    The n-best alignments contains N alignments for each utterance in the batch.
    The type of the n-best alignments is a list of Tensor, which is a tensor of Torch.

    Args:
        batch (dict):
            The batch of icefall.
        llm_gen_db (LLMGenDB):
            The serialized dictionary contains LLM Generated Labels
        params (AttributeDict):
            The parameters of icefall.
        device (torch.device):
            The device to run the model.
    Returns:
        llm_gen_pseudo_y (k2.RaggedTensor):
            LLM generated pseudo labels.
    """

    ids = list()
    # get ids from batch
    for i in range(len(batch["supervisions"]["cut"])):
        ids.append(batch["supervisions"]["cut"][i].id)

    feature_lens = batch["supervisions"]["num_frames"]

    assert llm_gen_db is not None, "hyp_cache is None"

    llm_gen_pseudo_y = list()
    is_all_in_cache = True
    for i in range(len(ids)):
        if llm_gen_db.get_entry(ids[i]) is None:
            is_all_in_cache = False
            break

    if not is_all_in_cache:
        logging.error("Not all ids are in llm_gen_db")
        logging.error(f"{ids=}")
        assert False

    # loop for the number of elements in a batch
    texts = list()
    for i in range(len(ids)):
        # wer is less then the threshold
        if llm_gen_db.get_wer(ids[i]) < params.llm_gen_threshold:
            texts.append(llm_gen_db.get_value(ids[i]))
        else:
            texts.append(llm_gen_db.get_origin(ids[i]))

    pseudo_y = sp.encode(texts, out_type=int)
    llm_gen_pseudo_y.append(k2.RaggedTensor(pseudo_y).to(device))

    return llm_gen_pseudo_y

def create_pseudo_alignment(x_len, y_len):
    """
    Create pseudo alignment from the given x_len and y_len.
    The pseudo alignment is a tensor of Torch.
    """
    padded_y_len = y_len + 1
    if x_len < padded_y_len:
        padded_y_len = x_len.item()
    y = torch.arange(padded_y_len, dtype=int)
    base_repeats = np.ones(padded_y_len, dtype=int)
    remaining_len = x_len - padded_y_len
    additional_repeats = np.random.multinomial(remaining_len, np.ones(padded_y_len)/padded_y_len)
    repeats = base_repeats + additional_repeats

    ret = torch.cat([torch.full((count,), num) for num, count in zip(y, repeats)])

    return ret

def create_increasing_sublists_torch(original_tensor, length=5):
    """
    Given an alignment tensor, create a list of pruned range tensor by the given length.
    """
    # 배치 크기와 원소 수
    batch_size, num_elements = original_tensor.shape

    # 최대값, 최소값
    max_val = torch.max(original_tensor, dim=1, keepdim=True)[0]
    min_val = torch.min(original_tensor, dim=1, keepdim=True)[0]

    half_length = torch.div(length-1, 2, rounding_mode='floor')  #(length - 1) // 2

    # 각 원소에 대해 시작값 계산
    starts = original_tensor - half_length

    # 최소값에 따라 시작값 조정
    starts = torch.clamp(starts, min_val, max_val - length + 1)

    # 각 시작점에 대해 서브리스트 생성 (증가하는 범위)
    result_tensor = torch.arange(0, length, device=original_tensor.device)
    result_tensor = result_tensor.expand(batch_size, num_elements, -1) + starts.unsqueeze(-1)

    # 최대값에 따라 클램핑 없이 증가 범위 유지
    result_tensor = torch.min(result_tensor, max_val.unsqueeze(-1))

    return result_tensor
