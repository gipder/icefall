# Copyright (c)  2022  Xiaomi Corporation (authors: Xiaoyu Yang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Optional, Tuple
import sys

import torch
import torch.nn.functional as F
import torch.nn as nn

from icefall.transformer_lm.encoder import Transformer
from icefall.utils import AttributeDict, add_eos, add_sos, make_pad_mask

from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

class CustomGPT2Model(nn.Module):
    def __init__(self, llm: str="gpt2"):
        super(CustomGPT2Model, self).__init__()
        self.plm = GPT2LMHeadModel.from_pretrained(llm)
        self.config = GPT2Config.from_pretrained(llm)
        self.tokenizer = GPT2Tokenizer.from_pretrained(llm)
        self.criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.0)

    def forward(self, x=None, y=None, sentence_lengths=None, return_logits=False, **kwargs):
        try:
            outputs = self.plm(input_ids=x)
        except Exception as e:
            print(e)
            print(f'x.shape: {x.shape}')
            sys.exit(3)
        logits = outputs.logits
        nll_loss = self.criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        mask = make_pad_mask(sentence_lengths).reshape(-1).to(nll_loss.device)
        nll_loss.masked_fill_(mask, 0)

        has_nan = torch.isnan(nll_loss).any()
        if has_nan:
            print("from GPT outputs")
            print(f"outputs: {outputs}")
            print(f"outputs[0]: {outputs[0]}")
            print(f"torch.min(outputs[0]): {torch.min(outputs[0])}")
            print(f"torch.max(outputs[0]): {torch.max(outputs[0])}")
            print(f"logits: {logits}")
            print(f"logits.shape: {logits.shape}")
            print(f'x.shape: {x.shape}')
            torch.set_printoptions(profile="full")
            print(f'x: {x}')
            print(f'y: {y}')
            print(f'y.shape: {y.shape}')
            print(f'nll_loss: {nll_loss}')
            print(f'sentence_lengths: {sentence_lengths}')
            torch.set_printoptions(profile="dafault")
            import sys
            sys.exit()
        return nll_loss, logits if return_logits else nll_loss

    def train(self, mode:bool=True):        
        for param in self.plm.parameters():
            param.requires_grad = False
        
        for param in self.plm.lm_head.parameters():
            param.requires_grad = True        
    """
    def eval(self):        
        for param in self.plm.parameters():
            param.requires_grad = False
    """
