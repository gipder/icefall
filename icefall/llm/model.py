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
import sentencepiece as spm

from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

class CustomGPT2Model(nn.Module):
    def __init__(self,
            llm: str="gpt2",
            use_icefall_vocab: bool=False,
            params: Optional[AttributeDict]=None
            ):
        super(CustomGPT2Model, self).__init__()
        self.plm = GPT2LMHeadModel.from_pretrained(llm)
        self.config = GPT2Config.from_pretrained(llm)
        self.tokenizer = None

        self.use_icefall_vocab = use_icefall_vocab
        if not use_icefall_vocab:
            self.tokenizer = GPT2Tokenizer.from_pretrained(llm)            
        else:
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.Load(params.bpe_model)            

        self.criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.0)
        
        if self.use_icefall_vocab:
            # config.vocab_size is the size of the pretrained model's vocab
            # tokenizer.vocab_size() is the size of the tokenizer's vocab from icefall
            #self.proj_layer = nn.Linear(self.tokenizer.vocab_size(), self.tokenizer.vocab_size())
            #original code
            # the external embeddings isn't used 
            """
            self.embd_layer = nn.Embedding(num_embeddings=self.tokenizer.vocab_size(), 
                                           embedding_dim=self.config.n_embd)
            """
            # resize the token embeddings to match the size of the tokenizer's vocab
            self.plm.resize_token_embeddings(self.tokenizer.vocab_size())
        self.proj_layers = []
        if params.num_proj > 0:
            for i in range(params.num_proj):
                self.proj_layers.append(nn.Linear(self.tokenizer.vocab_size(), self.tokenizer.vocab_size()))
                #self.proj_layers.append(nn.ReLU())
        self.proj_layers = nn.ModuleList(self.proj_layers)

    def forward(self, x=None, y=None, sentence_lengths=None, return_logits=False, **kwargs):
        try:
            #if self.use_icefall_vocab:
            #    x = self.embd_layer(x)
            #    outputs = self.plm(inputs_embeds=x)
            #else:
            outputs = self.plm(input_ids=x)
            #print(f"{outputs.logits.shape=}")
            #outputs = self.proj_layers(outputs)
            #for i in range(len(self.proj_layers)):
            #    outputs.logits = self.proj_layers[i](outputs.logits)
            
        except Exception as e:
            print(e)
            print(f'x.shape: {x.shape}')
            sys.exit(3)
        logits = outputs.logits
        if self.use_icefall_vocab:
            for i in range(len(self.proj_layers)):
                logits = self.proj_layers[i](logits)
                    
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
        if not self.use_icefall_vocab:
            for param in self.plm.lm_head.parameters():
                param.requires_grad = True
        else:
            for param in self.plm.lm_head.parameters():
                param.requires_grad = True
            #for param in self.proj_layer.parameters():
            #    param.requires_grad = True
            # original code
            #for param in self.embd_layer.parameters():
            #    param.requires_grad = True
            # this loop is able to use the pretrained model's token embeddings
            for param in self.plm.transformer.wte.parameters():
                param.requires_grad = True
            for param in self.proj_layers.parameters():
                param.requires_grad = True
            """
            if len(self.proj_layers) > 0:
                for i in range(len(self.proj_layers)):
                    for param in self.proj_layers[i].parameters():
                        param.requires_grad = True
            """

    """
    def eval(self):
        for param in self.plm.parameters():
            param.requires_grad = False
    """
