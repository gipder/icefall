# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple
from icefall.utils import AttributeDict, add_eos, add_sos, make_pad_mask
from pretrained_model import CustomGPT2Model

class Decoder(nn.Module):
    """This class modifies the stateless decoder from the following paper:

        RNN-transducer with stateless prediction network
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9054419

    It removes the recurrent connection from the decoder, i.e., the prediction
    network. Different from the above paper, it adds an extra Conv1d
    right after the embedding layer.

    TODO: Implement https://arxiv.org/pdf/2109.07513.pdf
    """

    def __init__(
        self,
        vocab_size: int,
        decoder_dim: int,
        blank_id: int,
        context_size: int,
    ):
        """
        Args:
          vocab_size:
            Number of tokens of the modeling unit including blank.
          decoder_dim:
            Dimension of the input embedding, and of the decoder output.
          blank_id:
            The ID of the blank symbol.
          context_size:
            Number of previous words to use to predict the next word.
            1 means bigram; 2 means trigram. n means (n+1)-gram.
        """
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=decoder_dim,
        )
        self.blank_id = blank_id

        assert context_size >= 1, context_size
        self.context_size = context_size
        self.vocab_size = vocab_size
        if context_size > 1:
            self.conv = nn.Conv1d(
                in_channels=decoder_dim,
                out_channels=decoder_dim,
                kernel_size=context_size,
                padding=0,
                groups=decoder_dim // 4,  # group size == 4
                bias=False,
            )

    def forward(self, y: torch.Tensor, need_pad: bool = True) -> torch.Tensor:
        """
        Args:
          y:
            A 2-D tensor of shape (N, U).
          need_pad:
            True to left pad the input. Should be True during training.
            False to not pad the input. Should be False during inference.
        Returns:
          Return a tensor of shape (N, U, decoder_dim).
        """
        y = y.to(torch.int64)
        # this stuff about clamp() is a temporary fix for a mismatch
        # at utterance start, we use negative ids in beam_search.py
        if torch.jit.is_tracing():
            # This is for exporting to PNNX via ONNX
            embedding_out = self.embedding(y)
        else:
            embedding_out = self.embedding(y.clamp(min=0)) * (y >= 0).unsqueeze(-1)

        print(f"{embedding_out.shape=}")
        if self.context_size > 1:
            embedding_out = embedding_out.permute(0, 2, 1)
            if need_pad is True:
                embedding_out = F.pad(embedding_out, pad=(self.context_size - 1, 0))
            else:
                # During inference time, there is no need to do extra padding
                # as we only need one output
                assert embedding_out.size(-1) == self.context_size
            embedding_out = self.conv(embedding_out)
            embedding_out = embedding_out.permute(0, 2, 1)
        embedding_out = F.relu(embedding_out)
        return embedding_out

class PretrainedDecoder(Decoder):
    """This class modifies the stateless + pretrained model (GPT2) decoder:

    It removes the recurrent connection from the decoder, i.e., the prediction
    network. Different from the above paper, it adds an extra Conv1d
    right after the embedding layer.
    """
    def __init__(
        self,
        vocab_size: int,
        decoder_dim: int,
        blank_id: int,
        context_size: int,
        llm_name: str,
        use_icefall_vocab: bool = False,
        params: Optional[AttributeDict] = None,
        use_embedding: bool = False,
    ):
        """
        Args:
          vocab_size:
            Number of tokens of the modeling unit including blank.
          decoder_dim:
            Dimension of the input embedding, and of the decoder output.
          blank_id:
            The ID of the blank symbol.
          context_size:
            Number of previous words to use to predict the next word.
            1 means bigram; 2 means trigram. n means (n+1)-gram.
          llm_name:
            Name of the pretrained language model.
          use_icefall_vocab:
            Whether to use the icefall vocab.
          params:
            Parameters for the pretrained language model from default parameters.
        """
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=decoder_dim,
        )
        self.blank_id = blank_id

        self.plm = CustomGPT2Model(llm=llm_name,
                                   use_icefall_vocab=use_icefall_vocab,
                                   params=params,
                                   use_embedding=use_embedding,
                                   )
        self.mid_proj_layer = nn.Linear(decoder_dim, self.plm.config.n_embd)
        if use_icefall_vocab:
            assert decoder_dim == self.plm.config.n_embd

        if use_embedding:
            conv_input_dim = self.plm.config.n_embd
        else:
            conv_input_dim = decoder_dim
        assert context_size >= 1, context_size
        self.context_size = context_size
        self.vocab_size = vocab_size
        if context_size > 1:
            self.conv = nn.Conv1d(
                in_channels=conv_input_dim,
                out_channels=decoder_dim,
                kernel_size=context_size,
                padding=0,
                groups=decoder_dim // 4,  # group size == 4
                bias=False,
            )

    def forward(self, y: torch.Tensor, need_pad: bool = True) -> torch.Tensor:
        """
        Args:
          y:
            A 2-D tensor of shape (N, U).
          need_pad:
            True to left pad the input. Should be True during training.
            False to not pad the input. Should be False during inference.
        Returns:
          Return a tensor of shape (N, U, decoder_dim).
        """
        y = y.to(torch.int64)
        # this stuff about clamp() is a temporary fix for a mismatch
        # at utterance start, we use negative ids in beam_search.py
        if torch.jit.is_tracing():
            # This is for exporting to PNNX via ONNX
            embedding_out = self.embedding(y)
        else:
            embedding_out = self.embedding(y.clamp(min=0)) * (y >= 0).unsqueeze(-1)

        print(f"{embedding_out.shape=}")
        # pretrained language model
        embedding_out = self.mid_proj_layer(embedding_out)
        print(f"after mid_proj_layer: {embedding_out.shape=}")
        embedding_out = self.plm(x=embedding_out)
        print(f"after plm: {embedding_out.shape=}")

        if self.context_size > 1:
            embedding_out = embedding_out.permute(0, 2, 1)
            if need_pad is True:
                embedding_out = F.pad(embedding_out, pad=(self.context_size - 1, 0))
            else:
                # During inference time, there is no need to do extra padding
                # as we only need one output
                assert embedding_out.size(-1) == self.context_size
            embedding_out = self.conv(embedding_out)
            embedding_out = embedding_out.permute(0, 2, 1)
        embedding_out = F.relu(embedding_out)
        return embedding_out

    def train(self, mode:bool=True):
        self.plm.train()
        super().train()

    def eval(self, mode:bool=True):
        self.plm.eval()
        for p in self.parameters():
            p.requires_grad = False

if __name__ == "__main__":
    x = torch.randint(0, 500, (1, 10, ))

    # test Decoder
    decoder = Decoder(
                vocab_size=500,
                decoder_dim=512,
                blank_id=0,
                context_size=2,
            )

    out = decoder(x)
    print(f"{x}")
    print(f"{out=}")
    print(f"{x.shape=}")
    print(f"{out.shape=}")
    print("basic Decoder test done")
    print("*" * 80)

    params = AttributeDict()
    params.bpe_model="/home/sskim/work/icefall/egs/librispeech/ASR/data/lang_bpe_500/bpe.model"
    params.max_sent_len = 256
    params.num_proj = 2
    params.batch_size = 5
    params.sos_id = 1
    params.eos_id = 1
    params.blank_id = 0
    decoder = PretrainedDecoder(
                vocab_size=500,
                decoder_dim=768,
                blank_id=0,
                context_size=2,
                llm_name="gpt2",
                use_icefall_vocab=True,
                params=params,
                use_embedding=False,
                )

    out = decoder(x)
    print(f"{x}")
    print(f"{out=}")
    print(f"{x.shape=}")
    print(f"{out.shape=}")
    num_param = sum(p.numel() for p in decoder.parameters())
    print(f"all params: {num_param=}")
    decoder.train()
    num_param = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"decoder.train() trainable params: {num_param=}")
    decoder.eval()
    num_param = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"decoder.eval() trainable params: {num_param=}")
    print("PretrainedDecoder test done")


