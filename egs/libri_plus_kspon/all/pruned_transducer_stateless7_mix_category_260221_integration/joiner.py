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


class Joiner(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
        num_category: int,
        joiner_type: str = "original",
    ):
        """
        Args:
            encoder_dim: The dimension of the encoder output.
            decoder_dim: The dimension of the decoder output.
            joiner_dim: The dimension of the joiner.
            vocab_size: The size of the vocabulary.
            num_category: The number of categories.
            joiner_type: The type of joiner.
                        "original",
                        "autojoint",                        
                        "autojoint_shortcut",                        
        """
        super().__init__()

        self.encoder_proj = nn.Linear(encoder_dim, joiner_dim)
        self.decoder_proj = nn.Linear(decoder_dim, joiner_dim)
        self.joiner_type = joiner_type
        if self.joiner_type == "original":
            self.middle_linear = None
            self.category_linear = None
            self.output_linear = nn.Linear(joiner_dim, vocab_size)
        elif self.joiner_type in ["autojoint", "autojoint_shortcut"]:
            self.middle_linear = nn.Linear(joiner_dim, vocab_size)
            self.category_linear = nn.Linear(vocab_size, num_category)
            self.output_linear = nn.Linear(num_category, vocab_size)
        

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_out: torch.Tensor,
        project_input: bool = True,
    ) -> torch.Tensor:
        """
        Args:
          encoder_out:
            Output from the encoder. Its shape is (N, T, s_range, C).
          decoder_out:
            Output from the decoder. Its shape is (N, T, s_range, C).
           project_input:
            If true, apply input projections encoder_proj and decoder_proj.
            If this is false, it is the user's responsibility to do this
            manually.
        Returns:
          Return a tensor of shape (N, T, s_range, C).
        """
        assert encoder_out.ndim == decoder_out.ndim
        assert encoder_out.ndim in (2, 4)

        if project_input:
            logit_org = self.encoder_proj(encoder_out) + self.decoder_proj(decoder_out)
        else:
            logit_org = encoder_out + decoder_out

        if self.joiner_type == "original":
            logit = self.output_linear(torch.tanh(logit_org))
            return logit, None
        elif self.joiner_type in ["autojoint"]:       
            logit_org = self.middle_linear(torch.tanh(logit_org))
            category_logit = self.category_linear(logit_org)
            logit = self.output_linear(category_logit)
            return logit, category_logit
        elif self.joiner_type in ["autojoint_shortcut"]:
            logit_org = self.middle_linear(torch.tanh(logit_org))
            category_logit = self.category_linear(logit_org)
            logit = self.output_linear(category_logit) + logit_org
            return logit, category_logit
