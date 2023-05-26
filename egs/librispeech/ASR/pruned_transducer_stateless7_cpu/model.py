# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang, Wei Kang)
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

import time
import random
import sys

import k2
import torch
import torch.nn as nn
from encoder_interface import EncoderInterface
from scaling import penalize_abs_values_gt

from icefall.utils import add_sos


class Transducer(nn.Module):
    """It implements https://arxiv.org/pdf/1211.3711.pdf
    "Sequence Transduction with Recurrent Neural Networks"
    """

    def __init__(
        self,
        encoder: EncoderInterface,
        decoder: nn.Module,
        joiner: nn.Module,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
        compress_time_axis: bool = False
    ):
        """
        Args:
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dm) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
        """
        super().__init__()
        assert isinstance(encoder, EncoderInterface), type(encoder)
        assert hasattr(decoder, "blank_id")

        self.encoder = encoder
        self.decoder = decoder
        self.joiner = joiner

        self.simple_am_proj = nn.Linear(
            encoder_dim,
            vocab_size,
        )

        self.simple_lm_proj = nn.Linear(decoder_dim, vocab_size)
        # for debugging
        self.inner_cnt = 0
        self.timer = 0

        # test
        self.compress_time_axis = compress_time_axis
        if self.compress_time_axis:
            self.compress_layer = nn.MaxPool1d(2, stride=2)

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        Returns:
          Return the transducer loss.

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        compress_time_axis = False #self.compress_time_axis
        compress_verbose = False
        d_verbose = True
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axe

        assert x.size(0) == x_lens.size(0) == y.dim0

        encoder_out, x_lens = self.encoder(x, x_lens)
        assert torch.all(x_lens > 0)

        # Now for the decoder, i.e., the prediction network
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = self.decoder(sos_y_padded)

        # Note: y does not start with SOS
        # y_padded : [B, S]
        y_padded = y.pad(mode="constant", padding_value=0)

        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros((x.size(0), 4), dtype=torch.int64, device=x.device)
        boundary[:, 2] = y_lens
        boundary[:, 3] = x_lens

        lm = self.simple_lm_proj(decoder_out)
        if self.compress_time_axis:
            encoder_out_half = encoder_out.permute(0, 2, 1)
            encoder_out_half = self.compress_layer(encoder_out_half)
            encoder_out_half = encoder_out_half.permute(0, 2, 1)

        am = self.simple_am_proj(encoder_out)
        if d_verbose: print("decoder_out.shape: " + str(decoder_out.shape))
        if d_verbose: print("lm.shape: " + str(lm.shape))
        if d_verbose: print("encoder_out.shape: " +str(encoder_out.shape))
        if d_verbose: print("am.shape: " + str(am.shape))
        with torch.cuda.amp.autocast(enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=y_padded,
                termination_symbol=blank_id,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                reduction="sum",
                return_grad=True,
            )
        # if compress
        """
        if False: #self.compress_time_axis:
            boundary_half = boundary.clone()
            boundary_half[:, -1] = boundary[:, -1] // 2
            print(boundary_half)
            #sys.exit(3)
            with torch.cuda.amp.autocast(enabled=False):
              simple_loss_half, (px_grad_half, py_grad_half) = k2.rnnt_loss_smoothed(
                  lm=lm.float(),
                  am=am_half.float(),
                  symbols=y_padded,
                  termination_symbol=blank_id,
                  lm_only_scale=lm_scale,
                  am_only_scale=am_scale,
                  boundary=boundary_half,
                  reduction="sum",
                  return_grad=True,
              )
        """

        if d_verbose: print("px_grad.shape: " + str(px_grad.shape))
        if d_verbose: print("py_grad.shape: " + str(py_grad.shape))
        if d_verbose: print("boundary: " + str(boundary))
        if d_verbose: print("prune_rage: " + str(prune_range))
        # ranges : [B, T, prune_range]

        """
        ranges = k2.get_topk_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
            k=1,
        )

        """
        ranges = k2.get_topk_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
            k=1,
        )

        #even_row = torch.arange(start=0, end=ranges.shape[1], step=2)
        """
        if self.compress_time_axis:
            ranges_half = k2.get_rnnt_prune_ranges(
              px_grad=px_grad_half,
              py_grad=py_grad_half,
              boundary=boundary_half,
              s_range=prune_range,
          )

        """

        # am_pruned : [B, T, prune_range, encoder_dim]
        # lm_pruned : [B, T, prune_range, decoder_dim]

        #even_row_eo = torch.arange(start=0, end=self.joiner.encoder_proj(encoder_out).shape[1], step=2)
        """
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out)[:, even_row_eo],
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges[:, even_row],
        )
        """
        pruned_loss_list = list()
        for k in range(0, 1):
            print("GARBAGE: " + str(k))
            k_range = ranges[:, k, :, :]
            am_pruned, lm_pruned = k2.do_rnnt_pruning(
                am=self.joiner.encoder_proj(encoder_out),
                lm=self.joiner.decoder_proj(decoder_out),
                ranges=k_range,
            )

            #print(self.joiner.encoder_proj(encoder_out).shape)
            #print(am_pruned.shape)
            #if self.compress_time_axis:
            #    am_pruned_half, lm_pruned_half = k2.do_rnnt_pruning(
            #      am=self.joiner.encoder_proj(encoder_out_half),
            #      lm=self.joiner.decoder_proj(decoder_out),
            #      ranges=ranges_half,
            #    )

            if d_verbose: print("am_pruned.shape: " + str(am_pruned.shape))
            if d_verbose: print("lm_pruned.shape: " + str(lm_pruned.shape))
            if d_verbose: print("ranges.shape: " + str(ranges.shape))

            # logits : [B, T, prune_range, vocab_size]
            # project_input=False since we applied the decoder's input projections
            # prior to do_rnnt_pruning (this is an optimization for speed).
            #if self.compress_time_axis:
            #    logits_half = self.joiner(am_pruned_half, lm_pruned_half, project_input=False)
            #    logits = logits_half
            #else:
            logits = self.joiner(am_pruned, lm_pruned, project_input=False)
            if d_verbose: print("logits.shape: " + str(logits.shape))

            #print("logits.shape: " + str(logits.shape))
            #print(logits)
            #logits_half = torch.randn((logits.shape[0], logits.shape[1]//2, logits.shape[2], logits.shape[-1]))
            #logits_half = self.compress_layer(logits[0].permute(-1, 1, 0))
            #logits_half = logits_half.permute(-1, 1, 0)
            #print(logits_half)
            #for i in torch.arange(start=0, end=logits.shape[0]):
            #    logits_half[i] = self.compress_layer(logits[i].permute(-1, 1, 0)).permute(-1, 1, 0)
            #print("logits_half.shape: " + str(logits_half.shape))
            #print(logits_half)
            #print(y_padded)
            #print(boundary)
            #boundary_half = boundary.clone()
            #print(boundary_half[:, -1])
            #boundary_half[:, -1] = boundary[:, -1]//2
            #print(boundary_half)
            #print(ranges.shape)
            #ranges_half = self.compress_layer(ranges.to(torch.float).permute(0, -1, 1)).permute(0, -1, 1).to(torch.int64)
            #print(ranges.dtype)
            #print(ranges[1][:10])
            #print(ranges_half[1][:10])

            #ranges_test = ranges[:, even_row]
            #with torch.cuda.amp.autocast(enabled=False):
            #    pruned_loss = k2.rnnt_loss_pruned(
            #        logits=logits_half.float(),
            #        symbols=y_padded,
            #        ranges=ranges_half,
            #        termination_symbol=blank_id,
            #        boundary=boundary_half,
            #        reduction="sum",
            #    )

            with torch.cuda.amp.autocast(enabled=False):
                pruned_loss = k2.rnnt_loss_pruned(
                    logits=logits.float(),
                    symbols=y_padded,
                    ranges=k_range,
                    termination_symbol=blank_id,
                    boundary=boundary,
                    reduction="sum",
                )
            print("pruned_loss: " + str(pruned_loss))
            pruned_loss_list.append(pruned_loss)

        #if self.compress_time_axis:
        #    with torch.cuda.amp.autocast(enabled=False):
        #        pruned_loss_half = k2.rnnt_loss_pruned(
        #            logits=logits_half.float(),
        #            symbols=y_padded,
        #            ranges=ranges_half,
        #            termination_symbol=blank_id,
        #            boundary=boundary_half,
        #            reduction="sum",
        #        )
        #    print(ranges)
        #    print(ranges.shape)
        #    print(ranges_half)
        #    print(ranges_half.shape)

        if compress_verbose is True:
            limit = 5
            self.inner_cnt = self.inner_cnt + 1
            if self.inner_cnt >= limit:
                sys.exit(3)
        print("simple_loss: " + str(simple_loss))
        pruned_loss = sum(pruned_loss_list)/len(pruned_loss_list)
        print("averaged pruned loss: " + str(pruned_loss))
        #print("simple_loss_half: " + str(pruned_loss_half))
        return (simple_loss, pruned_loss)
