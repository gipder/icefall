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
        compress_time_axis = self.compress_time_axis
        compress_verbose = False
        d_verbose = False
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

        if d_verbose: print("decoder_out.shape: " + str(decoder_out.shape))
        lm = self.simple_lm_proj(decoder_out)
        if d_verbose: print("lm.shape: " + str(lm.shape))
        if d_verbose: print("encoder_out.shape: " +str(encoder_out.shape))
        am = self.simple_am_proj(encoder_out)
        if d_verbose: print("am.shape: " + str(am.shape))

        # if self.training and random.random() < 0.25:
        #    lm = penalize_abs_values_gt(lm, 100.0, 1.0e-04)
        # if self.training and random.random() < 0.25:
        #    am = penalize_abs_values_gt(am, 30.0, 1.0e-04)

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

        if d_verbose: print("px_grad.shape: " + str(px_grad.shape))
        if d_verbose: print("py_grad.shape: " + str(py_grad.shape))
        # ranges : [B, T, prune_range]
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )

        # am_pruned : [B, T, prune_range, encoder_dim]
        # lm_pruned : [B, T, prune_range, decoder_dim]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        if d_verbose: print("am_pruned.shape: " + str(am_pruned.shape))
        if d_verbose: print("lm_pruned.shape: " + str(lm_pruned.shape))
        if d_verbose: print("ranges.shape: " + str(ranges.shape))

        def _compress_with_maxpool(am_pruned, lm_pruned, ranges, boundary):
            assert am_pruned.shape[1] == lm_pruned.shape[1]
            assert am_pruned.shape[1] == ranges.shape[1]
            max_t = am_pruned.shape[1]
            #reshape (B, T, S, C) -> (B, C, S, T)
            am_pruned_transposed = am_pruned.permute(0, 3, 2, 1)
            lm_pruned_transposed = lm_pruned.permute(0, 3, 2, 1)
            ranges_transposed = ranges.permute(0, 2, 1)

            size = 2
            am0 = am_pruned_transposed.shape[0]
            am1 = am_pruned_transposed.shape[1]
            am2 = am_pruned_transposed.shape[2]
            am3 = am_pruned_transposed.shape[3] // size
            lm0 = lm_pruned_transposed.shape[0]
            lm1 = lm_pruned_transposed.shape[1]
            lm2 = lm_pruned_transposed.shape[2]
            lm3 = lm_pruned_transposed.shape[3] // size
            ra0 = ranges_transposed.shape[0]
            ra1 = ranges_transposed.shape[1]
            ra2 = ranges_transposed.shape[2] // size


            am_pruned_result = torch.randn((am0, am1, am2, am3), dtype=am_pruned_transposed.dtype)
            lm_pruned_result = torch.randn((am0, am1, am2, am3), dtype=lm_pruned_transposed.dtype)
            ranges_result = torch.randn((ra0, ra1, ra2), dtype=torch.float)
            for b in range(0, am0):
                am_pruned_result[b] = self.compress_layer(am_pruned_transposed[b])
                lm_pruned_result[b] = self.compress_layer(lm_pruned_transposed[b])
                ranges_result[b] = self.compress_layer(ranges_transposed[b].type(torch.float))

            #reshape (B, C, S, T) -> (B, T, S, C)
            am_pruned_result = am_pruned_result.permute(0, 3, 2, 1)
            lm_pruned_result = lm_pruned_result.permute(0, 3, 2, 1)
            ranges_result = ranges_result.permute(0, 2, 1)
            ranges_result = ranges_result.type(ranges_transposed.dtype)

            #change boundary
            gap_t = max_t - am_pruned_result.shape[1]
            boundary_result = boundary.clone()
            boundary_result[:, -1] = boundary_result[:, -1] - gap_t

            #device
            am_pruned_result = am_pruned_result.to(am_pruned.device)
            lm_pruned_result = lm_pruned_result.to(lm_pruned.device)
            ranges_result = ranges_result.to(ranges.device)
            return am_pruned_result, lm_pruned_result, ranges_result, boundary_result
        """
        def _compress_am_and_lm(am_pruned, lm_pruned, ranges, boundary):
            import random
            section_size = 4
            assert am_pruned.shape[1] == lm_pruned.shape[1]
            assert am_pruned.shape[1] == ranges.shape[1]
            #logits_result = torch.zeros(1, 1)
            #ranges_result = torch.zeors(1, 1)
            max_t = am_pruned.shape[1]
            for i in range(0, am_pruned.shape[1], section_size):
                am_pruned_section = am_pruned[:, i:i+section_size, :]
                lm_pruned_section = lm_pruned[:, i:i+section_size, :]
                ranges_section = ranges[:, i:i+section_size, :]
                rand_idx = random.randint(i, i+section_size-1)
                am_pruned_section = torch.cat((am_pruned_section[:, :rand_idx-i], am_pruned_section[:, rand_idx+1-i:]), dim=1)
                lm_pruned_section = torch.cat((lm_pruned_section[:, :rand_idx-i], lm_pruned_section[:, rand_idx+1-i:]), dim=1)
                ranges_section = torch.cat((ranges_section[:, :rand_idx-i], ranges_section[:, rand_idx+1-i:]), dim=1)
                if i == 0:
                    am_pruned_result = am_pruned_section
                    lm_pruned_result = lm_pruned_section
                    ranges_result = ranges_section
                else:
                    am_pruned_result = torch.cat((am_pruned_result, am_pruned_section), dim=1)
                    lm_pruned_result = torch.cat((lm_pruned_result, lm_pruned_section), dim=1)
                    ranges_result = torch.cat((ranges_result, ranges_section), dim=1)

            gap_t = max_t - am_pruned_result.shape[1]
            boundary_result = boundary.clone()
            boundary_result[:, -1] = boundary_result[:, -1] - gap_t
            return am_pruned_result, lm_pruned_result, ranges_result, boundary_result
        """

        # logits : [B, T, prune_range, vocab_size]
        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        if compress_time_axis is False:
            logits = self.joiner(am_pruned, lm_pruned, project_input=False)
        else:
            c_am_pruned, c_lm_pruned, c_ranges, c_boundary = _compress_with_maxpool(am_pruned, lm_pruned, ranges, boundary)
            logits = self.joiner(c_am_pruned, c_lm_pruned, project_input=False)
            c_logits = logits
        if d_verbose:print("logits.shape: " + str(logits.shape))
        if compress_time_axis is False:
            if d_verbose: print(boundary)
            #start = time.time()
            with torch.cuda.amp.autocast(enabled=False):
                pruned_loss = k2.rnnt_loss_pruned(
                    logits=logits.float(),
                    symbols=y_padded,
                    ranges=ranges,
                    termination_symbol=blank_id,
                    boundary=boundary,
                    reduction="sum",
                )
            #self.timer = self.timer + (time.time() - start )
        else:
            if d_verbose:
                print("Before compressing")
                print("compress_time_axis: " + str(compress_time_axis))
                print("logits.shape: " + str(logits.shape))
                print("y_padded.shape: " + str(y_padded.shape))
                print("ranges.shape: " + str(ranges.shape))
                print("boundary.shape: " + str(boundary.shape))
                print("After compressing")

            def _compress_time(logits, ranges, boundary):
                section_size = 10
                assert logits.shape[1] == ranges.shape[1]
                #logits_result = torch.zeros(1, 1)
                #ranges_result = torch.zeors(1, 1)
                max_t = logits.shape[1]
                for i in range(0, logits.shape[1], section_size):
                    logits_section = logits[:, i:i+section_size, :]
                    ranges_section = ranges[:, i:i+section_size, :]
                    rand_idx = random.randint(i, i+section_size-1)
                    logits_section = torch.cat((logits_section[:, :rand_idx-i], logits_section[:, rand_idx+1-i:]), dim=1)
                    ranges_section = torch.cat((ranges_section[:, :rand_idx-i], ranges_section[:, rand_idx+1-i:]), dim=1)
                    if i == 0:
                        logits_result = logits_section
                        ranges_result = ranges_section
                    else:
                        logits_result = torch.cat((logits_result, logits_section), dim=1)
                        ranges_result = torch.cat((ranges_result, ranges_section), dim=1)

                gap_t = max_t - logits_result.shape[1]
                boundary_result = boundary.clone()
                boundary_result[:, -1] = boundary_result[:, -1] - gap_t
                return logits_result, ranges_result, boundary_result

            #c_logits, c_ranges, c_boundary = _compress_time(logits, ranges, boundary)
            if d_verbose:
                print("compressed_logits.shape: " + str(c_logits.shape))
            if d_verbose:
                print("compressed_ranges.shape: " + str(c_ranges.shape))
            with torch.cuda.amp.autocast(enabled=False):
                pruned_loss = k2.rnnt_loss_pruned(
                    logits=c_logits.float(),
                    symbols=y_padded,
                    ranges=c_ranges,
                    termination_symbol=blank_id,
                    boundary=c_boundary,
                    reduction="sum",
                )

        if compress_verbose is True:
            limit = 1
            self.inner_cnt = self.inner_cnt + 1
            if self.inner_cnt >= limit:
                sys.exit(3)
        return (simple_loss, pruned_loss)
