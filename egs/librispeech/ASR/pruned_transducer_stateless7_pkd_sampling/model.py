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


import random

import k2
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder_interface import EncoderInterface
from scaling import penalize_abs_values_gt

from icefall.utils import add_sos
from typing import Union

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
        self.kd_criterion = nn.KLDivLoss(reduction='batchmean')
        self.teacher_simple_am_proj = None
        self.teacher_simple_lm_proj = None
        self.ctc_layer = nn.Linear(encoder_dim, vocab_size)
        self.ctc_criterion = None
        self.alphas_criterion = nn.MSELoss(reduction='none')

    def forward(
        self,
        x: torch.tensor,
        x_lens: torch.tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        use_ctc: bool = False,
        use_kd: bool = False,
        use_efficient: bool = False,
        use_1best: bool = False,
        use_nbest: bool = False,
        use_pruned: bool = False,
        use_sq_sampling: bool = False,
        use_sq_simple_loss_range: bool = False,
        use_topk_shuff: bool = False,
        teacher_model: nn.Module = None,
        pruned_kd_range: int = 5,
        sq_sampling_num: int = 1,
        topk: int = 1,
        nbest_beam_search_alignment: Union[list, torch.tensor] = None,
        nbest_sampling_y: Union[list, k2.RaggedTensor] = None,
    ) -> torch.tensor:
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
          use_ctc:
            If True, use CTC loss
          use_kd:
            If True, use knowledge distillation
          use_efficient:
            If True, compress the logits for RNN-T from K-dims to 3-dims
            The 3-dims consists of Prob of y, Prob. of blank, and Prob. of others
            The Prob. of others can be calculated as 1 - Prob. of y - Prob. of blank
          use_1best:
            If True, use 1-best alignment from teacher model
          use_nbest:
            If True, use N-best alignment from teacher model
          use_pruned:
            If True, use pruned range from teacher model
          use_sq_sampling:
            If True, use sequence sampling
          use_sq_simple_loss_range:
            If True, use simple loss to get the range for alignments
          use_topk_shuffle:
            If True, instead of considering all N-best alignments,
            consider only one alignment at each epoch.
          teacher_model:
            The pretrained teacher model for knowledge distillation
          pruned_kd_range:
            The prune range for knowledge distillation. It is different from "prune_range"
          sq_sampling_num:
            The number of samples for sequence sampling
          topk:
            The number of topk alignments to consider. It is the same as N of N-best
          nbest_beam_search_alignment:
            The N-best alignments from beam search
          nbest_sampling_y:
            The N-best label sequences for sequence sampling

        Returns:
          Return the transducer loss.

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs +
              ctc_scale * ctc_probs +
              kd_scale * kd_probs +
              kd_scale * sampling_scale * sampling_probs

        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0

        # getting student encoder output and length
        #print(f"{x=}")
        org_x_lens = x_lens
        encoder_out, x_lens = self.encoder(x, x_lens)
        assert torch.all(x_lens > 0)
        #print(f"{encoder_out=}")
        #print(f"{x_lens=}")
        # Now for the decoder, i.e., the prediction network
        #print(f"{y=}")
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
        am = self.simple_am_proj(encoder_out)

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
        #print(f"{simple_loss=}")
        # simple_loss is inf or nan, then terminate training
        #if not torch.isfinite(simple_loss):
        #    print("simple_loss is inf or nan, then terminate training")
        #    import sys
        #    sys.exit(0)
        #print(f"inside model.py, {teacher_logits=}")
        #if use_pkd:
        #    ranges = teacher_ranges
        #else:

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
        # logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        with torch.cuda.amp.autocast(enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction="none",
            )

        if torch.isnan(pruned_loss).any() or torch.isinf(pruned_loss).any():
            idx = torch.where(torch.isnan(pruned_loss) | torch.isinf(pruned_loss))
            print(f"{pruned_loss=}")
            print("pruned_loss is nan or inf, then terminate training")
            print(f"{idx=}")
            #print(f"{logits=}")
            print(f"{x_lens=}")
            print(f"{y_lens=}")
            print(f"{ranges[idx]=}")
            print(f"{y_padded[idx]=}")
            print(f"{y_padded[0]=}")
            import sys
            sys.exit(0)
        pruned_loss = pruned_loss.sum()

        # get logit with pruned_kd_range
        if use_kd:
            if use_sq_simple_loss_range:
                tmp_ranges = k2.get_rnnt_prune_ranges(
                    px_grad=px_grad,
                    py_grad=py_grad,
                    boundary=boundary,
                    s_range=pruned_kd_range,
                )
            else:
                tmp_ranges = ranges

            # get teacher encoder out
            teacher_encoder_out, teacher_encoder_out_lens = teacher_model.get_encoder_out(
                x=x,
                x_lens=org_x_lens,
                use_grad=False,
            )

            assert pruned_kd_range > 1, f"{pruned_kd_range=} should be larger than 1"
            # get teacher logits with student range
            teacher_logits = teacher_model.get_logits_with_encoder_out_and_ranges(
                encoder_out=teacher_encoder_out,
                encoder_out_lens=teacher_encoder_out_lens,
                y=y,
                ranges=tmp_ranges,
                prune_range=pruned_kd_range,
                use_grad=False,
            )

            assert not (use_1best and use_nbest), "use_1best and use_nbest cannot be True at the same time"
            assert not (use_1best and use_pruned), "use_1best and use_pruned cannot be True at the same time"
            assert not (use_nbest and use_pruned), "use_nbest and use_pruned cannot be True at the same time"
            if use_1best:
                alignment = nbest_beam_search_alignment[0].unsqueeze(-1).to(torch.int64)
                teacher_logits = teacher_model.get_logits_with_encoder_out_and_ranges(
                    encoder_out=teacher_encoder_out,
                    encoder_out_lens=teacher_encoder_out_lens,
                    y=nbest_sampling_y[0],
                    ranges=alignment,
                    prune_range=1,
                    use_grad=False,
                )
                student_logits = self.get_logits_with_encoder_out_and_ranges(
                    encoder_out=encoder_out,
                    encoder_out_lens=x_lens,
                    y=nbest_sampling_y[0],
                    ranges=alignment,
                    prune_range=1,
                    use_grad=True,
                )
                student = F.log_softmax(student_logits, dim=-1)
                teacher = F.softmax(teacher_logits, dim=-1)
                max_len = student.size(1)
                mask = torch.arange(max_len, device=student.device).expand(student.size(0), max_len) < x_lens.unsqueeze(1)
                mask = mask.unsqueeze(-1).unsqueeze(-1)
                student = student * mask.float()
                teacher = teacher * mask.float()

            elif use_nbest:
                if not use_topk_shuff:
                    nbest_student = None
                    nbest_teacher = None
                    for n in range(0, topk):
                        alignment = nbest_beam_search_alignment[n].unsqueeze(-1).to(torch.int64)
                        teacher_logits = teacher_model.get_logits_with_encoder_out_and_ranges(
                            encoder_out=teacher_encoder_out,
                            encoder_out_lens=teacher_encoder_out_lens,
                            y=nbest_sampling_y[n],
                            ranges=alignment,
                            prune_range=1,
                            use_grad=False,
                        )
                        student_logits = self.get_logits_with_encoder_out_and_ranges(
                            encoder_out=encoder_out,
                            encoder_out_lens=x_lens,
                            y=nbest_sampling_y[n],
                            ranges=alignment,
                            prune_range=1,
                            use_grad=True,
                        )
                        student = F.log_softmax(student_logits, dim=-1)
                        teacher = F.softmax(teacher_logits, dim=-1)
                        max_len = student.size(1)
                        mask = torch.arange(max_len, device=student.device).expand(student.size(0), max_len) < x_lens.unsqueeze(1)
                        mask = mask.unsqueeze(-1).unsqueeze(-1)
                        student = student * mask.float()
                        teacher = teacher * mask.float()
                        nbest_student = torch.cat([nbest_student, student], dim=2) if nbest_student is not None else student
                        nbest_teacher = torch.cat([nbest_teacher, teacher], dim=2) if nbest_teacher is not None else teacher
                    student = nbest_student
                    teacher = nbest_teacher
                else:
                    alignment = nbest_beam_search_alignment[0].unsqueeze(-1).to(torch.int64)
                    teacher_logits = teacher_model.get_logits_with_encoder_out_and_ranges(
                        encoder_out=teacher_encoder_out,
                        encoder_out_lens=teacher_encoder_out_lens,
                        y=nbest_sampling_y[0],
                        ranges=alignment,
                        prune_range=1,
                        use_grad=False,
                    )
                    student_logits = self.get_logits_with_encoder_out_and_ranges(
                        encoder_out=encoder_out,
                        encoder_out_lens=x_lens,
                        y=nbest_sampling_y[0],
                        ranges=alignment,
                        prune_range=1,
                        use_grad=True,
                    )
                    student = F.log_softmax(student_logits, dim=-1)
                    teacher = F.softmax(teacher_logits, dim=-1)
                    max_len = student.size(1)
                    mask = torch.arange(max_len, device=student.device).expand(student.size(0), max_len) < x_lens.unsqueeze(1)
                    mask = mask.unsqueeze(-1).unsqueeze(-1)
                    student = student * mask.float()
                    teacher = teacher * mask.float()
            elif use_pruned:
                student_logits = self.get_logits_with_encoder_out_and_ranges(
                    encoder_out=encoder_out,
                    encoder_out_lens=x_lens,
                    y=y,
                    ranges=tmp_ranges,
                    prune_range=pruned_kd_range,
                    use_grad=True,
                )

                teacher_logits = teacher_model.get_logits_with_encoder_out_and_ranges(
                    encoder_out=teacher_encoder_out,
                    encoder_out_lens=teacher_encoder_out_lens,
                    y=y,
                    ranges=tmp_ranges,
                    prune_range=pruned_kd_range,
                    use_grad=False,
                )
                student = F.log_softmax(student_logits, dim=-1)
                teacher = F.softmax(teacher_logits, dim=-1)
                max_len = student.size(1)
                mask = torch.arange(max_len, device=student.device).expand(student.size(0), max_len) < x_lens.unsqueeze(1)
                mask = mask.unsqueeze(-1).unsqueeze(-1)
                student = student * mask.float()
                teacher = teacher * mask.float()
            elif use_efficient:
                assert False, "Not implemented yet"
                pass
            else:
                student_logits = self.get_logits_with_encoder_out_and_ranges(
                    encoder_out=encoder_out,
                    encoder_out_lens=x_lens,
                    y=y,
                    ranges=ranges,
                    prune_range=prune_range,
                    use_grad=True,
                )

                teacher_logits = teacher_model.get_logits_with_encoder_out_and_ranges(
                    encoder_out=teacher_encoder_out,
                    encoder_out_lens=teacher_encoder_out_lens,
                    y=y,
                    ranges=ranges,
                    prune_range=prune_range,
                    use_grad=False,
                )
                student = F.log_softmax(student_logits, dim=-1)
                teacher = F.softmax(teacher_logits, dim=-1)
                max_len = student.size(1)
                mask = torch.arange(max_len, device=student.device).expand(student.size(0), max_len) < x_lens.unsqueeze(1)
                mask = mask.unsqueeze(-1).unsqueeze(-1)
                student = student * mask.float()
                teacher = teacher * mask.float()
            kd_loss = self.kd_criterion(student, teacher)

        if use_sq_sampling:
            assert use_1best is False, "1best kd is not supported with sequence sampling"
            assert use_nbest is False, "nbest kd is not supported with sequence sampling"

            sampling_loss = torch.tensor(0.0).to(logits.device)
            teacher_sampling_ranges = list()
            teacher_sampling_logits = list()
            sampling_y = list()
            for n in range(1, sq_sampling_num+1):
                yy = list()
                y_list = y.tolist()
                # in the case of the number of samples is 1
                # y_list[(i+1)%len(y_list)] means the next sample in a batch
                for i in range(len(y_list)):
                    yy.append(y_list[(i+n)%len(y_list)])

                sampling_y.append(k2.RaggedTensor(yy).to(x.device))
                # the yy is the less probable label sequence

                sampling_ret = teacher_model.get_ranges_and_logits_with_encoder_out(
                    encoder_out=teacher_encoder_out,
                    encoder_out_lens=teacher_encoder_out_lens,
                    y=sampling_y[-1],
                    prune_range=pruned_kd_range,
                    am_scale=am_scale,
                    lm_scale=lm_scale,
                    use_teacher_ctc_alignment=False,
                    use_efficient=use_efficient,
                    use_time_compression=False,
                    compression_threshold=0.0,
                    use_beam_search=False,
                    use_beam_search_alignment=False,
                    beam_search_alignment=None,
                    use_grad=False,
                )

                teacher_sampling_ranges.append(sampling_ret[0])
                teacher_sampling_logits.append(sampling_ret[1])

            for i in range(sq_sampling_num):
                sampling_logits = self.get_logits(
                    encoder_out=encoder_out,
                    encoder_out_lens=x_lens,
                    y=sampling_y[i],
                    prune_range=pruned_kd_range,
                    am_scale=am_scale,
                    lm_scale=lm_scale,
                    ranges=teacher_sampling_ranges[i],
                    use_grad=True,
                )

                student_sampling_logits = sampling_logits

                student_sampling = F.log_softmax(student_sampling_logits, dim=-1)
                teacher_sampling = F.softmax(teacher_sampling_logits[i], dim=-1)

                # T-axis direction masking
                max_len = student_sampling_logits.size(1)
                mask = torch.arange(max_len, device=student_sampling_logits.device).expand(student_sampling_logits.size(0), max_len) < x_lens.unsqueeze(1)
                mask = mask.unsqueeze(-1).unsqueeze(-1)

                student_sampling = student_sampling * mask.float()
                teacher_sampling = teacher_sampling * mask.float()

                sampling_loss += self.kd_criterion(student_sampling, teacher_sampling)
            # getting average for the sampling loss
            sampling_loss /= sq_sampling_num

        if use_ctc:
            ctc_out = self.ctc_layer(encoder_out)
            ctc_out = F.log_softmax(ctc_out, dim=-1).transpose(0, 1)
            ctc_loss = self.ctc_criterion(ctc_out, y_padded, x_lens, y_lens)

        ret = dict()
        ret["simple_loss"] = simple_loss
        ret["pruned_loss"] = pruned_loss
        if use_kd:
            ret["kd_loss"] = kd_loss
        else:
            ret["kd_loss"] = torch.tensor(0.0)

        if use_ctc:
            ret["ctc_loss"] = ctc_loss
        else:
            ret["ctc_loss"] = torch.tensor(0.0)

        if use_sq_sampling:
            ret["sampling_loss"] = sampling_loss
        else:
            ret["sampling_loss"] = torch.tensor(0.0)

        return ret

    def forced_alignment(
            self,
            x: torch.Tensor,
            x_lens: torch.Tensor,
            y: k2.RaggedTensor,
            ctc_prune_range: int = 1,
            use_ctc_layer: bool = True,
    ) -> torch.Tensor:
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

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
        """
        boundary = torch.zeros((x.size(0), 4), dtype=torch.int64, device=x.device)
        boundary[:, 2] = y_lens
        boundary[:, 3] = x_lens
        """

        if use_ctc_layer:
            logits = self.ctc_layer(encoder_out)
        else:
            logits = encoder_out
        logits = F.log_softmax(logits, dim=-1)
        emission = logits

        ctc_range = torch.zeros(emission.shape[0], emission.shape[1], dtype=torch.int32, device=emission.device)
        with torch.no_grad():
            for i in range(0, emission.shape[0]):
                trellis = get_trellis(emission[i].cpu(), y.tolist()[i], self.decoder.blank_id)
                path = backtrack(trellis, emission[i].cpu(), y.tolist()[i], self.decoder.blank_id)
                fpath, mpath = get_alignment(path, emission[i].cpu(), y.tolist()[i], self.decoder.blank_id)
                """
                print(f"{fpath=}")
                print(f"{mpath=}")
                print(f"{fpath=}")
                print(f"{len(fpath)=}")
                print(f"{encoder_out_lens[i]=}")
                """
                ctc_range[i] = torch.tensor(fpath, dtype=torch.int32, device=emission.device)
                ctc_range[i, x_lens[i]:] = self.decoder.blank_id

        ctc_range = ctc_range.unsqueeze(-1).to(torch.int64)
        # am_pruned : [B, T, prune_range, encoder_dim]
        # lm_pruned : [B, T, prune_range, decoder_dim]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ctc_range,
        )

        # logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        ctc_range.requires_grad_(False)
        logits = logits.detach_()

        return ctc_range, logits

    def get_ranges_and_logits(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        use_teacher_ctc_alignment: bool = False,
        use_efficient: bool = False,
        use_time_compression: bool = False,
        compression_threshold: float = 0.95,
        use_beam_search: bool = False,
        use_beam_search_alignment: bool = False,
        beam_search_alignment: torch.Tensor = None,
        use_grad: bool = False,
    ) -> torch.Tensor:
        """
        this function works in teacher model when use_pkd is True
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
            The weight for am probs
          lm_scale:
            The weight for lm probs
          use_teacher_ctc_alignment:
            If True, use teacher ctc alignment to get the pruned range
          use_efficient:
            If True, compress the logits for RNN-T from K-dims to 3-dims
            The 3-dims consists of Prob of y, Prob. of blank, and Prob. of others
            The Prob. of others can be calculated as 1 - Prob. of y - Prob. of blank
          use_time_compression:
            If True, compress the logits in time-axis by ignoring the t-th frame
            when the probability of blank is higher than a threshold
          use_beam_search:
            If True, use beam search alignment from teacher instead of ctc alignment
          beam_size_alignment:
            The alignment from beam search

        Returns:
          Return Tensor (the pruned range, logits, compressed length)

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

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
        am = self.simple_am_proj(encoder_out)

        #print("GARBAGE " * 10)
        #print(f"{use_teacher_ctc_alignment=}")
        if use_teacher_ctc_alignment:
            assert self.ctc_layer is not None
            ctc_logits = self.ctc_layer(encoder_out)
            ctc_logits = F.log_softmax(ctc_logits, dim=-1)
            emission = ctc_logits

            ctc_ranges = torch.zeros(
                emission.shape[0], emission.shape[1],
                dtype=torch.int32, device=emission.device
            )
            # get forced alignment results from CTC model
            with torch.no_grad():
                for i in range(0, emission.shape[0]):
                    trellis = get_trellis(emission[i].cpu(), y.tolist()[i], self.decoder.blank_id)
                    path = backtrack(trellis, emission[i].cpu(), y.tolist()[i], self.decoder.blank_id)
                    fpath, mpath = get_alignment(path, emission[i].cpu(), y.tolist()[i], self.decoder.blank_id)
                    ctc_ranges[i] = torch.tensor(fpath, dtype=torch.int32, device=emission.device)
                    #print(f"{ctc_ranges[i]=}")
                    # if the first position is not 0, then we need to
                    # shift one step to the right and prepend 0 to the first position
                    if ctc_ranges[i, 0] != 0:
                        ctc_ranges[i] = torch.roll(ctc_ranges[i], shifts=1, dims=-1)
                        ctc_ranges[i, 0] = 0
                    ctc_ranges[i, x_lens[i]:] = self.decoder.blank_id

            ctc_ranges = ctc_ranges.to(torch.int32)
            # something to do with ctc_range
            # ctc_range : from [B, T, 1] to [B, T, prune_range]
            ranges = torch.zeros((ctc_ranges.size(0), ctc_ranges.size(-1), prune_range),
                                dtype=torch.int64, device=ctc_ranges.device)

            idx = ctc_ranges + prune_range - 1 >= y_lens.unsqueeze(-1)
            # change the first idx to fit the boundary
            for i in range(0, idx.size(0)):
                ctc_ranges[i, idx[i]] = y_lens[i] - ( prune_range - 1 )
            # check if the ctc_ranges is out of boundary
            # the logic is a little bit weird, but it works
            for i in range(0, idx.size(0)):
                ctc_ranges[i][ctc_ranges[i] < 0] = 0
            ranges[:, :, 0] = ctc_ranges
            for i in range(1, prune_range):
                ranges[:, :, i] = ranges[:, :, i-1] + 1

            # padding beyound x_lens with max values
            idx = torch.argmax(ranges[:, :, 0], dim=-1)
            """
            if torch.any(idx == 0):
                #print full tensor in pytorch
                torch.set_printoptions(threshold=100000)
                print(f"{ctc_ranges[-1]=}")
                print(f"{ranges[-1]=}")
                print(f"{idx=}")
                print(f"{x_lens=}")
                print(f"{y_lens=}")
                print("HERE HERE")
                import sys
                #sys.exit(0)
            """
            padding_values = ranges[range(ranges.size(0)), idx, :]

            # TODO: optimization
            for i in range(0, idx.size(0)):
                ranges[i, idx[i]:] = padding_values[i]
            """
            torch.set_printoptions(threshold=100000)
            print(f"{ctc_ranges[-1]=}")
            print(f"{ranges[-1]=}")
            print(f"{idx=}")
            print(f"{x_lens=}")
            print(f"{y_lens=}")
            print("HERE HERE")
            import sys
            sys.exit(0)
            """
            #print(f"ranges: {ranges}")
        elif use_beam_search_alignment:
            # previous logic
            #elif use_beam_search and beam_search_alignment is not None:
            # beam_search_alignment [B, T]
            #torch.set_printoptions(threshold=100000)
            #print("before getting ranges")
            #print(f"{prune_range=}")
            #print(f"{beam_search_alignment[0]=}")
            #print(f"{beam_search_alignment[-1]=}")
            ranges = torch.zeros((beam_search_alignment.size(0), beam_search_alignment.size(-1), prune_range),
                                dtype=torch.int64, device=decoder_out.device)

            # to make alignments in the middle of ranges
            beam_search_alignment -= prune_range//2
            idx = beam_search_alignment + prune_range - 1 >= y_lens.unsqueeze(-1)
            # change the first idx to fit the boundary
            for i in range(0, idx.size(0)):
                beam_search_alignment[i, idx[i]] = y_lens[i] - ( prune_range - 1 )
            # check if the ctc_ranges is out of boundary
            # the logic is a little bit weird, but it works
            for i in range(0, idx.size(0)):
                beam_search_alignment[i][beam_search_alignment[i] < 0] = 0
            ranges[:, :, 0] = beam_search_alignment
            for i in range(1, prune_range):
                ranges[:, :, i] = ranges[:, :, i-1] + 1

            # padding beyound x_lens with max values
            idx = torch.argmax(ranges[:, :, 0], dim=-1)
            padding_values = ranges[range(ranges.size(0)), idx, :]
            for i in range(0, idx.size(0)):
                ranges[i, idx[i]:] = padding_values[i]
            #if torch.any(idx == 0):
                #print full tensor in pytorch

            #torch.set_printoptions(threshold=100000)
            #print("after getting ranges")
            #print(f"{beam_search_alignment[0]=}")
            #print(f"{beam_search_alignment[-1]=}")
            #print(f"{ranges[0]=}")
            #print(f"{ranges[-1]=}")
            #print(f"{idx=}")
            #print(f"{x_lens=}")
            #print(f"{y_lens=}")
            #print(f"{boundary=}")
            #print(f"{beam_search_alignment.shape=}")
            #print("HERE HERE")
            #import sys
            #sys.exit(0)

        else:
            with torch.no_grad():
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

        # logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        ranges.requires_grad_(False)
        if use_grad is False:
            logits = logits.detach_()

        masks = None

        ret = (ranges, logits)

        return ret

    def get_ranges_and_logits_with_encoder_out(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        use_teacher_ctc_alignment: bool = False,
        use_efficient: bool = False,
        use_time_compression: bool = False,
        compression_threshold: float = 0.95,
        use_beam_search: bool = False,
        use_beam_search_alignment: bool = False,
        beam_search_alignment: torch.Tensor = None,
        use_grad: bool = False,
    ) -> torch.Tensor:
        """
        this function works in teacher model when use_pkd is True
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
            The weight for am probs
          lm_scale:
            The weight for lm probs
          use_teacher_ctc_alignment:
            If True, use teacher ctc alignment to get the pruned range
          use_efficient:
            If True, compress the logits for RNN-T from K-dims to 3-dims
            The 3-dims consists of Prob of y, Prob. of blank, and Prob. of others
            The Prob. of others can be calculated as 1 - Prob. of y - Prob. of blank
          use_time_compression:
            If True, compress the logits in time-axis by ignoring the t-th frame
            when the probability of blank is higher than a threshold
          use_beam_search:
            If True, use beam search alignment from teacher instead of ctc alignment
          beam_size_alignment:
            The alignment from beam search

        Returns:
          Return Tensor (the pruned range, logits, compressed length)

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert y.num_axes == 2, y.num_axes
        assert torch.all(encoder_out_lens > 0)

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
        boundary = torch.zeros((encoder_out.size(0), 4), dtype=torch.int64, device=encoder_out.device)
        boundary[:, 2] = y_lens
        boundary[:, 3] = encoder_out_lens

        lm = self.simple_lm_proj(decoder_out)
        am = self.simple_am_proj(encoder_out)

        #print("GARBAGE " * 10)
        #print(f"{use_teacher_ctc_alignment=}")
        if use_teacher_ctc_alignment:
            assert self.ctc_layer is not None
            ctc_logits = self.ctc_layer(encoder_out)
            ctc_logits = F.log_softmax(ctc_logits, dim=-1)
            emission = ctc_logits

            ctc_ranges = torch.zeros(
                emission.shape[0], emission.shape[1],
                dtype=torch.int32, device=emission.device
            )
            # get forced alignment results from CTC model
            with torch.no_grad():
                for i in range(0, emission.shape[0]):
                    trellis = get_trellis(emission[i].cpu(), y.tolist()[i], self.decoder.blank_id)
                    path = backtrack(trellis, emission[i].cpu(), y.tolist()[i], self.decoder.blank_id)
                    fpath, mpath = get_alignment(path, emission[i].cpu(), y.tolist()[i], self.decoder.blank_id)
                    ctc_ranges[i] = torch.tensor(fpath, dtype=torch.int32, device=emission.device)
                    #print(f"{ctc_ranges[i]=}")
                    # if the first position is not 0, then we need to
                    # shift one step to the right and prepend 0 to the first position
                    if ctc_ranges[i, 0] != 0:
                        ctc_ranges[i] = torch.roll(ctc_ranges[i], shifts=1, dims=-1)
                        ctc_ranges[i, 0] = 0
                    ctc_ranges[i, encoder_out_lens[i]:] = self.decoder.blank_id

            ctc_ranges = ctc_ranges.to(torch.int32)
            # something to do with ctc_range
            # ctc_range : from [B, T, 1] to [B, T, prune_range]
            ranges = torch.zeros((ctc_ranges.size(0), ctc_ranges.size(-1), prune_range),
                                dtype=torch.int64, device=ctc_ranges.device)

            idx = ctc_ranges + prune_range - 1 >= y_lens.unsqueeze(-1)
            # change the first idx to fit the boundary
            for i in range(0, idx.size(0)):
                ctc_ranges[i, idx[i]] = y_lens[i] - ( prune_range - 1 )
            # check if the ctc_ranges is out of boundary
            # the logic is a little bit weird, but it works
            for i in range(0, idx.size(0)):
                ctc_ranges[i][ctc_ranges[i] < 0] = 0
            ranges[:, :, 0] = ctc_ranges
            for i in range(1, prune_range):
                ranges[:, :, i] = ranges[:, :, i-1] + 1

            # padding beyound x_lens with max values
            idx = torch.argmax(ranges[:, :, 0], dim=-1)
            padding_values = ranges[range(ranges.size(0)), idx, :]

            # TODO: optimization
            for i in range(0, idx.size(0)):
                ranges[i, idx[i]:] = padding_values[i]
            #print(f"ranges: {ranges}")
        elif use_beam_search_alignment:
            ranges = torch.zeros((beam_search_alignment.size(0), beam_search_alignment.size(-1), prune_range),
                                dtype=torch.int64, device=decoder_out.device)

            # to make alignments in the middle of ranges
            beam_search_alignment -= prune_range//2
            idx = beam_search_alignment + prune_range - 1 >= y_lens.unsqueeze(-1)
            # change the first idx to fit the boundary
            for i in range(0, idx.size(0)):
                beam_search_alignment[i, idx[i]] = y_lens[i] - ( prune_range - 1 )
            # check if the ctc_ranges is out of boundary
            # the logic is a little bit weird, but it works
            for i in range(0, idx.size(0)):
                beam_search_alignment[i][beam_search_alignment[i] < 0] = 0
            ranges[:, :, 0] = beam_search_alignment
            for i in range(1, prune_range):
                ranges[:, :, i] = ranges[:, :, i-1] + 1

            # padding beyound x_lens with max values
            idx = torch.argmax(ranges[:, :, 0], dim=-1)
            padding_values = ranges[range(ranges.size(0)), idx, :]
            for i in range(0, idx.size(0)):
                ranges[i, idx[i]:] = padding_values[i]

        else:
            with torch.no_grad():
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

        # logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        ranges.requires_grad_(False)
        if use_grad is False:
            logits = logits.detach_()

        ret = (ranges, logits)

        return ret

    def get_logits_with_encoder_out_and_ranges(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        y: k2.RaggedTensor,
        ranges: torch.Tensor,
        prune_range: int = 5,
        use_grad: bool = False,
    ) -> torch.Tensor:
        """
        this function works in teacher model when use_pkd is True
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          ranges:
            Pre-computed ranges for each utterance
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          use_grad:
            If True, the logits will have grad

        Returns:
          Return Tensor logits

        """
        assert y.num_axes == 2, y.num_axes
        assert torch.all(encoder_out_lens > 0)

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
        boundary = torch.zeros((encoder_out.size(0), 4), dtype=torch.int64, device=encoder_out.device)
        boundary[:, 2] = y_lens
        boundary[:, 3] = encoder_out_lens

        # am_pruned : [B, T, prune_range, encoder_dim]
        # lm_pruned : [B, T, prune_range, decoder_dim]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        # logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        ranges.requires_grad_(False)
        if use_grad is False:
            logits = logits.detach_()

        ret = logits

        return ret

    def get_encoder_out(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        use_grad: bool = False,
    ):
        """
        This function is used to retrieve the model's encoder output and
        the lengths of the encoder outputs.
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          use_grad:
            If False, the returned tensors will be detached from the computation
            graph. If True, the returned tensors will have gradients.

        Returns:
          Return Tensor (encoder_out, encoder_out_lens)

        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape

        assert x.size(0) == x_lens.size(0)

        encoder_out, encoder_out_lens = self.encoder(x, x_lens)
        assert torch.all(x_lens > 0)
        if use_grad is False:
            encoder_out = encoder_out.detach()
            x_lens = x_lens.detach()

        return encoder_out, encoder_out_lens

    def get_logits(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        ranges: torch.Tensor = None,
        use_grad: bool = False,
    ) -> torch.Tensor:
        """
        this function works in teacher model when use_pkd is True
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
            The weight for am probs
          lm_scale:
            The weight for lm probs
          ranges:
            The pre-calculated ranges for getting logits
          use_grad:
            If True, the logits will be calculated with grad, otherwise, the logits
            will be calculated without grad.

        Returns:
          Return Tensor (the pruned range, logits, compressed length)

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert y.num_axes == 2, y.num_axes
        assert torch.all(encoder_out_lens > 0)

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
        boundary = torch.zeros((encoder_out.size(0), 4), dtype=torch.int64, device=encoder_out.device)
        boundary[:, 2] = y_lens
        boundary[:, 3] = encoder_out_lens

        # am_pruned : [B, T, prune_range, encoder_dim]
        # lm_pruned : [B, T, prune_range, decoder_dim]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        # logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        ranges.requires_grad_(False)
        if use_grad is False:
            logits = logits.detach_()

        masks = None

        return logits

    def reset_simple_layer(self):
        self.simple_lm_proj.reset_parameters()
        self.simple_am_proj.reset_parameters()

    def set_ctc_loss(self, reduction="mean", zero_infinity=True, blank="0"):
        self.ctc_criterion = torch.nn.CTCLoss(
            blank=blank,
            reduction=reduction,
            zero_infinity=zero_infinity,
        )

    def get_enc_and_dec_ouput(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor
    ):
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0

        encoder_out, x_lens = self.encoder(x, x_lens)
        assert torch.all(x_lens > 0)

        # Now for the decoder, i.e., the prediction network
        # row_splits = y.shape.row_splits(1)
        # y_lens = row_splits[1:] - row_splits[:-1]

        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = self.decoder(sos_y_padded)

        return (encoder_out, decoder_out)

    def set_teacher_simple_layer(self, encoder_dim: int, decoder_dim: int, vocab_size: int, device: torch.device):
        self.teacher_simple_am_proj = nn.Linear(encoder_dim, vocab_size)
        self.teacher_simple_lm_proj = nn.Linear(decoder_dim, vocab_size)
        self.teacher_simple_am_proj.to(device)
        self.teacher_simple_lm_proj.to(device)

    def get_alphas(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        ranges: torch.Tensor = None,
        rnnt_type: str = "regular",
    ) -> torch.Tensor:
        """
        this function works in teacher model when use_pkd is True
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
          use_teacher_ctc_alignment:
            If True, use teacher ctc alignment to get the pruned range
          use_efficient:
            If True, compress the logits for RNN-T from K-dims to 3-dims
            The 3-dims consists of Prob of y, Prob. of blank, and Prob. of others
            The Prob. of others can be calculated as 1 - Prob. of y - Prob. of blank
          use_time_compression:
            If True, compress the logits in time-axis by ignoring the t-th frame
            when the probability of blank is higher than a threshold

        Returns:
          Return Tensor (the pruned range, logits, compressed length)

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0

        encoder_out, x_lens = self.encoder(x, x_lens)
        assert torch.all(x_lens > 0)
        assert ranges is not None

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

        # am_pruned : [B, T, prune_range, encoder_dim]
        # lm_pruned : [B, T, prune_range, decoder_dim]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        # logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        ranges.requires_grad_(False)

        px, py = k2.get_rnnt_logprobs_pruned(
            logits=logits,
            symbols=y_padded,
            ranges=ranges,
            termination_symbol=blank_id,
            boundary=boundary,
            rnnt_type=rnnt_type,
        )

        assert px.ndim == 3, px.shape
        B, S, T1 = px.shape
        T = py.shape[-1]
        assert px.shape[-1] in [T, T + 1], (px.shape, T)  # if T, then "modified".
        assert py.shape == (B, S + 1, T), (py.shape, B, S, T)
        assert px.dtype == py.dtype, (px.dtype, py.dtype)
        if boundary is not None:
            assert boundary.dtype == torch.int64, boundary.dtype
            assert boundary.shape == (B, 4), (boundary.shape, B)
            for s_begin, t_begin, s_end, t_end in boundary.tolist():
                assert 0 <= s_begin <= s_end <= S, (s_begin, s_end, S)
                assert 0 <= t_begin <= t_end <= T, (t_begin, t_end, T)

        px, py = px.contiguous(), py.contiguous()
        _, alphas = k2.mutual_information_recursion(px, py, boundary=boundary,
                                                    return_grad=False, return_alphas=True)

        return alphas


def get_trellis(emission, tokens, blank_id=0):
     num_frame = emission.size(0)
     num_tokens = len(tokens)

     # Trellis has extra diemsions for both time axis and tokens.
     # The extra dim for tokens represents <SoS> (start-of-sentence)
     # The extra dim for time axis is for simplification of the code.
     trellis = torch.empty((num_frame + 1, num_tokens + 1))
     trellis[0, 0] = 0
     trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
     trellis[0, -num_tokens:] = -float("inf")
     trellis[-num_tokens:, 0] = float("inf")

     for t in range(num_frame):
         trellis[t + 1, 1:] = torch.maximum(
             # Score for staying at the same token
             trellis[t, 1:] + emission[t, blank_id],
             # Score for changing to the next token
             trellis[t, :-1] + emission[t, tokens],
         )
     return trellis

from dataclasses import dataclass

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    # print(f"{t_start=}")
    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align")
    return path[::-1]

def get_prune_ranges_from_ctc_alignment(ctc_ranges, prune_range):
    # TODO
    # should be more shopisticated in the future
    # from ctc_ranges: [B, T, 1]
    # to ctc_ranges: [B, T, prune_range]
    B = ctc_ranges.size(0)
    T = ctc_ranges.size(1)
    R = prune_range
    ranges = torch.zeros((B, T, R), dtype=torch.int64, device=ctc_ranges.device)
    # filtering ctc_ranges
    ranges[:, :, 0] = ctc_ranges

    pass

def compress_time(emission, emission_lens, mask):
    B = emission.size(0)
    T = emission.size(1)
    dim = emission.size(-1)
    assert mask.ndim == 2
    assert mask.size(0) == B
    max_value = float('-inf')
    length = list()
    for b in range(B):
        max_value = max(max_value, mask[b, :emission_lens[b]].sum().item())
        length.append(mask[b, :emission_lens[b]].sum().item())

    ret = torch.zeros((B, max_value, dim), dtype=emission.dtype, device=emission.device)
    #print(f"{ret.shape=}")
    #print(f"{length=}")
    for b in range(B):
        #print(f"{emission[b].shape=}")
        ret[b, :length[b], :] = torch.masked_select(emission[b], mask[b].unsqueeze(-1)).view(-1, dim)[:length[b], :]
        #print(f"{ret[b].shape=}")

    ret = torch.where(torch.eq(ret, 0.0), 1e-10, ret)
    #print(f"{ret.shape=}")
    #print(f"{ret[0]=}")
    new_lens = torch.tensor(length, dtype=torch.int64, device=emission.device)
    return ret, new_lens

def get_alignment(paths, emission, tokens, blank_id=0):
    """
    # paths is a list of Point
    # emission is a tensor of shape [T, V]
    # tokens is a list of int
    # blank_id is an int
    """

    sos_idx = paths[0].time_index
    eos_idx = paths[-1].time_index
    last_token = paths[-1].token_index

    # make full length on time axis
    full_path = []
    for i in range(0, sos_idx):
        full_path.append(-1 + 1)
    for i in paths:
        full_path.append(i.token_index + 1)
    for i in range(eos_idx+1, emission.size(0)):
        full_path.append(last_token + 1)
        #full_path.append(-1 + 1)

    # modified path contains
    modified_path = []
    prev = 0
    for i in range(0, len(full_path)):
        if full_path[i] == blank_id:
            modified_path.append(full_path[i])
            prev = blank_id
        elif full_path[i] == -1:
            modified_path.append(blank_id)
            prev = -1
        elif full_path[i] == prev:
            modified_path.append(blank_id)
            prev = full_path[i]
        else:
            modified_path.append(full_path[i])
            prev = full_path[i]

    return full_path, modified_path

# This main is for testing purpose
if __name__ == "__main__":

    from train import get_params, get_transducer_model
    import k2
    import torch.nn as nn
    import torch.nn.functional as F

    params = get_params()
    params.vocab_size = 500
    params.blank_id = 0
    params.context_size = 2
    params.num_encoder_layers = "2,4,3,2,4"
    params.feedforward_dims = "1024,1024,2048,2048,1024"
    params.nhead = "8,8,8,8,8"
    params.encoder_dims = "384,384,384,384,384"
    params.attention_dims = "192,192,192,192,192"
    params.encoder_unmasked_dims = "256,256,256,256,256"
    params.zipformer_downsampling_factors = "1,2,4,8,2"
    params.cnn_module_kernels = "31,31,31,31,31"
    params.decoder_dim = 512
    params.joiner_dim = 512
    print(f"{params=}")
    device = torch.device("cuda", 0)
    #device = torch.device("cpu")
    print(f"{device=}")

    model = get_transducer_model(params)
    model.to(device)
    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of model parameters: {num_param}")

    N = 1
    T = 200
    C = 80
    x = torch.randn(N, T, C).to(device)
    x_lens = torch.tensor([T], dtype=torch.int32).to(device)
    y = torch.randint(low=0, high=C-1, size=(N, 10), dtype=torch.int32).to(device)
    y = k2.RaggedTensor(y).to(device)
    print(f"{y=}")
    model.eval()
    ranges, teacher_logits = model.get_ranges_and_logits(x, x_lens, y, prune_range=5)
    alphas = model.get_alphas(x, x_lens, y, prune_range=5, ranges=ranges)
    print(f"{alphas=}")
    print(f"{alphas.detach()=}")

    print(f"{ranges=}")
    print(f"{ranges.shape=}")
    print(f"{teacher_logits.shape=}")
    print(f"{model.simple_am_proj=}")
    # for student model
    params2 = get_params()
    params2.vocab_size = 500
    params2.blank_id = 0
    params2.context_size = 2
    params2.num_encoder_layers = "1,2,2,1,2"
    params2.feedforward_dims = "256,256,512,512,256"
    params2.nhead = "4,4,4,4,4"
    params2.encoder_dims = "256,256,256,256,256"
    params2.attention_dims = "192,192,192,192,192"
    params2.encoder_unmasked_dims = "256,256,256,256,256"
    params2.zipformer_downsampling_factors = "1,2,4,8,2"
    params2.cnn_module_kernels = "31,31,31,31,31"
    params2.decoder_dim = 512
    params2.joiner_dim = 512
    print(f"{params2=}")
    device = torch.device("cuda", 0)
    #device = torch.device("cpu")
    print(f"{device=}")

    smodel = get_transducer_model(params2)
    smodel.to(device)
    num_param = sum([p.numel() for p in smodel.parameters()])
    print(f"Number of student model parameters: {num_param}")

    smodel.train()
    smodel.set_ctc_loss(blank=0, reduction="mean", zero_infinity=True)

    ret = smodel(
        x,
        x_lens,
        y,
        prune_range=5,
        am_scale=0.5,
        lm_scale=0.5,
        use_pkd=True,
        teacher_ranges=ranges,
        teacher_logits=teacher_logits,
        use_ctc=True,)

    print(f"{ret['simple_loss']=}")
    print(f"{ret['pruned_loss']=}")
    print(f"{ret['pkd_loss']=}")
    print(f"{ret['ctc_loss']=}")

    print("*"*100)
    print("Test ctc alignment")

    params = get_params()
    params.vocab_size = 500
    params.blank_id = 0
    params.context_size = 2
    params.num_encoder_layers = "2,4,3,2,4"
    params.feedforward_dims = "1024,1024,2048,2048,1024"
    params.nhead = "8,8,8,8,8"
    params.encoder_dims = "384,384,384,384,384"
    params.attention_dims = "192,192,192,192,192"
    params.encoder_unmasked_dims = "256,256,256,256,256"
    params.zipformer_downsampling_factors = "1,2,4,8,2"
    params.cnn_module_kernels = "31,31,31,31,31"
    params.decoder_dim = 512
    params.joiner_dim = 512
    params.use_ctc = True
    print(f"{params=}")
    device = torch.device("cuda", 0)
    print(f"{device=}")

    model = get_transducer_model(params)
    model.to(device)
    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of model parameters: {num_param}")

    from icefall.checkpoint import (
        load_checkpoint,
    )

    import torchaudio
    import torch
    # SPEECH_FILE = torchaudio.utils.download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
    #SPEECH_FILE = "./1089-134686-0000.flac"
    SPEECH_FILE = "./4640-19187-0011.flac"
    #  HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS AND BRUISED POTATOES AND FAT MUTTON PIECES TO BE LADLED OUT IN THICK PEPPERED FLOUR FATTENED SAUCE

    waveform, _ = torchaudio.load(SPEECH_FILE)
    # print(f"{waveform.shape=}")

    # for feature extraction
    import librosa
    import numpy as np
    from lhotse.utils import (
        EPSILON,
        LOG_EPSILON,
        Seconds,
        compute_num_frames,
        is_module_available,
    )
    import k2
    x_stft = librosa.stft(
        waveform.squeeze().numpy(),
        n_fft=512,
        hop_length=160,
        win_length=400,
        window="hann",
        center=True,
        pad_mode="reflect",
    )

    from asr_datamodule import LibriSpeechAsrDataModule
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.return_cuts = True
    args.manifest_dir = Path("data/fbank")
    args.on_the_fly_feats = False
    args.max_duration = 50
    args.num_workers = 1
    args.input_strategy = "PrecomputedFeatures"
    librispeech = LibriSpeechAsrDataModule(args)

    test_clean_cuts = librispeech.test_clean_cuts()
    test_clean_dl = librispeech.test_dataloaders(test_clean_cuts)

    """
    spc = np.abs(x_stft).T
    print(f"{spc=}")
    sampling_rate = 16000
    fmin = 80
    fmax = 7600
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=512, n_mels=80, fmin=fmin, fmax=fmax)
    eps = np.finfo(np.float32).eps
    feats = np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))
    """
    load_checkpoint("./teacher_model/epoch-30.pt", model)
    #model.set_ctc_loss(blank=0, reduction="mean", zero_infinity=True)

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load("../data/lang_bpe_500/bpe.model")
    for idx, batch in enumerate(test_clean_dl):
        #print(f"{feature.shape=}")
        supervisions = batch["supervisions"]
        flag = False
        for ii in supervisions["text"]:
            if "VENICE" in ii:
                print(ii)
                j = 0
                for tt in batch["supervisions"]["text"]:
                    print(f"{j}: {tt}")
                    j = j + 1
                flag = True
            else:
                continue
        if flag is False:
            continue
        #print(f"{supervisions=}")
        iidx = 12 # VENICE
        feature = batch["inputs"][iidx].unsqueeze(0).to(device)
        #feature_lens = supervisions["num_frames"][iidx].unsqueeze(0).to(device)
        feature_lens = torch.tensor([feature.shape[1]]).to(device)
        print(f"{feature=}")
        print(f"{feature_lens=}")
        print(f"{feature.shape=}")
        print(f"{feature_lens.shape=}")
        encoder_out, encoder_out_lens = model.encoder(x=feature, x_lens=feature_lens)
        logits = model.ctc_layer(encoder_out)
        predids = torch.argmax(logits, dim=-1)
        tmp_predicted_ids = torch.unique_consecutive(predids[0, :])
        tmp_predicted_ids = tmp_predicted_ids[tmp_predicted_ids != 0].cpu().detach()
        print(f"{tmp_predicted_ids=}")
        hyp = sp.decode(tmp_predicted_ids.tolist())
        print(f"{hyp=}")

        emission = torch.log_softmax(logits, dim=-1)
        y = sp.encode([supervisions["text"][iidx]], out_type=int)
        y = k2.RaggedTensor(y)
        print(f"{y=}")
        print(f"{len(y.tolist()[0])=}")
        #sos_y = add_sos(y, sos_id=params.blank_id)
        #sos_y_padded = sos_y.pad(mode="constant", padding_value=params.blank_id)
        #y = sos_y_padded
        y = y.tolist()[0]
        #print(f"{y=}")
        #print(f"{len(y)=}")
        trellis = get_trellis(emission[0].cpu(), y, params.blank_id)
        print(f"{encoder_out_lens=}")
        print(f"{predids=}")
        print(f"{predids.shape=}")
        print(f"{supervisions['text'][iidx]=}")
        path = backtrack(trellis, emission[0], y)
        for p in path:
            print(p)
        fpath, mpath = get_alignment(path, emission[0], y, params.blank_id)
        print(f"{fpath=}")
        print(f"{len(fpath)=}")
        print(f"T: {len(fpath)}, U: {len(y)}")
        print(f"{mpath=}")
        print(f"{len(mpath)=}")
        break
    #x = torch.from_numpy(feats).unsqueeze(0)
    #x_lens = torch.tensor([x.shape[1]])

    # print(f"{x.shape=}")
    # tmp_predicted_ids = torch.unique_consecutive(predicted_ids[i, :])
    """
    print("*" * 100)
    print("Test compress_time_axis")
    for idx, batch in enumerate(test_clean_dl):
        feature = batch["inputs"].to(device)
        supervisions = batch["supervisions"]
        feature_lens = supervisions["num_frames"].to(device)
        encoder_out, encoder_out_lens = model.encoder(x=feature, x_lens=feature_lens)
        y = sp.encode(supervisions["text"], out_type=int)
        y = k2.RaggedTensor(y).to(device)
        ranges, logits = model.get_ranges_and_logits(x=feature, x_lens=feature_lens, y=y, prune_range=5)
        print(f"before compressing, {ranges.shape=}")
        threshold = 0.95
        ct_ranges, ct_logits, ct_masks = model.get_ranges_and_logits_with_time_compression(x=feature, x_lens=feature_lens, y=y, prune_range=5, threshold=threshold)
        print(f"after compressing, {ct_ranges.shape=}")
        ret = smodel(
            x=feature,
            x_lens=feature_lens,
            y=y,
            prune_range=5,
            am_scale=0.5,
            lm_scale=0.5,
            use_pkd=True,
            teacher_ranges=ranges,
            teacher_logits=logits,
            use_ctc=True,
            use_teacher_simple_proj=False,
            teacher_model=model,
            use_time_compression=True,
            teacher_compressed_ranges=ct_ranges,
            teacher_compressed_logits=ct_logits,
            teacher_compressed_masks=ct_masks,
            use_efficient=True,
        )
        break
    """
    print("*" * 100)
    print("Test using alphas")
    for idx, batch in enumerate(test_clean_dl):
        feature = batch["inputs"].to(device)
        #feature = feature[:3]
        supervisions = batch["supervisions"]
        feature_lens = supervisions["num_frames"].to(device)
        #feature_lens = feature_lens[:3]
        print(f"{feature.shape=}")
        print(f"{feature_lens.shape=}")
        encoder_out, encoder_out_lens = model.encoder(x=feature, x_lens=feature_lens)
        #encoder_out = encoder_out[:3]
        #encoder_out_lens = encoder_out_lens[:3]
        y = sp.encode(supervisions["text"], out_type=int)
        print(f"{y=}")
        print(f"{len(y)=}")
        #y = y[:3]
        y = k2.RaggedTensor(y).to(device)
        print(f"{feature.shape=}")
        print(f"{feature_lens.shape=}")
        print(f"{y.shape=}")
        ranges, logits = model.get_ranges_and_logits(x=feature, x_lens=feature_lens, y=y, prune_range=5)
        teacher_alphas = model.get_alphas(x=feature, x_lens=feature_lens, y=y, prune_range=5, ranges=ranges)
        #print(f"{teacher_alphas.shape=}")
        ret = smodel(
            x=feature,
            x_lens=feature_lens,
            y=y,
            prune_range=5,
            am_scale=0.5,
            lm_scale=0.5,
            use_pkd=True,
            teacher_ranges=ranges,
            teacher_logits=logits,
            use_ctc=False,
            use_teacher_simple_proj=False,
            teacher_model=model,
            use_time_compression=False,
            use_efficient=False,
            use_alphas=True,
            teacher_alphas=teacher_alphas,
        )
        break

    print("#" * 100)
    print("To check ragens and logits")
    for idx, batch in enumerate(test_clean_dl):
        feature = batch["inputs"].to(device)
        supervisions = batch["supervisions"]
        feature_lens = supervisions["num_frames"].to(device)
        y = sp.encode(supervisions["text"], out_type=int)
        y = k2.RaggedTensor(y).to(device)
        ranges, logits = model.get_ranges_and_logits(x=feature, x_lens=feature_lens, y=y, prune_range=5)
        print(f"{ranges=}")
        print(f"{logits=}")
        break

    print("#" * 100)
    print("To get hypotheses from modified_beam_search")

    from beam_search import (
        modified_beam_search,
        modified_beam_search_for_kd_nbest,
        Hypothesis,
        HypothesisList
    )
    for idx, batch in enumerate(test_clean_dl):
        feature = batch["inputs"].to(device)
        supervisions = batch["supervisions"]
        feature_lens = supervisions["num_frames"].to(device)
        feature = feature[:]
        feature_lens = feature_lens[:]
        print(f"{feature.shape=}")
        print(f"{feature_lens.shape=}")
        beam = 4
        #encoder_out, encoder_out_lens = model.encoder(x=feature, x_lens=feature_lens)
        hyp_tokens, hyp_timestamps = modified_beam_search_for_kd_nbest(
            model=model,
            x=feature,
            x_lens=feature_lens,
            beam=beam,
            temperature=1.0,
        )
        for i in range(len(hyp_tokens)):
            print(f"{hyp_tokens[i]=}")
            print(f"{hyp_timestamps[i]=}")
            print(f"{sp.decode(hyp_tokens[i])=}")

        hyp_alignment = torch.zeros(beam, encoder_out_lens[-1], dtype=torch.int)

        break


