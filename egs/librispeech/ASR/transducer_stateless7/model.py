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
import torchaudio
from encoder_interface import EncoderInterface
from scaling import penalize_abs_values_gt

from icefall.utils import add_sos
from typing import Union

import numpy as np


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

        self.simple_am_proj = nn.Linear(encoder_dim, vocab_size)
        self.simple_lm_proj = nn.Linear(decoder_dim, vocab_size)

        self.kd_criterion = nn.KLDivLoss(reduction="batchmean")
        self.ctc_layer = nn.Linear(encoder_dim, vocab_size)
        self.ctc_criterion = None

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        nbest_y: Union[list, k2.RaggedTensor] = None,
        use_kd: bool = False,
        teacher_logits: torch.Tensor = None,
        nbest_teacher_logits: Union[list, torch.tensor] = None,
        use_ctc: bool = False,
        teacher_model: nn.Module = None,
        use_efficient: bool = False,
        use_1best: bool = False,
        use_nbest: bool = False,
        use_pruned: bool = False,
        pseudo_y_alignment: torch.Tensor = None,
        nbest_pseudo_y_alignment: Union[list, torch.tensor] = None,
        topk: int = 1,
        use_topk_shuff: bool = False,
        use_sq_sampling: bool = False,
        sampling_y: Union[list, k2.RaggedTensor] = None,
        teacher_sampling_logits: Union[list, torch.tensor] = None,
        sq_sampling_num: int = 1,
        pruned_range: int = 5,
        use_sq_simple_loss_range: bool = False,
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
        Returns:
          Return the transducer loss.

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

        org_x_lens = x_lens.clone()
        org_encoder_out, x_lens = self.encoder(x, x_lens)
        assert torch.all(x_lens > 0)

        # Now for the decoder, i.e., the prediction network
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)
        sos_y_padded = sos_y_padded.to(torch.int64)

        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = self.decoder(sos_y_padded)

        encoder_out = self.joiner.encoder_proj(org_encoder_out)
        decoder_out = self.joiner.decoder_proj(decoder_out)

        #print(f"{encoder_out.shape=}")
        #print(f"{decoder_out.shape=}")
        logits = self.joiner(
            encoder_out=encoder_out,
            decoder_out=decoder_out,
            project_input=False,
        )

        # Note: y does not start with SOS
        # y_padded : [B, S]
        y_padded = y.pad(mode="constant", padding_value=0)

        assert hasattr(torchaudio.functional, "rnnt_loss"), (
            f"Current torchaudio version: {torchaudio.__version__}\n"
            "Please install a version >= 0.10.0"
        )

        loss = torchaudio.functional.rnnt_loss(
            logits=logits,
            targets=y_padded,
            logit_lengths=x_lens,
            target_lengths=y_lens,
            blank=blank_id,
            reduction="sum",
        )

        if use_kd:
            if use_efficient is False and use_1best is False and use_nbest is False and use_pruned is False:
                student = F.log_softmax(logits, dim=-1)
                teacher = F.softmax(teacher_logits, dim=-1)
                # T-axis direction masking
                max_len = logits.size(1)
                mask = torch.arange(max_len, device=logits.device).expand(logits.size(0), max_len) < x_lens.unsqueeze(1)
                mask = mask.unsqueeze(-1).unsqueeze(-1)
                student = student * mask.float()
                teacher = teacher * mask.float()

            # the case when use_efficient is True and use_1best is True
            # is not supported yet.
            assert not (use_efficient and use_1best)
            assert not (use_1best and use_pruned)
            assert not (use_efficient and use_pruned)

            if use_efficient:
                student = F.log_softmax(logits, dim=-1)
                teacher = F.softmax(teacher_logits, dim=-1)
                tensor_y_padded = y_padded.to(torch.int64)
                indices = tensor_y_padded.unsqueeze(1).unsqueeze(-1)
                logits = torch.softmax(logits, dim=-1)
                teacher_logits = torch.softmax(teacher_logits, dim=-1)
                # py
                py_logits = torch.gather(logits, -1, indices.expand(-1, logits.size(1), -1, -1))
                teacher_py_logits = torch.gather(teacher_logits, -1, indices.expand(-1, logits.size(1), -1, -1))
                # blank
                blank_indices = torch.full_like(indices, blank_id)
                blank_logits = torch.gather(logits, -1, blank_indices.expand(-1, logits.size(1), -1, -1))
                teacher_blank_logits = torch.gather(teacher_logits, -1, blank_indices.expand(-1, logits.size(1), -1, -1))
                # remainder
                rem_logits = torch.ones(py_logits.shape, device=logits.device) - py_logits - blank_logits
                teacher_rem_logits = torch.ones(teacher_py_logits.shape, device=teacher_logits.device) - teacher_py_logits - teacher_blank_logits

                # concatenate
                eps = 1e-10
                student = torch.cat([py_logits, blank_logits, rem_logits], dim=-1)
                student = torch.clamp(student, eps, 1.0)
                student = student.log()
                teacher_rem_logits[torch.where(teacher_rem_logits<0)] = eps
                teacher = torch.cat([teacher_py_logits, teacher_blank_logits, teacher_rem_logits], dim=-1)

                # T-axis direction masking
                max_len = logits.size(1)
                mask = torch.arange(max_len, device=logits.device).expand(logits.size(0), max_len) < x_lens.unsqueeze(1)
                mask = mask.unsqueeze(-1).unsqueeze(-1)
                student = student * mask.float()
                teacher = teacher * mask.float()

                # U-axis direction masking
                max_len = torch.max(y_lens)
                mask = torch.arange(max_len, device=logits.device).expand(logits.size(0), logits.size(1), max_len) < y_lens.unsqueeze(-1).unsqueeze(-1)
                mask = mask.unsqueeze(-1)
                student = student * mask.float()
                teacher = teacher * mask.float()

                use_debug = False
                if use_debug:
                    print(f"{logits.shape=}")
                    print(f"{teacher_logits.shape=}")
                    print(f"{tensor_y_padded.shape=}")
                    print(f"{tensor_y_padded=}")
                    print(f"{py_logits.shape=}")
                    print(f"{teacher_py_logits.shape=}")
                    print(f"{x_lens=}")
                    print(f"{y_lens=}")
                    print(f"{student[-1,0,:,:]=}")
                    print(f"{teacher[-1,0,:,:]=}")
                    print(f"{torch.where(teacher<0)=}")
                    print(f"{teacher[0, 344, 74, 2]=}")
                    print(f"{teacher[0, 344, 74, 2].log()=}")
                    import sys
                    sys.exit(0)

            if use_1best:
                # getting the 1-best pash
                # pseudo_y: [B, U]
                # pseudo_y_alignment: [B, S, U]
                #print(f"{logits.shape=}")
                #print(f"{pseudo_y=}")
                #print(f"{pseudo_y_alignment=}")

                # memory explosion
                #teacher_logits = teacher_model.get_logits(x, org_x_lens, pseudo_y, use_grad=False)
                #student_logits = self.get_logits(x, org_x_lens, pseudo_y, use_grad=True)
                #student = F.log_softmax(student_logits, dim=-1)
                #teacher = F.softmax(teacher_logits, dim=-1)
                student = F.log_softmax(logits, dim=-1)
                teacher = F.softmax(teacher_logits, dim=-1)
                idx = pseudo_y_alignment.unsqueeze(-1).expand(-1, -1, logits.shape[-1]).unsqueeze(2)
                idx = idx.to(torch.int64)
                teacher = torch.gather(teacher, 2, idx)
                student = torch.gather(student, 2, idx)
                max_len = logits.size(1)
                mask = torch.arange(max_len, device=logits.device).expand(logits.size(0), max_len) < x_lens.unsqueeze(1)
                mask = mask.unsqueeze(-1).unsqueeze(-1)
                student = student * mask.float()
                teacher = teacher * mask.float()
                #print(f"{student[-1,-1,:,:]=}")
                #print(f"{teacher[-1,-1,:,:]=}")

            if use_nbest and not use_topk_shuff:
                nbest_teacher = None
                nbest_student = None
                for n in range(0, topk):
                    n_logits = self.get_logits(
                        encoder_out=org_encoder_out,
                        encoder_out_lens=x_lens,
                        y=nbest_y[n],
                        use_grad=True,
                    )
                    student = F.log_softmax(n_logits, dim=-1)
                    teacher = F.softmax(nbest_teacher_logits[n], dim=-1)
                    idx = nbest_pseudo_y_alignment[n].unsqueeze(-1).expand(-1, -1, logits.shape[-1]).unsqueeze(2)
                    idx = idx.to(torch.int64)
                    teacher = torch.gather(teacher, 2, idx)
                    student = torch.gather(student, 2, idx)
                    max_len = logits.size(1)
                    mask = torch.arange(max_len, device=logits.device).expand(logits.size(0), max_len) < x_lens.unsqueeze(1)
                    mask = mask.unsqueeze(-1).unsqueeze(-1)
                    student = student * mask.float()
                    teacher = teacher * mask.float()

                    # concatenating
                    nbest_teacher = torch.cat([nbest_teacher, teacher], dim=2) if nbest_teacher is not None else teacher
                    nbest_student = torch.cat([nbest_student, student], dim=2) if nbest_student is not None else student
                teacher = nbest_teacher
                student = nbest_student

            if use_nbest and use_topk_shuff:
                student = F.log_softmax(logits, dim=-1)
                teacher = F.softmax(teacher_logits, dim=-1)
                idx = pseudo_y_alignment.unsqueeze(-1).expand(-1, -1, logits.shape[-1]).unsqueeze(2)
                idx = idx.to(torch.int64)
                teacher = torch.gather(teacher, 2, idx)
                student = torch.gather(student, 2, idx)
                max_len = logits.size(1)
                mask = torch.arange(max_len, device=logits.device).expand(logits.size(0), max_len) < x_lens.unsqueeze(1)
                mask = mask.unsqueeze(-1).unsqueeze(-1)
                student = student * mask.float()
                teacher = teacher * mask.float()

            if use_pruned:
                student = F.log_softmax(logits, dim=-1)
                teacher = F.softmax(teacher_logits, dim=-1)
                # getting the pruned path

                def create_increasing_sublists_torch(original_tensor, length=5):
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

                # memory explosion
                max_len = torch.max(y_lens)
                if pruned_range > max_len:
                    pruned_range = max_len
                idx = create_increasing_sublists_torch(pseudo_y_alignment, pruned_range)
                batch_idx = torch.arange(logits.size(0)).view(-1, 1, 1)
                batch_idx = batch_idx.expand(-1, idx.size(-2), idx.size(-1))
                teacher = teacher[batch_idx, torch.arange(logits.size(1)).view(1, -1, 1), idx]
                student = student[batch_idx, torch.arange(logits.size(1)).view(1, -1, 1), idx]
                max_len = logits.size(1)
                mask = torch.arange(max_len, device=logits.device).expand(logits.size(0), max_len) < x_lens.unsqueeze(1)
                mask = mask.unsqueeze(-1).unsqueeze(-1)
                student = student * mask.float()
                teacher = teacher * mask.float()

            kd_loss = self.kd_criterion(student, teacher)

        if use_sq_sampling:
            sampling_loss = torch.tensor(0.0).to(logits.device)
            for i in range(sq_sampling_num):
                # original working code
                sampling_logits = self.get_logits(
                    encoder_out=org_encoder_out,
                    encoder_out_lens=x_lens,
                    y=sampling_y[i],
                    use_grad=True,
                )

                if use_efficient:
                    logits_dict = dict()
                    logits_dict["student"] = sampling_logits
                    logits_dict["teacher"] = teacher_sampling_logits[i]
                    ret = get_efficient(logits_dict=logits_dict,
                                        x_lens=x_lens,
                                        y=sampling_y[i],
                                        )
                    student_sampling = ret["student"]
                    teacher_sampling = ret["teacher"]
                elif use_1best:
                    logits_dict = dict()
                    logits_dict["student"] = sampling_logits
                    logits_dict["teacher"] = teacher_sampling_logits[i]
                    student = F.log_softmax(logits_dict["student"], dim=-1)
                    teacher = F.softmax(logits_dict["teacher"], dim=-1)
                    sampled_y_alignments = torch.zeros(org_encoder_out.size(0),
                                                       org_encoder_out.size(1),
                                                       dtype=torch.int64,
                                                       device=org_encoder_out.device)
                    sampled_y_lens = [len(i) for i in sampling_y[i].tolist()]
                    for ii in range(len(sampled_y_lens)):
                        sampled_y_alignments[ii, :x_lens[ii]] = create_pseudo_alignment(x_lens[ii],
                                                                                      sampled_y_lens[ii])
                    idx = sampled_y_alignments.unsqueeze(-1).expand(-1, -1, sampling_logits.shape[-1]).unsqueeze(2)
                    idx = idx.to(torch.int64)
                    teacher = torch.gather(teacher, 2, idx)
                    student = torch.gather(student, 2, idx)
                    max_len = sampling_logits.size(1)
                    mask = torch.arange(max_len, device=sampling_logits.device).expand(sampling_logits.size(0), max_len) < x_lens.unsqueeze(1)
                    mask = mask.unsqueeze(-1).unsqueeze(-1)
                    student_sampling = student * mask.float()
                    teacher_sampling = teacher * mask.float()
                elif use_pruned:
                    logits_dict = dict()
                    logits_dict["student"] = sampling_logits
                    logits_dict["teacher"] = teacher_sampling_logits[i]
                    student = F.log_softmax(logits_dict["student"], dim=-1)
                    teacher = F.softmax(logits_dict["teacher"], dim=-1)
                    sampled_y_alignments = torch.zeros(org_encoder_out.size(0),
                                                       org_encoder_out.size(1),
                                                       dtype=torch.int64,
                                                       device=org_encoder_out.device)
                    sampled_y_lens = [len(i) for i in sampling_y[i].tolist()]
                    for ii in range(len(sampled_y_lens)):
                        sampled_y_alignments[ii, :x_lens[ii]] = create_pseudo_alignment(x_lens[ii],
                                                                                      sampled_y_lens[ii])
                    idx = sampled_y_alignments.unsqueeze(-1).expand(-1, -1, sampling_logits.shape[-1]).unsqueeze(2)
                    idx = idx.to(torch.int64)
                    teacher = torch.gather(teacher, 2, idx)
                    student = torch.gather(student, 2, idx)
                    max_len = sampling_logits.size(1)
                    mask = torch.arange(max_len, device=sampling_logits.device).expand(sampling_logits.size(0), max_len) < x_lens.unsqueeze(1)
                    mask = mask.unsqueeze(-1).unsqueeze(-1)
                    student_sampling = student * mask.float()
                    teacher_sampling = teacher * mask.float()
                    """
                    elif use_pruned:
                        if use_sq_simple_loss_range is False:
                            # original working code
                            student_sampling = F.log_softmax(sampling_logits, dim=-1)
                            teacher_sampling = F.softmax(teacher_sampling_logits[i], dim=-1)

                            # to make sampled pseudo alignment
                            sampled_y_alignments = torch.zeros_like(pseudo_y_alignment,
                                                                    dtype=pseudo_y_alignment.dtype)
                            sampled_y_lens = [len(i) for i in sampling_y[i].tolist()]
                            for ii in range(len(sampled_y_lens)):
                                sampled_y_alignments[ii, :x_lens[ii]] = create_pseudo_alignment(x_lens[ii],
                                                                                              sampled_y_lens[ii])

                            #print(f"{pseudo_y_alignment.shape=}")
                            #print(f"{pseudo_y_alignment[0]=}")
                            #print(f"{sampled_y_alignments.shape=}")
                            #print(f"{sampled_y_alignments[0]=}")
                            #print(f"{sampled_y_alignments[-1]=}")
                            #print(f"{sampling_logits.shape}")
                            #print(f"{sampling_y[i]=}")
                            #print(f"{[len(i) for i in sampling_y[i].tolist()]=}")
                            #print(f"{y_lens=}")
                            #print(f"{sampling_y[i][0]=}")
                            #student = F.log_softmax(logits, dim=-1)
                            #teacher = F.softmax(teacher_logits, dim=-1)
                            max_len = torch.max(y_lens)
                            if pruned_range > max_len:
                                pruned_range = max_len
                            idx = create_increasing_sublists_torch(sampled_y_alignments, pruned_range)
                            batch_idx = torch.arange(sampling_logits.size(0)).view(-1, 1, 1)
                            batch_idx = batch_idx.expand(-1, idx.size(-2), idx.size(-1))
                            teacher = teacher_sampling[batch_idx, torch.arange(sampling_logits.size(1)).view(1, -1, 1), idx]
                            student = student_sampling[batch_idx, torch.arange(sampling_logits.size(1)).view(1, -1, 1), idx]
                            max_len = sampling_logits.size(1)
                            mask = torch.arange(max_len, device=sampling_logits.device).expand(sampling_logits.size(0), max_len) < x_lens.unsqueeze(1)
                            mask = mask.unsqueeze(-1).unsqueeze(-1)
                            student = student * mask.float()
                            teacher = teacher * mask.float()
                        elif use_sq_simple_loss_range is True:
                            student_sampling = F.log_softmax(sampling_logits, dim=-1)

                            # getting teacher encoder output
                            with torch.no_grad():
                                teacher_sampling = F.softmax(teacher_sampling_logits[i], dim=-1)
                                teacher_encoder_out, teacher_encoder_out_lens = teacher_model.get_encoder_out(
                                    x=x,
                                    x_lens=org_x_lens,
                                    use_grad=False,
                                )
                                sos_sampling_y_padded = add_sos(sampling_y[i], sos_id=blank_id)
                                sos_sampling_y_padded = sos_sampling_y_padded.pad(mode="constant", padding_value=blank_id)
                                teacher_decoder_out = teacher_model.decoder(sos_sampling_y_padded)

                                sampling_y_padded = sampling_y[i].pad(mode="constant", padding_value=0)
                                teacher_am = teacher_model.simple_am_proj(teacher_encoder_out)
                                teacher_lm = teacher_model.simple_lm_proj(teacher_decoder_out)
                                sampling_row_splits = sampling_y[i].shape.row_splits(1)
                                sampling_y_lens = sampling_row_splits[1:] - sampling_row_splits[:-1]
                                sampling_boundary = torch.zeros((teacher_encoder_out.size(0), 4), dtype=torch.int64, device=teacher_encoder_out.device)
                                sampling_boundary[:, 2] = sampling_y_lens
                                sampling_boundary[:, 3] = teacher_encoder_out_lens

                                with torch.cuda.amp.autocast(enabled=False):
                                    # getting ranges
                                    _, (teacher_px_grad, teacher_py_grad) = k2.rnnt_loss_smoothed(
                                        lm=teacher_lm.float(),
                                        am=teacher_am.float(),
                                        symbols=sampling_y_padded.to(torch.int64),
                                        termination_symbol=blank_id,
                                        lm_only_scale=0.25,
                                        am_only_scale=0.0,
                                        boundary=sampling_boundary,
                                        reduction="sum",
                                        return_grad=True,
                                    )

                                teacher_sampling_ranges = k2.get_rnnt_prune_ranges(
                                    px_grad=teacher_px_grad,
                                    py_grad=teacher_py_grad,
                                    boundary=sampling_boundary,
                                    s_range=pruned_range,
                                )
                            idx = teacher_sampling_ranges
                            batch_idx = torch.arange(teacher_sampling_logits[i].size(0)).view(-1, 1, 1)
                            batch_idx = batch_idx.expand(-1, idx.size(-2), idx.size(-1))
                            teacher = teacher_sampling[batch_idx, torch.arange(teacher_sampling.size(1)).view(1, -1, 1), idx]
                            student = student_sampling[batch_idx, torch.arange(student_sampling.size(1)).view(1, -1, 1), idx]

                            # T-axis direction masking
                            max_len = sampling_logits.size(1)
                            mask = torch.arange(max_len, device=sampling_logits.device).expand(sampling_logits.size(0), max_len) < x_lens.unsqueeze(1)
                            mask = mask.unsqueeze(-1).unsqueeze(-1)
                            student = student * mask.float()
                            teacher = teacher * mask.float()

                            student_sampling = student_sampling * mask.float()
                            teacher_sampling = teacher_sampling * mask.float()
                    """
                else: #original code
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

        """
        if use_sq_sampling:
            sampling_loss = torch.tensor(0.0).to(logits.device)
            for i in range(sq_sampling_num):
                sampling_logits = self.get_logits_with_encoder_out(
                    encoder_out=encoder_out,
                    encoder_out_lens=x_lens,
                    y=sampling_y[i],
                    use_grad=True,
                )

                student_sampling_logits = sampling_logits

                student_sampling = F.log_softmax(student_sampling_logits, dim=-1)
                teacher_sampling = F.softmax(teacher_sampling_logits[i], dim=-1)
        """
        ret = dict()
        ret["org_loss"] = loss
        if use_kd:
            ret["kd_loss"] = kd_loss
        else:
            ret["kd_loss"] = torch.tensor(0.0)

        if use_sq_sampling:
            ret["sampling_loss"] = sampling_loss
        else:
            ret["sampling_loss"] = torch.tensor(0.0)

        return ret

    def get_logits(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        y: k2.RaggedTensor,
        use_grad: bool = False,
    ) -> torch.Tensor:

        #assert x.ndim == 3, x.shape
        #assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        #assert x.size(0) == x_lens.size(0) == y.dim0

        #encoder_out, x_lens = self.encoder(x, x_lens)
        assert torch.all(encoder_out_lens > 0)

        # Now for the decoder, i.e., the prediction network
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)
        sos_y_padded = sos_y_padded.to(torch.int64)

        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = self.decoder(sos_y_padded)

        encoder_out = self.joiner.encoder_proj(encoder_out)
        decoder_out = self.joiner.decoder_proj(decoder_out)

        logits = self.joiner(
            encoder_out=encoder_out,
            decoder_out=decoder_out,
            project_input=False,
        )

        if use_grad is False:
            logits = logits.detach()

        return logits

    def get_logits_with_encoder_out(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        y: k2.RaggedTensor,
        use_grad: bool = False,
    ) -> torch.Tensor:

        assert y.num_axes == 2, y.num_axes

        assert torch.all(encoder_out_lens > 0)

        # Now for the decoder, i.e., the prediction network
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)
        sos_y_padded = sos_y_padded.to(torch.int64)

        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = self.decoder(sos_y_padded)

        encoder_out = self.joiner.encoder_proj(encoder_out)
        decoder_out = self.joiner.decoder_proj(decoder_out)

        logits = self.joiner(
            encoder_out=encoder_out,
            decoder_out=decoder_out,
            project_input=False,
        )

        if use_grad is False:
            logits = logits.detach()

        return logits

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

    def get_ranges_and_logits_with_encoder_out(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        y: k2.RaggedTensor,
        pruned_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.25,
        use_grad: bool = False,
    ):
        """
        this function is used to get the pruned path and logits
        Args:
            encoder_out:
                A 3-D tensor of shape (N, T, C).
            encoder_out_lens:
                A 1-D tensor of shape (N,). It contains the number of frames in `x`
                before padding.
            y:
                A ragged tensor with 2 axes [utt][label]. It contains labels of each
                utterance.
            pruned_range:
                The range to be pruned
            am_scale:
                The acoustic model scale
            lm_scale:
                The language model scale
            use_grad:
                If False, the returned tensors will be detached from the computation
                graph. If True, the returned tensors will have gradients.
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

        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=pruned_range,
        )

        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        ranges.requires_grad_(False)
        if use_grad is False:
            logits = logits.detach()

        ret = (ranges, logits)

        return ret

    def get_logits_with_encoder_out_and_ranges(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        y: k2.RaggedTensor,
        ranges: torch.Tensor = None,
        use_grad: bool = False,
    ) -> torch.Tensor:

        assert y.num_axes == 2, y.num_axes

        assert torch.all(encoder_out_lens > 0)

        # Now for the decoder, i.e., the prediction network
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)
        sos_y_padded = sos_y_padded.to(torch.int64)

        boundary = torch.zeros((encoder_out.size(0), 4), dtype=torch.int64, device=encoder_out.device)
        boundary[:, 2] = y_lens
        boundary[:, 3] = encoder_out_lens

        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = self.decoder(sos_y_padded)

        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        logits = self.joiner(
            encoder_out=am_pruned,
            decoder_out=lm_pruned,
            project_input=False,
        )

        ranges.requires_grad_(False)
        if use_grad is False:
            logits = logits.detach()

        return logits

def get_efficient(
    logits_dict: Union[dict, torch.Tensor],
    x_lens: torch.Tensor,
    y: torch.Tensor,
    blank_id=0,
):
    y_padded = y.pad(mode="constant", padding_value=0)
    logits = logits_dict["student"]
    teacher_logits = logits_dict["teacher"]
    tensor_y_padded = y_padded.to(torch.int64)
    indices = tensor_y_padded.unsqueeze(1).unsqueeze(-1)
    logits = torch.softmax(logits, dim=-1)
    teacher_logits = torch.softmax(teacher_logits, dim=-1)
    # py
    py_logits = torch.gather(logits, -1, indices.expand(-1, logits.size(1), -1, -1))
    teacher_py_logits = torch.gather(teacher_logits, -1, indices.expand(-1, logits.size(1), -1, -1))
    # blank
    blank_indices = torch.full_like(indices, blank_id)
    blank_logits = torch.gather(logits, -1, blank_indices.expand(-1, logits.size(1), -1, -1))
    teacher_blank_logits = torch.gather(teacher_logits, -1, blank_indices.expand(-1, logits.size(1), -1, -1))
    # remainder
    rem_logits = torch.ones(py_logits.shape, device=logits.device) - py_logits - blank_logits
    teacher_rem_logits = torch.ones(teacher_py_logits.shape, device=teacher_logits.device) - teacher_py_logits - teacher_blank_logits

    # concatenate
    eps = 1e-10
    student = torch.cat([py_logits, blank_logits, rem_logits], dim=-1)
    student = torch.clamp(student, eps, 1.0)
    student = student.log()
    teacher_rem_logits[torch.where(teacher_rem_logits<0)] = eps
    teacher = torch.cat([teacher_py_logits, teacher_blank_logits, teacher_rem_logits], dim=-1)

    # T-axis direction masking
    max_len = logits.size(1)
    mask = torch.arange(max_len, device=logits.device).expand(logits.size(0), max_len) < x_lens.unsqueeze(1)
    mask = mask.unsqueeze(-1).unsqueeze(-1)
    student = student * mask.float()
    teacher = teacher * mask.float()

    row_splits = y.shape.row_splits(1)
    y_lens = row_splits[1:] - row_splits[:-1]
# U-axis direction masking
    max_len = torch.max(y_lens)
    mask = torch.arange(max_len, device=logits.device).expand(logits.size(0), logits.size(1), max_len) < y_lens.unsqueeze(-1).unsqueeze(-1)
    mask = mask.unsqueeze(-1)
    student = student * mask.float()
    teacher = teacher * mask.float()

    ret = dict()
    ret["student"] = student
    ret["teacher"] = teacher

    return ret

def create_increasing_sublists_torch(original_tensor, length=5):
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

def create_pseudo_alignment(x_len, y_len):
    #prepend 0 in the beginning
    padded_y_len = y_len +1
    if x_len < padded_y_len:
        padded_y_len = x_len.item()
    y = torch.arange(padded_y_len)
    base_repeats = np.ones(padded_y_len, dtype=int)
    remaining_len = x_len - padded_y_len
    #print(f"{x_len=}")
    #print(f"{y_len=}")
    #print(f"{padded_y_len=}")
    #print(f"{y=}")
    #print(f"{base_repeats=}")
    #print(f"{remaining_len=}")
    #random distribution
    additional_repeats = np.random.multinomial(remaining_len, np.ones(padded_y_len)/padded_y_len)
    repeats = base_repeats + additional_repeats

    #creating tensor
    ret = torch.cat([torch.full((count,), num) for num, count in zip(y, repeats)])

    return ret
