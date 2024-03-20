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

        self.kd_criterion = nn.KLDivLoss(reduction="batchmean")
        self.ctc_layer = nn.Linear(encoder_dim, vocab_size)
        self.ctc_criterion = None

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y_list: list() = None, #k2.RaggedTensor,
        use_kd: bool = False,
        teacher_logits_list: list() = None, #torch.Tensor = None,
        use_ctc: bool = False,
        teacher_model: nn.Module = None,
        use_efficient: bool = False,
        use_1best: bool = False,
        use_nbest: bool = False,
        nbest_num: int = 4,
        pseudo_y_alignment_list: list() = None, #torch.Tensor = None,
        use_sequence: bool = False,
        pseudo_y_sequence: torch.Tensor = None,
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
        org_x_lens = x_lens.clone()
        encoder_out, x_lens = self.encoder(x, org_x_lens)
        encoder_out = self.joiner.encoder_proj(encoder_out)
        logits_list = list()
        y_lens_list = list()
        y_padded_list = list()
        assert nbest_num > 0, f"{nbest_num=}"
        if nbest_num > len(y_list):
            nbest_num = len(y_list)

        for n in range(nbest_num):
            y = y_list[n]
            assert x.ndim == 3, x.shape
            assert org_x_lens.ndim == 1, org_x_lens.shape
            assert y.num_axes == 2, y.num_axes

            assert x.size(0) == org_x_lens.size(0) == y.dim0

            # here encoder_out is
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

            decoder_out = self.joiner.decoder_proj(decoder_out)

            logits = self.joiner(
                encoder_out=encoder_out,
                decoder_out=decoder_out,
                project_input=False,
            )

            # Note: y does not start with SOS
            # y_padded : [B, S]
            y_padded = y.pad(mode="constant", padding_value=0)
            logits_list.append(logits)
            y_padded_list.append(y_padded)
            y_lens_list.append(y_lens)

        assert hasattr(torchaudio.functional, "rnnt_loss"), (
            f"Current torchaudio version: {torchaudio.__version__}\n"
            "Please install a version >= 0.10.0"
        )

        # etracting the first element
        logits = logits_list[0]
        y_padded = y_padded_list[0]
        y_lens = y_lens_list[0]

        loss = torchaudio.functional.rnnt_loss(
            logits=logits,
            targets=y_padded,
            logit_lengths=x_lens,
            target_lengths=y_lens,
            blank=blank_id,
            reduction="sum",
        )

        if use_kd:
            teacher_logits = teacher_logits_list[0]
            student = F.log_softmax(logits, dim=-1)
            teacher = F.softmax(teacher_logits, dim=-1)

            # What does it mean for?
            if use_efficient is False and use_1best is False and use_sequence is False and use_nbest is False:
                # T-axis direction masking
                max_len = logits.size(1)
                mask = torch.arange(max_len, device=logits.device).expand(logits.size(0), max_len) < x_lens.unsqueeze(1)
                mask = mask.unsqueeze(-1).unsqueeze(-1)
                student = student * mask.float()
                teacher = teacher * mask.float()

            # the case when use_efficient is True and use_1best is True
            # is not supported yet.
            assert not (use_efficient and use_1best)
            assert not (use_efficient and use_sequence)
            assert not (use_1best and use_sequence)

            if use_efficient:
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
                teacher_rem_logits[torch.where(teacher_rem_logits<0.0)] = eps
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
                """
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
                    import sys
                    sys.exit(0)
                """

            if use_1best:
                # getting the 1-best path
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

            if use_nbest:
                # getting the 1-best path
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
                teacher_list = list()
                student_list = list()
                for n in range(nbest_num):
                    pseudo_y_alignment = pseudo_y_alignment_list[n]
                    logits = logits_list[n]
                    teacher_logits = teacher_logits_list[n]
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
                    student_list.append(student)
                    teacher_list.append(teacher)
                    #print(f"{student[-1,-1,:,:]=}")
                    #print(f"{teacher[-1,-1,:,:]=}")

            if use_sequence:
                idx = pseudo_y_alignment.unsqueeze(-1).expand(-1, -1, logits.shape[-1]).unsqueeze(2)
                idx = idx.to(torch.int64)
                teacher = torch.gather(teacher, 2, idx)
                student = torch.gather(student, 2, idx)
                # print all tensor
                #torch.set_printoptions(profile="full")
                #print(f"selected {student[-1, 12:14]=}")
                #print(f"selected {teacher[-1, 12:14]=}")
                # pick only the seq values
                pseudo_y_sequence = pseudo_y_sequence.to(torch.int64)
                teacher = torch.gather(teacher, -1, pseudo_y_sequence.unsqueeze(-1).unsqueeze(-1))
                student = torch.gather(student, -1, pseudo_y_sequence.unsqueeze(-1).unsqueeze(-1))
                max_len = logits.size(1)
                mask = torch.arange(max_len, device=logits.device).expand(logits.size(0), max_len) < x_lens.unsqueeze(1)
                mask = mask.unsqueeze(-1).unsqueeze(-1)
                student = student * mask.float()
                teacher = teacher * mask.float()
                #print(f"{pseudo_y_sequence[-1]=}")
                #print(f"{student[-1, 12:14]=}")
                #print(f"{teacher[-1, 12:14]=}")
                #print(f"{student[-1,-1,:,:]=}")
                #print(f"{teacher[-1,-1,:,:]=}")

                #import sys
                #sys.exit(0)

            if use_nbest is True:
                kd_loss = torch.tensor(0.0, device=encoder_out.device)
                for n in range(nbest_num):
                    student = student_list[n]
                    teacher = teacher_list[n]
                    #student = student.to(device=encoder_out.device)
                    #teacher = teacher.to(device=encoder_out.device)
                    #print(f"{student=}")
                    #print(f"{teacher=}")
                    kd_loss += self.kd_criterion(student, teacher)
                kd_loss = kd_loss / nbest_num
                #print(f"{teacher.shape=}")
                #print(f"{kd_loss=}")
                #import sys
                #sys.exit(0)
            else:
                kd_loss = self.kd_criterion(student, teacher)
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

        return ret

    def get_logits(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        use_grad: bool = False,
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
            logits = logits.detach_()

        return logits

    def get_logits_nbest(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y_list: list(), #k2.RaggedTensor,
        num_nbest: int = 4,
        use_grad: bool = False,
    ) -> torch.Tensor:

        logits_list = list()

        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert torch.all(x_lens > 0)

        encoder_out, x_lens = self.encoder(x, x_lens)
        encoder_out = self.joiner.encoder_proj(encoder_out)

        for n in range(num_nbest):
            y = y_list[n]
            assert y.num_axes == 2, y.num_axes
            assert x.size(0) == x_lens.size(0) == y.dim0

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
            decoder_out = self.joiner.decoder_proj(decoder_out)

            logits = self.joiner(
                encoder_out=encoder_out,
                decoder_out=decoder_out,
                project_input=False,
            )

            if use_grad is False:
                logits = logits.detach_()

            logits_list.append(logits)
        return logits_list

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
            logits = logits.detach_()

        return logits
