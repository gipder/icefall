# Copyright      2021  Piotr Żelasko
#                2022  Xiaomi Corp.        (authors: Fangjun Kuang)
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
from pathlib import Path

from lhotse import CutSet, load_manifest_lazy


class KsponSpeech:
    def __init__(self, manifest_dir: str):
        """
        Args:
          manifest_dir:
            It is expected to contain the following files::

                - ksponspeech_cuts_dev.jsonl.gz
                - ksponspeech_cuts_eval_clean.jsonl.gz
                - ksponspeech_cuts_eval_other.jsonl.gz
                - ksponspeech_cuts_train-100.jsonl.gz
                - ksponspeech_cuts_train.jsonl.gz
        """
        self.manifest_dir = Path(manifest_dir)

    def train_100_cuts(self) -> CutSet:
        logging.info("About to get train-100 cuts")
        return load_manifest_lazy(
            self.manifest_dir / "ksponspeech_cuts_train-100.jsonl.gz"
        )

    def train_all_cuts(self) -> CutSet:
        logging.info("About to get all train cuts")
        return load_manifest_lazy(
            self.manifest_dir / "ksponspeech_cuts_train.jsonl.gz"
        )

    def dev_cuts(self) -> CutSet:
        logging.info("About to get dev cuts")
        return load_manifest_lazy(
            self.manifest_dir / "ksponspeech_cuts_dev.jsonl.gz"
        )

    def eval_clean_cuts(self) -> CutSet:
        logging.info("About to get eval-clean cuts")
        return load_manifest_lazy(
            self.manifest_dir / "ksponspeech_cuts_eval_clean.jsonl.gz"
        )

    def eval_other_cuts(self) -> CutSet:
        logging.info("About to get eval-other cuts")
        return load_manifest_lazy(
            self.manifest_dir / "ksponspeech_cuts_eval_other.jsonl.gz"
        )
