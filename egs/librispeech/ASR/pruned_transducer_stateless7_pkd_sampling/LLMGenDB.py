from openai import OpenAI
import argparse
import logging

from train import add_model_arguments, get_params, get_transducer_model

import sentencepiece as spm
import torch
import torch.nn as nn
import k2
from asr_datamodule import LibriSpeechAsrDataModule
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)

from datetime import datetime

class LLMGenDict:
    def __init__(self, key, original_sentence, generated_sentence):
        self.key = key
        self.original_sentence = original_sentence
        self.generated_sentence = generated_sentence

    def update_generated_sentence(self, new_generated_sentence):
        self.generated_sentence = new_generated_sentence

    def __str__(self):
        return f"Key: {self.key}\nOriginal Sentence: {self.original_sentence}\nGenerated Sentence: {self.generated_sentence}"


class LLMGenDB:
    def __init__(self):
        self.entries = {}  # LLMGenDict 객체들을 저장할 딕셔너리

    def add_entry(self, key, original_sentence, generated_sentence):
        if key not in self.entries:
            self.entries[key] = LLMGenDict(key, original_sentence, generated_sentence)
        else:
            print(f"Entry with key '{key}' already exists.")

    def get_entry(self, key):
        if key in self.entries:
            return self.entries[key]
        else:
            print(f"Entry with key '{key}' not found.")
            return None

    def get_original_sentence(self, key):
        if key in self.entries:
            return self.entries[key].original_sentence
        else:
            print(f"Entry with key '{key}' not found.")
            return None

    def get_generated_sentence(self, key):
        if key in self.entries:
            return self.entries[key].generated_sentence
        else:
            print(f"Entry with key '{key}' not found.")
            return None

    def get_value(self, key):
        return self.get_generated_sentence(key)

    def delete_entry(self, key):
        if key in self.entries:
            del self.entries[key]
        else:
            print(f"Entry with key '{key}' not found.")

    def __str__(self):
        if not self.entries:
            return "LLMGenDB is empty."
        return "\n".join([str(entry) for entry in self.entries.values()])

# 예제 사용
# LLMGenDB 인스턴스 생성
llm_db = LLMGenDB()

# 데이터 추가 예제
llm_db.add_entry("id001", "Original sentence 1", "Generated sentence 1")
llm_db.add_entry("id002", "Original sentence 2", "Generated sentence 2")

# 데이터 조회 예제
print(llm_db.get_entry("id001"))  # id001에 해당하는 객체 출력

# 데이터 삭제 예제
#llm_db.delete_entry("id002")  # id002에 해당하는 객체 삭제

# 전체 DB 출력 예제
print(llm_db)

print(llm_db.get_original_sentence("id001"))
print(llm_db.get_generated_sentence("id001"))
print(llm_db.get_original_sentence("id002"))
print(llm_db.get_generated_sentence("id002"))
print(llm_db.get_value("id001"))

def create_client():
    # init the client but point it to TGI
    client = OpenAI(
        base_url="http://localhost:18244/v1",
        api_key="-"
    )
    """
    chat_completion = client.chat.completions.create(
        model="tgi",
        messages=[
            {"role": "system", "content": "You are an ASR expert." },
            {"role": "user", "content": "What is deep learning?"}
        ],
        stream=False
    )

    print(chat_completion)
    """
    return client

def make_message(asr_label: str):
    message = [
        {"role": "system", "content": "You are an ASR expert."},
        {"role": "user", "content": f"Please generate ASR error pattern with WER 50% from this sentence, {asr_label}"},
        {"role": "user", "content": f"Please show me only the pattern without anything"},
        {"role": "system", "content": "The pattern is as follows: "}
    ]

    return message

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="",
        required=False,
        help="The experiment directory where the model is saved."
    )
    return parser

@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    decoding_method = "llm-generate"
    params.res_dir = params.exp_dir / f"log-{decoding_method}"

    # already append the time information
    #now = datetime.now()
    #time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    setup_logger(f"{params.res_dir}/log-llm-generate")

    args.return_cuts = True
    librispeech = LibriSpeechAsrDataModule(args)

    train_clean_100_cuts = librispeech.train_clean_100_cuts()
    train_small_cuts = librispeech.train_small_cuts()
    test_clean_cuts = librispeech.test_clean_cuts()
    test_other_cuts = librispeech.test_other_cuts()

    train_clean_100_dl = librispeech.test_dataloaders(train_clean_100_cuts)
    train_small_dl = librispeech.test_dataloaders(train_small_cuts)
    test_clean_dl = librispeech.test_dataloaders(test_clean_cuts)
    test_other_dl = librispeech.test_dataloaders(test_other_cuts)

    test_sets = ["train-small"]
    test_dls = [train_small_dl]

    client = create_client()
    for test_set, test_dl in zip(test_sets, test_dls):
        logging.info(f"LLM Generating {test_set}")
        for batch_idx, batch in enumerate(test_dl):
            #print(batch_idx)
            #print(batch)
            texts = batch["supervisions"]["text"]
            cuts = batch["supervisions"]["cut"]
            message = make_message(texts[0])
            print(f"{message=}")
            chat_completion = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                messages=message,
                stream=False
            )
            print(f"{texts[0]=}")
            print(f"{cuts[0].id=}")
            #print(chat_completion)
            print(chat_completion.choices[0].message.content)
            break

if __name__ == "__main__":
    main()

