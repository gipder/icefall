from openai import OpenAI
import argparse
import logging

from train import add_model_arguments, get_params, get_transducer_model

import sentencepiece as spm
import torch
import torch.nn as nn
#import k2
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
import re
import os
import pickle
from jiwer import wer
import getpass

class LLMGenDict:
    def __init__(self, key, original_sentence, generated_sentence):
        self.key = key
        self.original_sentence = original_sentence
        self.generated_sentence = generated_sentence
        self.wer = wer(self.original_sentence, self.generated_sentence)

    def update_generated_sentence(self, new_generated_sentence):
        self.generated_sentence = new_generated_sentence

    def __str__(self):
        return f"Key: {self.key}\nOriginal Sentence: {self.original_sentence}\nGenerated Sentence: {self.generated_sentence}\nWER: {self.wer}"


class LLMGenDB:
    def __init__(self):
        self.entries = {}  # LLMGenDict 객체들을 저장할 딕셔너리

    def keys(self):
        return self.entries.keys()

    def values(self):
        return self.entries.values()

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

    def get_origin(self, key):
        return self.get_original_sentence(key)

    def get_wer(self, key):
        return self.entries[key].wer

    def delete_entry(self, key):
        if key in self.entries:
            del self.entries[key]
        else:
            print(f"Entry with key '{key}' not found.")

    def __str__(self):
        if not self.entries:
            return "LLMGenDB is empty."
        return "\n".join([str(entry) for entry in self.entries.values()])

def create_client(port="1824",
                  model="meta-llama/Meta-Llama-3-70B-Instruct") -> OpenAI:

    if model == "gpt-4o" or model == "gpt-4" or model == "gpt-3.5-turbo":
        apikey = getpass.getpass("Please enter your OpenAI API key: ")
        client = OpenAI(
            api_key=apikey
        )

    else:
        # init the client but point it to TGI
        client = OpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="-"
        )

    return client

def make_message(asr_label: str):
    message = [
        {"role": "system", "content": "You are an ASR expert."},
        {"role": "user", "content": f"Please generate ASR error pattern with WER 25% from this sentence, {asr_label}."},
        {"role": "user", "content": f"Please show me only the generated sentence without anything."},
        {"role": "system", "content": "The pattern is as follows: "}
    ]

    return message

def clear_sentence(sentence: str):
    sentence = sentence.upper().strip()
    sentence = re.sub(r'[^a-zA-Z\s\']', '', sentence)
    sentence = re.sub(r'[\r\n]', '', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

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

    parser.add_argument(
        "--llm-model",
        type=str,
        default="meta-llama/Meta-Llama-3-70B-Instruct",
        help="The model name or path to the model."
    )

    parser.add_argument(
        "--port",
        type=str,
        default="1824",
        help="Port number for a TGI server"
    )

    parser.add_argument(
        "--use-debug",
        type=str2bool,
        default=False,
        help="Whether to use debug mode or not."
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

    #test_sets = ["train-clean-100"]
    #test_dls = [train_clean_100_dl]
    test_sets = ["train-small"]
    test_dls = [train_small_dl]

    client = create_client(port=params.port,
                           model=params.llm_model)
    llm_gen_db = LLMGenDB()
    logging.info("LLM Generating Start")
    idx = 0
    for test_set, test_dl in zip(test_sets, test_dls):
        logging.info(f"LLM Generating {test_set}")
        for batch_idx, batch in enumerate(test_dl):
            texts = batch["supervisions"]["text"]
            cuts = batch["supervisions"]["cut"]
            message = make_message(texts[0])
            for n in range(len(texts)):
                message = make_message(texts[n])
                chat_completion = client.chat.completions.create(
                    model=params.llm_model,
                    messages=message,
                    stream=False
                )
                #make the character uppercase and remove the leading and trailing whitespaces
                content = clear_sentence(chat_completion.choices[0].message.content)
                llm_gen_db.add_entry(cuts[n].id, texts[n], content)
                idx += 1

            if batch_idx % 100 == 0:
                logging.info(f"Batch {batch_idx} is done. {idx} samples are done.")
            # testing save the DB in pickle
            llm_gen_db_file = f"{params.res_dir}/llm_gen_db.{os.path.basename(params.llm_model)}.pkl"
            # if the file already exists, then delete it
            if os.path.exists(llm_gen_db_file):
                os.remove(llm_gen_db_file)

            with open(llm_gen_db_file, "wb") as f:
                pickle.dump(llm_gen_db, f)

            use_debug = params.use_debug
            if use_debug:
                # testing load the DB from pickle
                with open(llm_gen_db_file, "rb") as f:
                    llm_gen_db_from_pickle = pickle.load(f)
                for key in llm_gen_db_from_pickle.entries.keys():
                    print("origin: " + llm_gen_db_from_pickle.get_origin(key))
                    print("generated: " + llm_gen_db_from_pickle.get_value(key))
                    print("wer: " + str(llm_gen_db_from_pickle.get_wer(key)))
                import sys
                sys.exit(0)

if __name__ == "__main__":
    main()

