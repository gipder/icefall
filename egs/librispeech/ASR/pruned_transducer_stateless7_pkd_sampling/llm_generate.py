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
import concurrent.futures
from contextlib import nullcontext

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
    def __init__(self, model=""):
        self.entries = {}  # LLMGenDict 객체들을 저장할 딕셔너리
        self.model = model

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

    def get_model(self):
        return self.model

    def delete_entry(self, key):
        if key in self.entries:
            del self.entries[key]
        else:
            print(f"Entry with key '{key}' not found.")

    def __str__(self):
        print(f"Model: {self.model}")
        if not self.entries:
            return "LLMGenDB is empty."
        return "\n".join([str(entry) for entry in self.entries.values()])

def create_client(port="1824",
                  model="meta-llama/Meta-Llama-3-70B-Instruct",
                  apikey=None) -> OpenAI:

    if model.startswith("gpt-"):
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

    parser.add_argument(
        "--use-multiprocessing",
        type=str2bool,
        default=False,
        help="Whether to use multiprocessing or not."
    )

    parser.add_argument(
        "--delete-existing-db",
        type=str2bool,
        default=False,
        help="Whether to delete the existing DB or not."
    )

    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help="How many process will be used when use-multiprocessing is True."
    )

    return parser

def get_content(client, params, texts, n):
    message = make_message(texts[n])
    chat_completion = client.chat.completions.create(
        model=params.llm_model,
        messages=message,
        stream=False
    )
    #make the character uppercase and remove the leading and trailing whitespaces
    content = clear_sentence(chat_completion.choices[0].message.content)
    return content

def create_client_and_get_content(apikey, params, uid, text):
    client = create_client(port=params.port,
                           model=params.llm_model,
                           apikey=apikey)
    message = make_message(text)
    chat_completion = client.chat.completions.create(
        model=params.llm_model,
        messages=message,
        stream=False,
    )
    #make the character uppercase and remove the leading and trailing whitespaces
    content = clear_sentence(chat_completion.choices[0].message.content)
    return content, uid, text

def process_one_sample(apikey, params, uid, text, llm_gen_db):
    client = create_client(port=params.port,
                           model=params.llm_model,
                           apikey=apikey)
    message = make_message(text)
    chat_completion = client.chat.completions.create(
        model=params.llm_model,
        messages=message,
        stream=False,
    )
    #make the character uppercase and remove the leading and trailing whitespaces
    content = clear_sentence(chat_completion.choices[0].message.content)

    llm_gen_db.add_entry(uid, text, content)

def process_multi_samples(executor, num_processes, futures, apikey, params, cuts, texts, llm_gen_db):
    #        if cuts[n].id in llm_gen_db_keys:
    #            logging.info(f"Skipping {cuts[n].id}")
    #            continue
    #        futures.append(executor.submit(create_client_and_get_content,
    #                                       apikey,
    #                                       params,
    #                                       cuts[n].id,
    #                                       texts[n]))
    #    concurrent.futures.wait(futures)
    #    for future in futures:
    #        content, cuts_id, text = future.result()
    #        llm_gen_db.add_entry(cuts_id, text, content)
    #        idx += 1
    #    futures.clear()
    #if futures is not clear, then clear it
    idx = 0
    if len(futures) > 0:
        futures.clear()
    for n in range(0, len(cuts), num_processes):
        for p in range(num_processes):
            if n + p >= len(cuts):
                break
            if cuts[n+p].id in llm_gen_db.keys():
                logging.info(f"Skipping {cuts[n+p].id}")
                continue
            futures.append(executor.submit(create_client_and_get_content,
                                           apikey,
                                           params,
                                           cuts[n+p].id,
                                           texts[n+p]))
        concurrent.futures.wait(futures)
        for future in futures:
            content, cuts_id, text = future.result()
            llm_gen_db.add_entry(cuts_id, text, content)
            idx += 1
        futures.clear()
    return idx

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
    setup_logger(f"{params.res_dir}/log-llm-generate")

    if params.llm_model.startswith("gpt-"):
        if os.environ.get("OPENAI_API_KEY") is None:
            apikey = getpass.getpass("Please enter your OpenAI API key: ")
        else:
            apikey = os.environ.get("OPENAI_API_KEY")
    else:
        if params.use_multiprocessing is True:
            logging.error("Multiprocessing is not supported with TGI. Please set use_multiprocessing to False.")
            return
        apikey = None

    # for debugging
    debug_idx = 0

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

    test_sets = ["train-clean-100"]
    test_dls = [train_clean_100_dl]
    #test_sets = ["train-small"]
    #test_dls = [train_small_dl]

    if params.use_multiprocessing is False:
        client = create_client(port=params.port,
                               model=params.llm_model,
                               apikey=apikey)

    llm_gen_db = LLMGenDB()
    logging.info("LLM Generating Start")
    idx = 0

    executor_context = concurrent.futures.ProcessPoolExecutor() if params.use_multiprocessing else nullcontext()

    for test_set, test_dl in zip(test_sets, test_dls):
        logging.info(f"LLM Generating {test_set}")
        # testing save the DB in pickle
        llm_gen_db_file = f"{params.res_dir}/llm_gen_db.{test_set}.{os.path.basename(params.llm_model)}.pkl"
        # if the file already exists, then delete it
        if params.delete_existing_db and os.path.exists(llm_gen_db_file):
            os.remove(llm_gen_db_file)
            logging.info(f"File {llm_gen_db_file} already exists. Deleting it.")
        else:
            if os.path.exists(llm_gen_db_file):
                logging.info(f"File {llm_gen_db_file} already exists. Loading it.")
                with open(llm_gen_db_file, "rb") as f:
                    llm_gen_db = pickle.load(f)
                num_samples = len(llm_gen_db.entries)
                logging.info(f"Loaded {num_samples} samples.")
                llm_gen_db_keys = llm_gen_db.keys()

        with executor_context as executor:
            futures = []
            for batch_idx, batch in enumerate(test_dl):
                texts = batch["supervisions"]["text"]
                cuts = batch["supervisions"]["cut"]
                if params.use_multiprocessing:
                    idx += process_multi_samples(executor,
                                                 params.num_processes,
                                                 futures, apikey, params,
                                                 cuts, texts, llm_gen_db)
                else:
                    for n in range(len(texts)):
                        if cuts[n].id in llm_gen_db_keys:
                            logging.info(f"Skipping {cuts[n].id}")
                            continue
                        content = get_content(client, params, texts, n)
                        llm_gen_db.add_entry(cuts[n].id, texts[n], content)
                        idx += 1

                if params.use_debug:
                    logging.info(f"Debugging")
                    with open(llm_gen_db_file, "wb") as f:
                        pickle.dump(llm_gen_db, f)
                    # testing load the DB from pickle
                    with open(llm_gen_db_file, "rb") as f:
                        llm_gen_db_from_pickle = pickle.load(f)
                    for key in llm_gen_db_from_pickle.entries.keys():
                        print("key: " + key)
                        print("origin: " + llm_gen_db_from_pickle.get_origin(key))
                        print("generated: " + llm_gen_db_from_pickle.get_value(key))
                        print("wer: " + str(llm_gen_db_from_pickle.get_wer(key)))
                    if debug_idx == 2:
                        import sys
                        sys.exit(0)
                    debug_idx += 1

                if batch_idx % 100 == 0:
                    logging.info(f"Batch {batch_idx} is done. {idx} samples are done.")

            with open(llm_gen_db_file, "wb") as f:
                pickle.dump(llm_gen_db, f)


if __name__ == "__main__":
    main()

