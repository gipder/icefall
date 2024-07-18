import sys
import os
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
import pickle
import jiwer
import getpass
import random
import concurrent.futures
from contextlib import nullcontext
from llm_gen import LLMGenDict, LLMGenDB
from hyp_gen import HYPGenDict, HYPGenDB
from my_utils  import comma_separated_list_in_float

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

def make_message(asr_label: str, params,
                 hyp_gen_db: HYPGenDB = None, keys_in_wer_range: list() = None ) -> List[Dict[str, str]]:
    if params.mode == "semantic":
        return semantic_message(asr_label, params.target_wer)
    elif params.mode == "acoustic":
        return acoustic_message(asr_label, hyp_gen_db, keys_in_wer_range, params.few_shot)

    assert False, f"Unknown mode: {params.mode}"

    return

def semantic_message(asr_label: str, wer: float = 15.0) -> List[Dict[str, str]]:
    message = [
        {"role": "system", "content": f"You are a ASR expert."},
        {"role": "user", "content": f"Please generate ASR error pattern with WER {wer} from this sentence, {asr_label}."},
        {"role": "user", "content": f"Please show me only the generated sentence without anything."},
        {"role": "system", "content": f"The pattern is as follows: "}
    ]

    return message

def acoustic_message(asr_label: str, hyp_gen_db: HYPGenDB, keys_in_wer_range: list(), few_shot: int=1) -> List[Dict[str, str]]:
    message = list()
    message.append({"role": "system", "content": f"You are a ASR expert."})
    message.append({"role": "system", "content": "Generate ASR Error Patterns from examples of users."})
    random_choices = random.choices(keys_in_wer_range, k=few_shot)
    for key in random.choices(keys_in_wer_range, k=few_shot):
        message.append({"role": "user", "content": f"INPUT: {hyp_gen_db.get_origin(key)}"})
        message.append({"role": "system", "content": f"OUTPUT: {hyp_gen_db.get_value(key)[0]}"})
    message.append({"role": "user", "content": f"INPUT: {asr_label}"})
    message.append({"role": "system", "content": f"OUTPUT: "})

    return message

def clear_sentence(sentence: str):
    sentence = sentence.upper().strip()
    sentence = re.sub(r'[^a-zA-Z\s\']', '', sentence)
    sentence = re.sub(r'[\r\n]', '', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

def get_keys_in_range(hyp_gen_db, start: float = 0, end: float = 100):
    keys = list()
    for key in hyp_gen_db.keys():
        if hyp_gen_db.get_wer(key) >= start and hyp_gen_db.get_wer(key) < end:
            keys.append(key)
    return keys

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

    parser.add_argument(
        "--save-interval",
        type=int,
        default=1000,
        help="How frequently we save the model to disk."
    )

    parser.add_argument(
        "--target-wer",
        type=float,
        default=15.0,
        help="The target WER from sentences generated by LLM."
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="semantic",
        help="The mode about prompts to LLM, 'semantic' or 'acoustic'."
    )

    parser.add_argument(
        "--use-hyp-gen",
        type=str2bool,
        default=False,
        help="Whether to use hyp-gen or not."
    )

    parser.add_argument(
        "--hyp-gen-db",
        type=str,
        default="",
        help="The file path to the hyp-gen DB."
    )

    parser.add_argument(
        "--wer-range",
        type=comma_separated_list_in_float,
        default="0,15",
        help="Range of WER to generate acoustic prompts."
    )

    parser.add_argument(
        "--few-shot",
        type=int,
        default=1,
        help="How many examples prompts use to generate sentences."
    )

    return parser

def get_content(client, params, texts, n, hyp_gen_db, keys_in_wer_range):
    message = make_message(texts[n], params,
                           hyp_gen_db=hyp_gen_db,
                           keys_in_wer_range=keys_in)
    chat_completion = client.chat.completions.create(
        model=params.llm_model,
        messages=message,
        stream=False
    )
    #make the character uppercase and remove the leading and trailing whitespaces
    content = clear_sentence(chat_completion.choices[0].message.content)
    return content

def create_client_and_get_content(apikey, params, uid, text, hyp_gen_db, keys_in_wer_range):
    client = create_client(port=params.port,
                           model=params.llm_model,
                           apikey=apikey,)
    message = make_message(text, params,
                           hyp_gen_db=hyp_gen_db,
                           keys_in_wer_range=keys_in_wer_range)

    chat_completion = client.chat.completions.create(
        model=params.llm_model,
        messages=message,
        stream=False,
    )
    #make the character uppercase and remove the leading and trailing whitespaces
    content = clear_sentence(chat_completion.choices[0].message.content)
    return content, uid, text

def process_multi_samples(executor, num_processes, futures, apikey, params, cuts, texts, llm_gen_db,
                          hyp_gen_db, keys_in_wer_range):
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
                                           texts[n+p],
                                           hyp_gen_db,
                                           keys_in_wer_range
                                           ))
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

    logging.info(params)

    # loading hyp_gen_db
    # check first where the file exists
    hyp_gen_db = None
    if params.use_hyp_gen:
        assert os.path.exists(params.hyp_gen_db), f"{params.hyp_gen_db} does not exist."
        f = open(params.hyp_gen_db, "rb")
        hyp_gen_db = pickle.load(f)
        keys_in_wer_range = get_keys_in_range(hyp_gen_db, params.wer_range[0], params.wer_range[1])
        if len(keys_in_wer_range) == 0:
            logging.error(f"No keys in the WER range {params.wer_range}.")
            return
        else:
            logging.info(f"Keys in the WER range {params.wer_range}: {len(keys_in_wer_range)}")

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

    llm_gen_db = LLMGenDB(model=params.llm_model, target_wer=params.target_wer)
    logging.info("LLM Generating Start")
    save_interval = params.save_interval

    executor_context = concurrent.futures.ProcessPoolExecutor() if params.use_multiprocessing else nullcontext()

    llm_gen_db_keys = list()
    for test_set, test_dl in zip(test_sets, test_dls):
        logging.info(f"LLM Generating {test_set}")
        idx = 0
        # testing save the DB in pickle
        llm_gen_db_file = f"{params.res_dir}/llm_gen_db.{test_set}.wer{params.target_wer}.mode_{params.mode}.{os.path.basename(params.llm_model)}.pkl"
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
                llm_gen_db_keys = llm_gen_db.keys()
                idx += num_samples
                logging.info(f"Loaded {num_samples} samples.")

        with executor_context as executor:
            futures = []
            for batch_idx, batch in enumerate(test_dl):
                texts = batch["supervisions"]["text"]
                cuts = batch["supervisions"]["cut"]
                if params.use_multiprocessing:
                    idx += process_multi_samples(executor,
                                                 params.num_processes,
                                                 futures, apikey, params,
                                                 cuts, texts, llm_gen_db,
                                                 hyp_gen_db, keys_in_wer_range)
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
                    print("num samples: " + str(len(llm_gen_db_from_pickle.entries)))
                    if debug_idx == 1:
                        import sys
                        sys.exit(0)
                    debug_idx += 1

                if batch_idx % 100 == 0:
                    logging.info(f"Batch {batch_idx} is done. {idx} samples are done.")

                if idx > save_interval:
                    logging.info(f"Saving the DB to {llm_gen_db_file}. {idx} samples are done.")
                    with open(llm_gen_db_file, "wb") as f:
                        pickle.dump(llm_gen_db, f)
                    save_interval += params.save_interval
                    while idx > save_interval:
                        save_interval += params.save_interval

            # final save
            with open(llm_gen_db_file, "wb") as f:
                logging.info(f"Saving the DB to {llm_gen_db_file}. {idx} samples are done.")
                pickle.dump(llm_gen_db, f)

if __name__ == "__main__":
    main()

