import torch
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer

import argparse
import logging
import math
from pathlib import Path

import torch
from dataset import get_dataloader
from train import get_params

import torch.nn.functional as F
from icefall.checkpoint import average_checkpoints, load_checkpoint
from icefall.transformer_lm.model import TransformerLM
from icefall.utils import AttributeDict, setup_logger, str2bool, make_pad_mask

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--llm",
        type=str,
        help="pre-trained large language model, GPT2 or something",
        default="gpt2",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="transformer_lm/exp_full_libri_16layer_maxlen200_8gpu",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of RNN layers the model",
    )

    parser.add_argument(
        "--lm-data",
        type=str,
        help="Path to the LM test data for computing perplexity",
        default="transformer_lm/libri_lm_training_bpe500/sorted_lm_data-test.pt",
    )

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.lm_data = Path(args.lm_data)

    params = get_params()
    params.update(vars(args))
    params.max_sent_len = 200
    params.sos_id = 50256
    params.eos_id = 50256
    params.blank_id = -100

    setup_logger(f"{params.exp_dir}/log-ppl-{params.llm}/")
    logging.info("Computing perplexity started")
    logging.info(params)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    logging.info("About to create model")
    # GPT 모델과 토크나이저 초기화
    model = GPT2LMHeadModel.from_pretrained(params.llm)
    tokenizer = GPT2Tokenizer.from_pretrained(params.llm)

    model.to(device)
    model.eval()

    num_param = sum([p.numel() for p in model.parameters()])
    num_param_requires_grad = sum(
        [p.numel() for p in model.parameters() if p.requires_grad]
    )

    logging.info(f"Number of model parameters: {num_param}")
    logging.info(
        f"Number of model parameters (requires_grad): "
        f"{num_param_requires_grad} "
        f"({num_param_requires_grad/num_param_requires_grad*100}%)"
    )


    logging.info(f"Loading LM test data from {params.lm_data}")
    test_dl = get_dataloader(
        filename=params.lm_data,
        is_distributed=False,
        params=params,
        llm=True,
    )

    tot_loss = 0.0
    num_tokens = 0
    num_sentences = 0
    for batch_idx, batch in enumerate(test_dl):
        x, y, sentence_lengths = batch
        x = x.to(device)
        y = y.to(device)
        print(f"x: {x}")
        print(f"y: {y}")
        print(f"sentence_lengths: {sentence_lengths}")

        with torch.no_grad():
            outputs = model(x, labels=y)
            """
            logits = outputs.logits

            nll_loss = F.cross_entropy(
                logits.reshape(-1, model.config.vocab_size), y.reshape(-1), reduction="none"
            )

            mask = make_pad_mask(sentence_lengths).reshape(-1).to(device)
            nll_loss.masked_fill_(mask, 0)
            """
            loss = outputs.loss

        #loss = nll_loss.sum().cpu().item()
        tot_loss += loss
        num_tokens += sentence_lengths.sum().cpu().item()
        num_sentences += x.size(0)

    ppl = math.exp(tot_loss / num_sentences)
    logging.info(
        f"total nll: {tot_loss}, num tokens: {num_tokens}, "
        f"num sentences: {num_sentences}, ppl: {ppl:.3f}"
    )


if __name__ == "__main__":
    main()

