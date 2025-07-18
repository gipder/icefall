import torch
import k2
from pypinyin import pinyin, Style
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
import re
import copy
import logging

def map_to_table(y, N, num_vocabs=4336, blank_id=0, unk_id=2):
    """
    y: k2.RaggedTensor, values in range 0~4335 # list
    N: total number of mapping buckets (including 0)
    blank_id
    Returns: mapped LongTensor of shape (B, U)
    """
    mapped = list()
    for x in y.tolist():
        x = torch.tensor(x, dtype=torch.long)
        x_mapped = torch.zeros_like(x)

        # Case 1: 0~2 → 0
        mask_zero = (x <= unk_id)
        x_mapped = x_mapped.masked_fill(mask_zero, 0)

        # Case 2: 3~4335 → scaled to 1 ~ N-1
        above_unk = unk_id + 1
        mask_scale = (x >= above_unk)
        x_clipped = torch.clamp(x, min=above_unk, max=num_vocabs-1)
        x_scaled = (x_clipped - above_unk).float() / (num_vocabs-1-above_unk) * (N - unk_id) + 1
        x_scaled = x_scaled.round().long()
        x_mapped = x_mapped.masked_scatter(mask_scale, x_scaled[mask_scale])
        mapped.append(x_mapped.tolist())

    return mapped


def pinyin_to_vector(pinyin_str, max_len=6):
    pinyin_str = pinyin_str.lower()
    chars = "abcdefghijklmnopqrstuvwxyz123456789"
    char_to_idx = {c: i for i, c in enumerate(chars)}

    vec = np.zeros((max_len, len(chars)), dtype=int)
    for i, ch in enumerate(pinyin_str[:max_len]):
        if ch in char_to_idx:
            vec[i, char_to_idx[ch]] = 1
    return vec.flatten()


def cluster_tokens(tokens=dict(),
                   num_clusters=10,
                   exceptions=("<blk>", "<unk>", "<sos/eos>", "#0", "#1", "#2"),
                   blank_id=0,
                   random_state=42
                   ):
    """
    Clusters tokens based on their pinyin representations.
    Args:
        tokens (dict): Dictionary mapping token strings to their IDs.
        num_clusters (int): Number of clusters to form.
        exceptions (Tuple or List): List of tokens that should be treated as exceptions.
        blank_id (int): ID for the blank token.
        random_state (int): Random state for reproducibility.
    Returns:
        dict: A dictionary mapping IDs to their cluster IDs.
    """
    pinyins = []
    for token in tokens.keys():
        char = token[0]
        py = pinyin(char, style=Style.NORMAL, strict=False)
        if py and py[0]:
            pinyins.append(py[0][0])
        else:
            pinyins.append("")

    vectors = np.array([pinyin_to_vector(py) for py in pinyins])

    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    labels = kmeans.fit_predict(vectors)
    labels = labels + 1  # Adjust labels to start from 1

    clusters = {}
    clusters[blank_id] = []  # Cluster for exceptions
    for label, token in zip(labels, tokens.keys()):
        if label not in clusters:
            clusters[label] = []

        if token in exceptions:
            clusters[blank_id].append(token)
        else:
            clusters[label].append(token)

    reverse_clusters = {
        tokens[token]: cluster_id
        for cluster_id, token_list in clusters.items()
        for token in token_list
    }

    return reverse_clusters

def cluster_tokens(tokens=dict(),
                   num_clusters=10,
                   exceptions=("<blk>", "<unk>", "<sos/eos>", "#0", "#1", "#2"),
                   blank_id=0,
                   random_state=42
                   ):
    """
    Clusters tokens based on their pinyin representations.
    Args:
        tokens (dict): Dictionary mapping token strings to their IDs.
        num_clusters (int): Number of clusters to form.
        exceptions (Tuple or List): List of tokens that should be treated as exceptions.
        blank_id (int): ID for the blank token.
        random_state (int): Random state for reproducibility.
    Returns:
        dict: A dictionary mapping IDs to their cluster IDs.
    """
    pinyins = []
    for token in tokens.keys():
        char = token[0]
        py = pinyin(char, style=Style.NORMAL, strict=False)
        if py and py[0]:
            pinyins.append(py[0][0])
        else:
            pinyins.append("")

    vectors = np.array([pinyin_to_vector(py) for py in pinyins])

    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    labels = kmeans.fit_predict(vectors)
    labels = labels + 1  # Adjust labels to start from 1

    clusters = {}
    clusters[blank_id] = []  # Cluster for exceptions
    for label, token in zip(labels, tokens.keys()):
        if label not in clusters:
            clusters[label] = []

        if token in exceptions:
            clusters[blank_id].append(token)
        else:
            clusters[label].append(token)

    reverse_clusters = {
        tokens[token]: cluster_id
        for cluster_id, token_list in clusters.items()
        for token in token_list
    }

    return reverse_clusters


def ipa_to_vector(ipa_text_list=list(), ipa_set=set(),
                  max_len=15,
                  exceptions=("<blk>", "<unk>",
                              "<sos/eos>",
                              "#0", "#1", "#2"),
                  ):
    ipa_to_idx = {ip: i for i, ip in enumerate(ipa_set)}
    dim = len(ipa_set)
    vec = np.zeros((max_len, dim), dtype=int)
    for i, ip in enumerate(ipa_text_list[:max_len]):
        vec[i, ipa_to_idx[ip]] = 1
        """
        if ip in ipa_to_idx:
            vec[i, ipa_to_idx[ip]+bias] = 1
        elif ip in exceptions:
            print("GARBAGE " * 100 )
            vec[i, 0] = 1
        else:
            print("GARBAGE2 " * 100)
            vec[i, 1] = 1
        """
    return vec.flatten()


def cluster_tokens_using_ipa(tokens=dict(),
                             num_clusters=10,
                             exceptions=("<blk>", "<unk>",
                                         "<sos/eos>"),
                             blank_id=0,
                             punc_symbols=(".", ",", "!",
                                           ":", ";", "?",
                                           "▁", "'", "\"",
                                           "-"),
                             punc_id=1,
                             random_state=42,
                             front_space='▁',
                             skip_symbols=("#0", "#1", "#2")# it is different with '_'
                             ):
    """
    Clusters tokens based on their IPA representations.
    Args:
        tokens (dict): A dictionary contains BPE tokens.
        num_clusters (int): Number of clusters to form.
        exceptions (Tuple or List): List of tokens that should be treated as exceptions.
        blank_id (int): ID for the blank token.
        punc_symbols (Tuple or List): List of punctuation symbols in BPE tokens.
        punc_id (int): ID for punctuation symbols
        random_state (int): Random state for reproducibility.
        front_space (str): a front space symbol in BPE tokens
    Returns:
        dict: A dictionary mapping IDs to their cluster IDs.
              key: original token id, value: group(cluster) id
    """
    kor_and_digit = re.compile(r'[가-힣\d]')

    espeak_ko = EspeakBackend(language='kok', language_switch='remove-flags')
    espeak_en = EspeakBackend(language='en-us')
    separator = Separator(phone=' ', word='=', syllable='|')

    ipa_set = set()
    ipa_converted_tokens = list()

    eng_tokens = list()
    kor_tokens = list()
    for token in tokens:
        if (
            token in exceptions
            or token in punc_symbols
            or token in skip_symbols
        ):
            continue
        else:
            modified_token = token.replace(front_space, '')
            if kor_and_digit.search(modified_token):
                kor_tokens.append(modified_token)
            else:
                eng_tokens.append(modified_token)

    kor_ipas = espeak_ko.phonemize(kor_tokens, strip=True,
                                   separator=separator)
    eng_ipas = espeak_en.phonemize(eng_tokens, strip=True,
                                   separator=separator)
    ipas = eng_ipas + kor_ipas

    for ipa in ipas:
        ipa_list = ipa.strip().split()
        if len(ipa_list) == 0:
            print("This is NOT possible")
            import sys
            sys.exit(0)

        ipa_converted_tokens.append(ipa_list)

        for i in ipa_list:
            ipa_set.add(i)

    vectors = np.array([ipa_to_vector(ipa_text_list=ipa_list,
                                      ipa_set=ipa_set,
                                      exceptions=exceptions)
                        for ipa_list in ipa_converted_tokens])

    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)

    labels = kmeans.fit_predict(vectors)
    bias = 2  # because of blank symbols and punc symbols
    labels = labels + bias  # Adjust labels to start from 2

    clusters = {}
    clusters[blank_id] = []
    clusters[punc_id] = []
    tokens_clone = copy.deepcopy(tokens)
    for key in tokens.keys():
        if key in exceptions:
            clusters[blank_id].append(key)
            del tokens_clone[key]
        elif key in punc_symbols:
            clusters[punc_id].append(key)
            del tokens_clone[key]
        elif key in skip_symbols:
            del tokens_clone[key]

    for label, token in zip(labels, tokens_clone.keys()):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(token)

    logging.info(f"Clustering is done: {clusters}")
    reverse_clusters = {
        tokens[token]: cluster_id
        for cluster_id, token_list in clusters.items()
        for token in token_list
    }

    return reverse_clusters


def cluster_tokens_using_ipa_by_language(
    tokens=dict(),
    num_language_clusters=list(),
    last_eng_id=499,
    exceptions=("<blk>", "<unk>",
                "<sos/eos>"),
    blank_id=0,
    punc_symbols=(".", ",", "!",
                  ":", ";", "?",
                  "▁", "'", "\"",
                  "-", "%", "&"),
    punc_id=1,
    random_state=42,
    front_space='▁',
    skip_symbols=("#0", "#1", "#2")# it is different with '_'
):
    """
    Clusters tokens based on their IPA representations.
    Args:
        tokens (dict): A dictionary contains BPE tokens.
        num_language_clusters (list): Numbers of language clusters ([eng, kor])
                                      except for blank_id 0 and punc_id 1
        last_eng_id: The last english token ID
        exceptions (Tuple or List): List of tokens that should be treated as exceptions.
        blank_id (int): ID for the blank token.
        punc_symbols (Tuple or List): List of punctuation symbols in BPE tokens.
        punc_id (int): ID for punctuation symbols
        random_state (int): Random state for reproducibility.
        front_space (str): a front space symbol in BPE tokens
    Returns:
        dict: A dictionary mapping IDs to their cluster IDs.
              key: original token id, value: group(cluster) id
    """
    kor_and_digit = re.compile(r'[가-힣a-z\d]')

    espeak_ko = EspeakBackend(language='kok', language_switch='remove-flags')
    espeak_en = EspeakBackend(language='en-us')
    separator = Separator(phone=' ', word='=', syllable='|')

    ipa_set = set()
    eng_ipa_converted_tokens = list()
    kor_ipa_converted_tokens = list()

    eng_tokens = list()
    kor_tokens = list()
    for token in tokens:
        if (
            token in exceptions
            or token in punc_symbols
            or token in skip_symbols
        ):
            continue
        else:
            modified_token = token.replace(front_space, '')
            if tokens[token] > last_eng_id:
                kor_tokens.append(modified_token)
            else:
                eng_tokens.append(modified_token)

    kor_ipas = espeak_ko.phonemize(kor_tokens, strip=True,
                                   separator=separator)
    eng_ipas = espeak_en.phonemize(eng_tokens, strip=True,
                                   separator=separator)

    # for eng ipa
    for eng_ipa in eng_ipas:
        eng_ipa_list = eng_ipa.strip().split()
        if len(eng_ipa_list) == 0:
            print("This is NOT possible")
            import sys
            sys.exit(0)

        eng_ipa_converted_tokens.append(eng_ipa_list)

        for i in eng_ipa_list:
            ipa_set.add(i)

    eng_vectors = np.array([ipa_to_vector(ipa_text_list=eng_ipa_list,
                                          ipa_set=ipa_set,
                                          exceptions=exceptions)
                            for eng_ipa_list in eng_ipa_converted_tokens])

    eng_kmeans = KMeans(n_clusters=num_language_clusters[0], random_state=random_state)
    eng_labels = eng_kmeans.fit_predict(eng_vectors)

    # for kor ipa
    for kor_ipa in kor_ipas:
        kor_ipa_list = kor_ipa.strip().split()
        if len(kor_ipa_list) == 0:
            print("This is NOT possible")
            import sys
            sys.exit(0)

        kor_ipa_converted_tokens.append(kor_ipa_list)

        for i in kor_ipa_list:
            ipa_set.add(i)

    kor_vectors = np.array([ipa_to_vector(ipa_text_list=kor_ipa_list,
                                          ipa_set=ipa_set,
                                          exceptions=exceptions)
                            for kor_ipa_list in kor_ipa_converted_tokens])

    kor_kmeans = KMeans(n_clusters=num_language_clusters[1], random_state=random_state)
    kor_labels = kor_kmeans.fit_predict(kor_vectors)

    # indexing bias + eng_label + kor_label
    bias = 2  # because of blank symbols and punc symbols
    eng_labels = eng_labels + bias
    kor_labels = kor_labels + bias + num_language_clusters[0]
    labels = np.concatenate((eng_labels, kor_labels))

    clusters = {}
    clusters[blank_id] = []
    clusters[punc_id] = []
    tokens_clone = copy.deepcopy(tokens)
    for key in tokens.keys():
        if key in exceptions:
            clusters[blank_id].append(key)
            del tokens_clone[key]
        elif key in punc_symbols:
            clusters[punc_id].append(key)
            del tokens_clone[key]
        elif key in skip_symbols:
            del tokens_clone[key]

    for label, token in zip(labels, tokens_clone.keys()):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(token)

    logging.info(f"Clustering is done: {clusters}")
    reverse_clusters = {
        tokens[token]: cluster_id
        for cluster_id, token_list in clusters.items()
        for token in token_list
    }

    return reverse_clusters
