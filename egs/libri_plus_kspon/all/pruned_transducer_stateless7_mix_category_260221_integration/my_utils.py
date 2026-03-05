import torch
import k2
import argparse
import os
from pypinyin import pinyin, Style
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
import re
import copy
import logging
import random
import pprint


def save_tsne_visualization(
    embedding_vectors,
    labels,
    token_ids,    
    visualization_top_k=10,
    random_state=42,
    save_visualization_dir=None,
    cmap="tab10",
    cluster_type="embedding",
):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import matplotlib

    # Set font to support CJK characters
    try:
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    except:
        pass
    matplotlib.rcParams['axes.unicode_minus'] = False

    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=30, max_iter=1000)
    reduced_embeddings = tsne.fit_transform(embedding_vectors)

    # Get cluster sizes and select top-k clusters
    cluster_sizes = {}
    for label in labels:
        cluster_sizes[label] = cluster_sizes.get(label, 0) + 1

    # Sort clusters by size and get top-k
    top_k_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)[:visualization_top_k]
    top_k_cluster_ids = set([c[0] for c in top_k_clusters])

    # Filter embeddings and labels for visualization
    viz_mask = np.isin(labels, list(top_k_cluster_ids))
    viz_embeddings = reduced_embeddings[viz_mask]
    viz_labels = labels[viz_mask]
    viz_token_ids = np.array(token_ids)[viz_mask]

    logging.info(f"Visualizing top {visualization_top_k} clusters out of {len(cluster_sizes)}")
    for cluster_id, size in top_k_clusters:
        logging.info(f"  Cluster {cluster_id}: {size} tokens")

    # Create visualization
    plt.figure(figsize=(16, 12))
    scatter = plt.scatter(viz_embeddings[:, 0], viz_embeddings[:, 1],
                         c=viz_labels, cmap=cmap, s=100, alpha=0.8,
                         edgecolors='black', linewidth=0.5)
    
    # Create legend with cluster sizes inside the plot
    legend_labels = [f"Top-{i+1} (n = {size})" for i, (cluster_id, size) in enumerate(top_k_clusters)]
    legend_handles = [plt.scatter([], [], c=[plt.cm.get_cmap(cmap)(i / len(top_k_clusters))], 
                                 s=250, alpha=0.8, edgecolors='black', linewidth=1)
                     for i in range(len(top_k_clusters))]
    plt.legend(legend_handles, legend_labels, loc='upper left', fontsize=15, 
              title='Cluster Sizes', title_fontsize=16, framealpha=0.95)
    
    #plt.title(f"Token Embedding Clusters (Top {visualization_top_k} by Size)",
    #         fontsize=16, fontweight='bold', pad=20)
    #plt.xlabel("t-SNE Dimension 1", fontsize=12, fontweight='bold')
    #plt.ylabel("t-SNE Dimension 2", fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')

    filename = f"token_{cluster_type}_clusters_top{visualization_top_k}.png"
    if save_visualization_dir:
        os.makedirs(save_visualization_dir, exist_ok=True)
        filename = os.path.join(save_visualization_dir, filename)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved token embedding cluster visualization to {filename}")

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

    str_clusters = pprint.pformat(clusters, compact=True)
    logging.info(f"Clustering is done: {str_clusters}")
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

    str_clusters = pprint.pformat(clusters, compact=True)
    logging.info(f"Clustering is done: {str_clusters}")
    reverse_clusters = {
        tokens[token]: cluster_id
        for cluster_id, token_list in clusters.items()
        for token in token_list
    }

    return reverse_clusters


def cluster_tokens_using_ipa_by_list(
    tokens=dict(),
    num_language_clusters=list(),
    last_first_lang_id=499,
    save_visualization=False,
    visualization_top_k=10,
    save_visualization_dir=None,
    cmap="tab10",
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
        num_language_clusters (list): Numbers of language clusters ([eng, kor, chn])
                                      except for blank_id 0 and punc_id 1
        last_first_lang_id: The last first language (english) token ID
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
    # Pattern matchers for each language
    korean_pattern = re.compile(r'[가-힣]')
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')

    # Initialize espeak backends for each language
    espeaks = list()
    espeak_en = EspeakBackend(language='en-us')
    espeak_ko = EspeakBackend(language='kok', language_switch='remove-flags')
    espeak_zh = EspeakBackend(language='cmn', language_switch='remove-flags')
    espeaks.append(espeak_en)
    espeaks.append(espeak_ko)
    espeaks.append(espeak_zh)
    separator = Separator(phone=' ', word='=', syllable='|')

    ipa_set = set()
    tokens_list = list()
    original_tokens_list = list()  # Track original tokens for visualization
    bucket_langs = list()
    num_lang = len(num_language_clusters)
    for i in range(num_lang):
        tokens_list.append(list())
        original_tokens_list.append(list())
        bucket_langs.append(set())

    for token in tokens:
        if (
            token in exceptions
            or token in punc_symbols
            or token in skip_symbols
        ):
            continue
        else:
            modified_token = token.replace(front_space, '').replace(' ', '').strip()
            
            # Classify token by language based on character patterns
            if chinese_pattern.search(modified_token):
                # Chinese tokens (index 2)
                if num_lang >= 3:
                    tokens_list[2].append(modified_token)
                    original_tokens_list[2].append(token)
                    bucket_langs[2].add("zh")
                else:
                    tokens_list[-1].append(modified_token)
                    original_tokens_list[-1].append(token)
                    bucket_langs[-1].add("zh")
            elif korean_pattern.search(modified_token):
                # Korean tokens (index 1)
                if num_lang >= 2:
                    tokens_list[1].append(modified_token)
                    original_tokens_list[1].append(token)
                    bucket_langs[1].add("ko")
                else:
                    tokens_list[-1].append(modified_token)
                    original_tokens_list[-1].append(token)
                    bucket_langs[-1].add("ko")
            else:
                # English and other tokens (index 0)
                tokens_list[0].append(modified_token)
                original_tokens_list[0].append(token)
                bucket_langs[0].add("en")

    ipas_list = list()
    for i in range(num_lang):
        # Choose backend based on detected language in the bucket
        if "zh" in bucket_langs[i]:
            backend = espeak_zh
        elif "ko" in bucket_langs[i]:
            backend = espeak_ko
        else:
            backend = espeak_en

        ipas_list.append(
            list(
                backend.phonemize(tokens_list[i], strip=True,
                                  separator=separator)
            )
        )
    
    labels = list()
    ipa_converted_tokens_list = list()
    for i in range(num_lang):
        ipa_converted_tokens_list.append(list())

    for i in range(num_lang):
        for ipa in ipas_list[i]:
            ipa_list = ipa.strip().split()
            if len(ipa_list) == 0:
                print("This is NOT possible")
                import sys
                sys.exit(0)

            ipa_converted_tokens_list[i].append(ipa_list)
            for j in ipa_list:
                ipa_set.add(j)

    # vectorize
    for i in range(num_lang):
        vectors = np.array([ipa_to_vector(ipa_text_list=ipa_list,
                                          ipa_set=ipa_set,
                                          exceptions=exceptions)
                            for ipa_list in ipa_converted_tokens_list[i]])
        kmeans = KMeans(n_clusters=num_language_clusters[i], random_state=random_state)
        labels.append(kmeans.fit_predict(vectors))

    if save_visualization:
        # support only 1 language for now
        save_tsne_visualization(
            embedding_vectors=vectors,
            labels=labels[0],
            token_ids=[tokens[token] for token in original_tokens_list[0]],
            #id_to_token={tokens[token]: token for token in original_tokens_list[0]},
            visualization_top_k=visualization_top_k,
            random_state=random_state,
            save_visualization_dir=save_visualization_dir,
            cmap=cmap,
            cluster_type="ipa",
        )


    # indexing bias + eng_label + kor_label
    bias = 2  # because of blank symbols and punc symbols
    biased_labels = list()
    for i in range(num_lang):
        biased_labels.append(labels[i] + bias)
        if i > 0:
            biased_labels[i] += num_language_clusters[i-1]
    labels = np.concatenate(biased_labels)

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

    str_clusters = pprint.pformat(clusters, compact=True)
    logging.info(f"Clustering is done: {str_clusters}")
    reverse_clusters = {
        tokens[token]: cluster_id
        for cluster_id, token_list in clusters.items()
        for token in token_list
    }

    return reverse_clusters


def cluster_random_tokens(
    tokens=dict(),
    num_language_clusters=list(),
    last_first_lang_id=499,
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
        last_first_lang_id: The last first language (english) token ID
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

    rng = random.Random(random_state)
    vocab_shuf = []

    for token in tokens.keys():
        if token not in exceptions and token not in punc_symbols:
            vocab_shuf.append(token)
    rng.shuffle(vocab_shuf)
    #print(f"{type(vocab_shuf)=}")
    num_clusters = sum(num_language_clusters)
    #print(f"{num_clusters=}")

    # 최소 한번 이상 나오는 것들
    # +2 는 blank 와 punk symbol 관련
    assignments = [i+2 for i in range(num_clusters)]
    assignments += [
        rng.randrange(2, num_clusters+2) for _ in range(len(vocab_shuf) - num_clusters)
    ]
    rng.shuffle(assignments)

    #print(f"{len(vocab_shuf)=}")
    #print(f"{len(assignments)=}")

    random_cluster_ids = {}
    for token, cluster_id in zip(vocab_shuf, assignments):
        random_cluster_ids[token] = cluster_id
    sorted_cluster_ids = sorted(random_cluster_ids.items(), key=lambda x: x[1])
    str_sorted_cluster_ids = pprint.pformat(sorted_cluster_ids, compact=True)
    logging.info(f"Clustering is done: {str_sorted_cluster_ids}")

    reverse_clusters = {}
    for token in tokens.keys():
        if token in exceptions:
            reverse_clusters[tokens[token]] = blank_id
        elif token in punc_symbols:
            reverse_clusters[tokens[token]] = punc_id
        else:
            reverse_clusters[tokens[token]] = int(random_cluster_ids[token])

    return reverse_clusters

def cluster_tokens_using_embedding(
            token_dict,
            num_category_list,            
            embedding_file,
            save_visualization=False,
            visualization_top_k=10,
            save_visualization_dir=None,
            cmap="tab10",
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
            skip_symbols=("#0", "#1", "#2")
    ):
    """
    Clusters tokens based on their embedding representations.
    Args:
        token_dict (dict): A dictionary contains BPE tokens.
        num_category_list (list): Numbers of language clusters ([eng, kor, chn])
        embedding_file (str): Path to the embedding file.
    Returns:
        dict: A dictionary mapping IDs to their cluster IDs.
              key: original token id, value: group(cluster) id
    """
    # load embeddings
    # embedding_file format:
    # params.token_embeddings[token_id] = {
    #                    'audio_embedding': tensor,
    #                    'text_embedding': tensor,
    #                }
    embeddings = torch.load(embedding_file)
    logging.info(f"Loaded embeddings from {embedding_file}")
    
    # Number of clusters (assume single language for now)
    num_clusters = num_category_list[0] if len(num_category_list) > 0 else 100
    
    # Prepare token IDs and their embeddings
    token_ids_with_emb = []
    embedding_vectors = []
    token_ids_without_emb = []
    
    for token, token_id in token_dict.items():
        # Skip special tokens
        if (token in exceptions or 
            token in punc_symbols or 
            token in skip_symbols):
            continue
            
        # Check if embedding exists for this token_id
        if token_id in embeddings:
            audio_emb = embeddings[token_id]['audio_embedding']
            text_emb = embeddings[token_id]['text_embedding']
            
            # Normalize embeddings before concatenation
            audio_emb_norm = torch.nn.functional.normalize(audio_emb, p=2, dim=-1)
            text_emb_norm = torch.nn.functional.normalize(text_emb, p=2, dim=-1)
            
            # Concatenate normalized audio and text embeddings
            concat_emb = torch.cat([audio_emb_norm, text_emb_norm], dim=-1)
            #concat_emb = torch.cat([text_emb_norm], dim=-1)
            embedding_vectors.append(concat_emb.cpu().numpy())
            token_ids_with_emb.append(token_id)
        else:
            token_ids_without_emb.append(token_id)
    
    logging.info(f"Tokens with embeddings: {len(token_ids_with_emb)}")
    logging.info(f"Tokens without embeddings: {len(token_ids_without_emb)}")
    
    # Convert to numpy array
    embedding_vectors = np.array(embedding_vectors)
    
    # Perform K-means clustering on tokens with embeddings
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    labels = kmeans.fit_predict(embedding_vectors)
    
    # Create reverse mapping for logging: token_id -> token
    id_to_token = {token_id: token for token, token_id in token_dict.items()}
    
    # save t-SNE visualization
    if save_visualization:
        save_tsne_visualization(
            embedding_vectors=embedding_vectors,
            labels=labels,
            token_ids=token_ids_with_emb,
            visualization_top_k=visualization_top_k,
            random_state=random_state,
            save_visualization_dir=save_visualization_dir,
            cmap=cmap,
            cluster_type="embedding",
        )
    
    # Adjust labels to start from 2 (0 for blank, 1 for punctuation)
    bias = 2
    labels = labels + bias
    
    # Create clusters dictionary
    clusters = {}
    clusters[blank_id] = []
    clusters[punc_id] = []
    
    # Assign clustered tokens
    for token_id, label in zip(token_ids_with_emb, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(token_id)
    
    # Handle tokens without embeddings - assign to random clusters
    rng = random.Random(random_state)
    for token_id in token_ids_without_emb:
        # Randomly assign to one of the existing clusters (bias to num_clusters+bias-1)
        random_label = rng.randint(bias, num_clusters + bias - 1)
        if random_label not in clusters:
            clusters[random_label] = []
        clusters[random_label].append(token_id)
    
    # Build reverse mapping: token_id -> cluster_id
    reverse_clusters = {}
    
    for token, token_id in token_dict.items():
        if token in exceptions:
            reverse_clusters[token_id] = blank_id
        elif token in punc_symbols:
            reverse_clusters[token_id] = punc_id
        elif token in skip_symbols:
            # Skip symbols are not included in the output
            continue
        else:
            # Find which cluster this token_id belongs to
            for cluster_id, token_list in clusters.items():
                if token_id in token_list:
                    reverse_clusters[token_id] = cluster_id
                    break
    
    # Convert clusters to show actual tokens instead of token_ids for logging
    clusters_for_log = {}
    for cluster_id, token_id_list in clusters.items():
        clusters_for_log[cluster_id] = [id_to_token.get(tid, f"<unknown:{tid}>") 
                                        for tid in token_id_list]
    
    str_clusters = pprint.pformat(clusters_for_log, compact=True)
    logging.info(f"Clustering is done: {str_clusters}")
    
    return reverse_clusters


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Save token embedding cluster visualization (t-SNE)"
    )
    parser.add_argument(
        "--tokens",
        required=True,
        help="Path to tokens.txt (token id per line)",
    )
    parser.add_argument(
        "--embedding",
        required=True,
        help="Path to embedding file (torch.load)",
    )
    parser.add_argument(
        "--num-category-list",
        default="100",
        help="Comma-separated cluster counts, e.g. '100' or '50,50,50'",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-k clusters by size to visualize",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Directory to save the visualization image",
    )
    parser.add_argument(
        "--cmap",
        default="tab10",
        help="Matplotlib colormap name for visualization (default: tab10)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--cluster-type",
        type=str,
        choices=["embedding", "ipa"],
        default="embedding",
        help="Type of clustering to use (embedding or ipa)",
    )

    args = parser.parse_args()

    token_dict = {}
    with open(args.tokens, "r", encoding="utf-8") as f:
        tokens_lines = f.read().splitlines()
    for line in tokens_lines:
        token = line.split(" ")[0]
        idx = line.split(" ")[-1]
        token_dict[token] = int(idx)

    num_category_list = [int(x) for x in args.num_category_list.split(",") if x.strip()]

    if args.cluster_type == "embedding":
        cluster_tokens_using_embedding(
            token_dict=token_dict,
            num_category_list=num_category_list,
            embedding_file=args.embedding,
            save_visualization=True,
            visualization_top_k=args.top_k,
            save_visualization_dir=args.out_dir,
            random_state=args.random_state,
            cmap=args.cmap,
        )
    elif args.cluster_type == "ipa":
        cluster_tokens_using_ipa_by_list(
            tokens=token_dict,
            num_language_clusters=num_category_list,
            save_visualization=True,
            visualization_top_k=args.top_k,
            save_visualization_dir=args.out_dir,
            random_state=args.random_state,
            cmap=args.cmap,
        )
