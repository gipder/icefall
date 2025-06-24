from pypinyin import pinyin, Style
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans

def pinyin_to_vector(pinyin_str, max_len=6):
    pinyin_str = pinyin_str.lower()
    chars = "abcdefghijklmnopqrstuvwxyz123456789"
    char_to_idx = {c: i for i, c in enumerate(chars)}

    vec = np.zeros((max_len, len(chars)), dtype=int)
    for i, ch in enumerate(pinyin_str[:max_len]):
        if ch in char_to_idx:
            vec[i, char_to_idx[ch]] = 1
    return vec.flatten()


exceptions = ("<blk>", "<unk>", "<sos/eos>", "#0", "#1")
token_file = "data/lang_char/tokens.txt"
token_dict = {}
with open(token_file, 'r', encoding='utf-8') as file:
    tokens = file.read().splitlines()

token_dict = {}
for l in tokens:
    ll = l.split(' ')
    key = ll[0]
    value = int(ll[-1])
    #if key in exceptions:
    #    continue
    token_dict[key] = value

pinyins = []
for token in token_dict.keys():
    char = token[0]
    py = pinyin(char, style=Style.NORMAL, strict=False)
    if py and py[0]:
        pinyins.append(py[0][0])
    else:
        pinyins.append("")

vectors = np.array([pinyin_to_vector(py) for py in pinyins])

kmeans = KMeans(n_clusters=20, random_state=0)
labels = kmeans.fit_predict(vectors)
labels = labels + 1  # Adjust labels to start from 1
print(f"Labels: {labels}")
clusters = {}
clusters[0] = []  # Cluster for exceptions
for label, token in zip(labels, token_dict.keys()):
    if label not in clusters:
        clusters[label] = []

    if token in exceptions:
        clusters[0].append(token)
    else:
        clusters[label].append(token)

idx = 0
for cluster_id, tokens in clusters.items():
    print(f"Cluster {cluster_id}: {', '.join(tokens)}")
    idx += 1
    print("---")
print(f"Total clusters: {idx}")
# to make a dictionary that emits the group index
#reverse_clusters = dict((token, cluster_id) for cluster_id, tokens in clusters.items() for token in tokens)
reverse_clusters = {token: cluster_id for cluster_id, tokens in clusters.items() for token in tokens}
reverse_clusters2 = {token_dict[token]: cluster_id for cluster_id, tokens in clusters.items() for token in tokens}
print(f"{reverse_clusters=}")
print(f"{reverse_clusters2=}")
print(f"{len(reverse_clusters)=}")
print(f"{len(reverse_clusters2)=}")
print(f"{reverse_clusters2[0]=}")
print(f"{reverse_clusters2[1]=}")
print(f"{reverse_clusters2[2]=}")
print(f"{reverse_clusters2[4336]=}")
print(f"{reverse_clusters2[4337]=}")
#print(f"{token_dict=}")
#reverse_clusters = dict(zip(clusters.values(), clusters.keys()))

#idx = 0
#for token, cluster_id in reverse_clusters.items():
#    print(f"{token} belongs to cluster {cluster_id}")
#    idx += 1
#
#print(f"{idx=}")

import sys
sys.exit(0)
#groups = defaultdict(token_dict.keys())
groups = defaultdict(list)

for ch in token_dict.keys():
    py = pinyin(ch, style=Style.NORMAL, strict=False)
    if py in py[0]:
        initial = py[0][0][0]
        print(f"{ch} - {py[0][0]}")
    else:
        initial = "#"
    groups[initial].append(ch)

sorted_groups = dict(sorted(groups.items()))

limited_gropus = dict(list(sorted_groups.items())[:50])

idx = 0
for initial, chars_group in limited_gropus.items():
    print(f"{idx} - {initial}: {', '.join(chars_group)}")
    idx += 1
