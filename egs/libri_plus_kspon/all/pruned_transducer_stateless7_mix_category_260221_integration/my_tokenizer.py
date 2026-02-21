from concurrent.futures import ThreadPoolExecutor
import os

class MyTokenizer:
    def __init__(self, vocab_file=""):
        self.vocab_dict = dict()  # {token_id: token}
        self.reversed_vocab_dict = dict()  # {token: token_id}
        self.vocab_file = vocab_file
        self.vocab_size = 0
        self.blank = '<blk>'
        self.sos = '<sos/eos>'
        self.eos = '<sos/eos>'
        self.unk = '<unk>'
        self.blank_id = 0
        self.sos_id = 1
        self.eos_id = 1
        self.unk_id = 2
        self.skip_tokens = ['#0', '#1']
        self._init_vocab()
        self.tokenize = self.build_tokenizer()

    def _init_vocab(self):
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            idx = 0
            for line in f:
                token = line.strip().split()[0]
                if token in self.skip_tokens:
                    continue
                self.vocab_dict[int(idx)] = token
                idx += 1
        self.vocab_size = len(self.vocab_dict)
        self.reversed_vocab_dict = {v: k for k, v in self.vocab_dict.items()}

    def build_tokenizer(self):
        tokens = list(self.vocab_dict.values())
        special_tokens = set([self.blank, self.eos, self.unk])
        usable_tokens = sorted([t for t in tokens if t not in special_tokens],
                               key=lambda x: -len(x))

        def tokenize(text):
            text = text.replace(' ', '▁')
            i = 0
            result = []
            while i < len(text):
                matched = False
                for token in usable_tokens:
                    if text[i:i+len(token)] == token:
                        result.append(token)
                        i += len(token)
                        matched = True
                        break
                if not matched:
                    result.append(self.unk)
                    i += 1
            return result

        return tokenize

    def encode(self, text):
        if isinstance(text, list):
            return [self.reversed_vocab_dict.get(token, self.unk) for token in text]
        tokens = self.tokenize(text)
        return [self.reversed_vocab_dict[token] for token in tokens]

    def encode_batch(self, texts, num_threads=1):
        num_threads = max(1, os.cpu_count())
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(self.encode, texts))
        return results

    def decode(self, ids, use_icefall=True):
        decoded = ''.join([self.vocab_dict.get(id, self.unk)
                           for id in ids]).replace('▁', ' ')
        if use_icefall:
            decoded = decoded.split(' ')
        return decoded

    def decode_batch(self, ids, use_icefall=True, num_threads=1):
        num_threads = max(1, os.cpu_count())
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(self.decode, ids))
        return results

    def return_token_list(self, ids):
        token_list = [self.vocab_dict.get(id, self.unk)
                           for id in ids]
        return token_list

    def return_token_list_batch(self, ids, num_threads=1):
        num_threads = max(1, os.cpu_count())
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(self.return_token_list, ids))
        return results


def remove_space_symbol(token):
    return token.replace('_', '')


def char_tokenizing(text):
    # 예: "안녕하세요 여러분" → ['_안', '녕', '하', '세', '요', '_여', '러', '분']
    result = []
    words = text.strip().split()
    for word in words:
        for i, ch in enumerate(word):
            if i == 0:
                result.append('_' + ch)  # 단어 시작
            else:
                result.append(ch)
    return result


def reconstruct_with_spacing(aligned_tokens):
    # '_안', '녕' → '안녕', '_여' → ' 여러분'
    result = ""
    for token in aligned_tokens:
        if token.startswith('_'):
            result += " " + token[1:]
        else:
            result += token
    return result.strip()


def get_norm_text(ref, hyp):
    refs = char_tokenizing(ref)
    hyps = char_tokenizing(hyp)
    rlen, hlen = len(refs), len(hyps)

    # 1. Edit distance 계산 (DP 테이블)
    scores = [[0] * (rlen + 1) for _ in range(hlen + 1)]

    for r in range(rlen + 1):
        scores[0][r] = r

    for h in range(1, hlen + 1):
        scores[h][0] = scores[h - 1][0] + 1

    for h in range(1, hlen + 1):
        for r in range(1, rlen + 1):
            hyp_nosp = remove_space_symbol(hyps[h - 1])
            ref_nosp = remove_space_symbol(refs[r - 1])
            sub_or_cor = scores[h - 1][r - 1] + (0 if hyp_nosp == ref_nosp else 1)
            ins = scores[h - 1][r] + 1
            delete = scores[h][r - 1] + 1
            scores[h][r] = min(sub_or_cor, ins, delete)

    # 2. Traceback & Alignment
    h, r = hlen, rlen
    hypnorm, refnorm = [], []

    while h > 0 or r > 0:
        if h == 0:
            last_h, last_r = h, r - 1
        elif r == 0:
            last_h, last_r = h - 1, r
        else:
            hyp_nosp = remove_space_symbol(hyps[h - 1])
            ref_nosp = remove_space_symbol(refs[r - 1])
            sub_or_cor = scores[h - 1][r - 1] + (0 if hyp_nosp == ref_nosp else 1)
            ins = scores[h - 1][r] + 1
            delete = scores[h][r - 1] + 1

            if sub_or_cor <= min(ins, delete):
                last_h, last_r = h - 1, r - 1
            elif ins < delete:
                last_h, last_r = h - 1, r
            else:
                last_h, last_r = h, r - 1

        chyp = hyps[last_h] if h != 0 and last_h != h else ''
        cref = refs[last_r] if r != 0 and last_r != r else ''
        h, r = last_h, last_r

        # 공백 정렬: 같은 글자면 ref의 공백 기준 사용
        if remove_space_symbol(chyp) == remove_space_symbol(cref):
            chyp = cref

        if cref or chyp:
            refnorm.append(cref)
            hypnorm.append(chyp)

    refnorm.reverse()
    hypnorm.reverse()

    # 3. ref 기준 공백을 hyp에 반영하여 복원
    ref_str = reconstruct_with_spacing(refnorm)
    hyp_str = reconstruct_with_spacing(hypnorm)

    return ref_str, hyp_str


def space_normalized_wer(ref=list(), hyp=list()):
    # make character level alignment
    print(''.join(ref))
    print(''.join(hyp))
    ref_char = ''.join(ref)
    hyp_char = ''.join(hyp)
    print(list(ref_char))
    print(list(hyp_char))


if __name__ == "__main__":
    my_tokenizer = MyTokenizer("tokens.txt")
    print("Vocabulary Size:", my_tokenizer.vocab_size)
    tokens = my_tokenizer.tokenize("안녕하세요 반갑습니다 THIS IS A SIMPLE TEST")
    print(tokens)
    print("Encoded:", my_tokenizer.encode("안녕하세요 반갑습니다 THIS IS A SIMPLE TEST"))
    print(my_tokenizer.encode(['안', '녕', '하', '세요', '▁반', '갑', '습니다', '▁THIS', '▁IS', '▁A', '▁S', 'IM', 'P', 'LE', '▁T', 'EST']))
    print("Decoded:", my_tokenizer.decode(my_tokenizer.encode("안녕하세요 반갑습니다 THIS IS A SIMPLE TEST")))

    # Multithread
    texts = ["안녕하세요 반갑습니다 THIS IS A SIMPLE TEST",
             "다음 문장은 멀티스레드로 인코딩됩니다",
             "이것은 테스트 문장입니다",
             "PYTHON 은 멀티스레딩을 지원합니다"]
    encoded_texts = my_tokenizer.encode_batch(texts)
    print("Batch Encoded:", encoded_texts)
    decoded_texts = my_tokenizer.decode_batch(encoded_texts)
    print("Batch Decoded:", decoded_texts)

    ref=['아닐껄', '그뭐지', '몰라', '뭔가', '근데', '그', '남자애가', '우리', '욕', '했을', '것', '같애']
    hyp=['아닐걸', '그', '뭐지', '몰라', '뭔가', '근데', '그', '남자애가', '우리', '욕했을', '거', '같애']

    ref = ' '.join(ref)
    hyp = ' '.join(hyp)

    refnorm, hypnorm = get_norm_text(ref, hyp)
    print("normalized reference: ", refnorm)
    print("normalized hypothesis: ", hypnorm)

