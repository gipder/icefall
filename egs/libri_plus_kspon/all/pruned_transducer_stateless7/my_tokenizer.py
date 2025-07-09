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

    def decode(self, ids):
        return ''.join([self.vocab_dict.get(id, self.unk)
                        for id in ids]).replace('▁', ' ')

    def decode_batch(self, ids, num_threads=1):
        num_threads = max(1, os.cpu_count())
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(self.decode, ids))
        return results


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
