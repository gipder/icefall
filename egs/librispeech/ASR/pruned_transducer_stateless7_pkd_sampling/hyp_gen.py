from datetime import datetime
import re
import os
import pickle
from jiwer import wer
import getpass
import concurrent.futures
from contextlib import nullcontext

class HYPGenDict:
    def __init__(self, key, original_sentence, hyp_sentences):
        self.key = key
        self.original_sentence = original_sentence
        self.hyp_sentences = list()
        if type(hyp_sentences) == list:
            self.hyp_sentences = hyp_sentences
        else:
            self.hyp_sentences.append(hyp_sentences)
        self.wer = wer(self.original_sentence, self.hyp_sentences[0])

    def __str__(self):
        return f"Key: {self.key}\nOriginal Sentence: {self.original_sentence}\nN-Best Hyp Sentences: {self.hyp_sentences}\nWER: {self.wer}"

class HYPGenDB:
    def __init__(self, test_set=""):
        self.entries = {}  # LLMGenDict 객체들을 저장할 딕셔너리
        self.test_set = test_set

    def keys(self):
        return self.entries.keys()

    def values(self):
        return self.entries.values()

    def add_entry(self, key, original_sentence, hyp_sentence):
        if key not in self.entries:
            self.entries[key] = HYPGenDict(key, original_sentence, hyp_sentence)
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

    def get_hyp_sentences(self, key):
        if key in self.entries:
            return self.entries[key].hyp_sentences
        else:
            print(f"Entry with key '{key}' not found.")
            return None

    def get_value(self, key):
        return self.get_hyp_sentences(key)

    def get_origin(self, key):
        return self.get_original_sentence(key)

    def get_wer(self, key):
        return self.entries[key].wer

    def get_test_set(self):
        return self.test_set

    def delete_entry(self, key):
        if key in self.entries:
            del self.entries[key]
        else:
            print(f"Entry with key '{key}' not found.")

    def __str__(self):
        print(f"Test Set: {self.test_set}")
        if not self.entries:
            return "HYPGenDB is empty."
        return "\n".join([str(entry) for entry in self.entries.values()])

