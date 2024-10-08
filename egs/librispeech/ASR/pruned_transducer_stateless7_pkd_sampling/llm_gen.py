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
    def __init__(self, model="", target_wer=15.0):
        self.entries = {}  # LLMGenDict 객체들을 저장할 딕셔너리
        self.model = model
        self.target_wer = target_wer

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
        print(f"WER: {self.wer}")
        if not self.entries:
            return "LLMGenDB is empty."
        return "\n".join([str(entry) for entry in self.entries.values()])

