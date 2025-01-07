from llm_generate import LLMGenDB, LLMGenDict
import pickle
import re

def clear_sentence(sentence: str):
    sentence = sentence.upper().strip()
    cleaned_sentence = re.sub(r'[^a-zA-Z\s\']', '', sentence)
    cleaned_sentence = re.sub(r'[\r\n]', ' ', cleaned_sentence)
    cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence)
    return cleaned_sentence

llm_gen_dict = pickle.load(open('../log-llm-generate/llm_gen_db.Meta-Llama-3-70B-Instruct.pkl', 'rb'))
for key in llm_gen_dict.keys():
    print(f"{key=}")
    cleaned_sentence = clear_sentence(llm_gen_dict.get_value(key))
    print(f"{cleaned_sentence=}")
    print(f"{llm_gen_dict.get_origin(key)=}")

