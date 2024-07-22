import pickle
from llm_gen import LLMGenDB, LLMGenDict
import sys

# load old pickle file
# pickle_rewrite.py <old_pickle> <new_pickle>
old_pickle = sys.argv[1]
new_pickle = sys.argv[2]
llm_gen = pickle.load(open(old_pickle, 'rb'))
pickle.dump(llm_gen, open(new_pickle, 'wb'))
