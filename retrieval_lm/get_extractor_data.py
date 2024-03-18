from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import torch
import os
import numpy as np
import json
import re
from tqdm.auto import tqdm
from typing import Iterable
from collections import OrderedDict
from src.utils import DEVICE
from utils import PROMPT_DICT, TASK_INST, load_file, load_jsonlines, save_file_jsonl, control_tokens, load_special_tokens
from metrics import match, accuracy


seed = 633

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def extract_context(question, context):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": question}
    ]
    encoded = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encoded.to(DEVICE)

    generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    return decoded[0]


gpt = "mistralai/Mistral-7B-Instruct-v0.2"
SYSTEM_PROMPT = """You are an expert research assistant. Please find the quotes from the document that are most relevant to answering the upcoming question, and then print them in numbered order. Quotes should be relatively short and verbatim.
If there are no relevant quotes, write "N/A" instead. Here is a document you will answer questions about:\n{context}\n"""

tokenizer = AutoTokenizer.from_pretrained(gpt)
model = AutoModelForCausalLM.from_pretrained(gpt, device_map=DEVICE)


input_data = load_file("eval_data/popqa_longtail_w_gs.jsonl")
