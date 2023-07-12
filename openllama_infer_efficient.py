import torch
from transformers_local.src.transformers import LlamaTokenizer, LlamaForCausalLM
import numpy as np
from datasets import load_dataset
import math

from attention_prediction import Attn_Pred_Model


# model_path = 'openlm-research/open_llama_3b'
model_path = 'openlm-research/open_llama_7b'
# model_path = 'openlm-research/open_llama_13b'

tokenizer = LlamaTokenizer.from_pretrained(model_path, truncation_side='right')
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='auto' #, load_in_4bit=True
)
model.eval()

ds=load_dataset('stas/openwebtext-10k')

num_samples=1000 # Number of training samples
sample_indices=np.random.permutation(len(ds['train']))
useful_samples=0

batch_size = 8
current_batch = []
attn_block_size = 1024 # should be square

num_buckets = int(math.sqrt(attn_block_size))
bucket_size = num_buckets
buckets_minimum = round(num_buckets*0.2)

attn_pred_model = Attn_Pred_Model(attn_block_size, num_buckets, bucket_size).cuda()
attn_pred_model.load_state_dict(torch.load("attn_pred_model.pt"))
attn_pred_model.eval()


acc_history = []
for sample_index in sample_indices:
    with torch.no_grad():
        if useful_samples>=num_samples:
            break
        prompt = ds['train'][sample_index.item()]['text']
        prompt_len = len(tokenizer.tokenize(prompt))
        if len(tokenizer.tokenize(prompt))<buckets_minimum*bucket_size or len(tokenizer.tokenize(prompt))>2*buckets_minimum*bucket_size:
            continue
        useful_samples+=1
        current_batch.append(prompt)
        if len(current_batch)<batch_size:
            continue
        print(useful_samples)
        input_ids = tokenizer(current_batch, return_tensors="pt", truncation=True, max_length=buckets_minimum*bucket_size).input_ids
        current_batch=[]
        generation_output = model.generate(
        input_ids=input_ids, max_new_tokens=100, use_efficient_caching=True
        ) # max_new_tokens should be larger, to analyze behaviour on generated text
        print(tokenizer.batch_decode(generation_output))
        # breakpoint()
    
