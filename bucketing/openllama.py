import torch
from transformers_local.src.transformers import LlamaTokenizer, LlamaForCausalLM
import numpy as np
from datasets import load_dataset
import math

class Attn_Pred_Model(torch.nn.Module):
    def __init__(self, num_buckets):
        super(Conv_Model, self).__init__()
        self.positional_bias_forward = torch.nn.Parameter(torch.zeros(num_buckets, requires_grad=True).cuda())
        self.positional_bias_backward = torch.nn.Parameter(torch.zeros(num_buckets, requires_grad=True).cuda())
        self.beta = torch.nn.Parameter(torch.zeros(1, requires_grad=True).cuda())
        self.alpha = torch.nn.Parameter(torch.zeros(1, requires_grad=True).cuda())

    def forward(self, x):
        



# model_path = 'openlm-research/open_llama_3b'
model_path = 'openlm-research/open_llama_7b'
# model_path = 'openlm-research/open_llama_13b'

tokenizer = LlamaTokenizer.from_pretrained(model_path, truncation_side='right')
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='auto'
)


ds=load_dataset('stas/openwebtext-10k')

num_samples=1000 # Number of training samples
sample_indices=np.random.permutation(len(ds['train']))
useful_samples=0

batch_size = 10
current_batch = []
attn_block_size = 1024 # should be square

num_buckets = int(math.sqrt(attn_block_size))
bucket_size = num_buckets
buckets_minimum = round(num_buckets*0.2)

for sample_index in sample_indices:
    if useful_samples>=num_samples:
        break
    prompt = ds['train'][sample_index.item()]['text']
    prompt_len = len(tokenizer.tokenize(prompt))
    if len(tokenizer.tokenize(prompt))<attn_block_size or len(tokenizer.tokenize(prompt))>2*attn_block_size:
        continue
    useful_samples+=1
    current_batch.append(prompt)
    if len(current_batch)<batch_size:
        continue
    print(useful_samples)
    input_ids = tokenizer(current_batch, return_tensors="pt", truncation=True, max_length=attn_block_size).input_ids
    current_batch=[]
    generation_output = model.generate(
    input_ids=input_ids, max_new_tokens=1, output_attentions=True, return_dict_in_generate=True, bucketize_out_attn_norm=2, bucket_size=bucket_size
    ) # max_new_tokens should be larger, to analyze behaviour on generated text
    
    attn_list = generation_output['attentions'][0]

    for attn in attn_list:


        
