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

attn_pred_model = Attn_Pred_Model(attn_block_size=attn_block_size, num_buckets=num_buckets, bucket_size=bucket_size, past_steps=bucket_size, buckets_minimum=buckets_minimum).cuda()
attn_pred_model.train()

optimizer = torch.optim.SGD(attn_pred_model.parameters(), lr=0.001)

acc_history = []
for sample_index in sample_indices:
    with torch.no_grad():
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
        input_ids=input_ids, max_new_tokens=1, output_attentions=True, return_dict_in_generate=True, bucketize_out_attn_norm=2, num_buckets=num_buckets
        ) # max_new_tokens should be larger, to analyze behaviour on generated text
        
        attn_list = generation_output['attentions'][0]
    for attn in attn_list:
        optimizer.zero_grad()
        attn = attn.cuda()

        pred = attn_pred_model(attn)
        true = attn*attn_pred_model.mask
        loss = torch.mean(torch.square(torch.norm(true - pred, dim=-1)))
    
        loss.backward()
        optimizer.step()

        
        _, true_top_k = torch.topk(true, k=buckets_minimum-1, dim=-1)
        _, pred_top_k = torch.topk(pred, k=buckets_minimum-1, dim=-1)

        last_bucket = torch.div(torch.arange(attn_block_size), bucket_size, rounding_mode='floor').unsqueeze(dim=-1).unsqueeze(dim=0).unsqueeze(dim=0).expand(tuple(true_top_k.shape[:-1])+(1,)).cuda()
        true_top_k = torch.cat([true_top_k, last_bucket], dim=-1)
        pred_top_k = torch.cat([pred_top_k, last_bucket], dim=-1)

        true_top_k = torch.zeros_like(true).scatter_(dim=-1, index=true_top_k, src=torch.ones_like(true))
        pred_top_k = torch.zeros_like(pred).scatter_(dim=-1, index=pred_top_k, src=torch.ones_like(pred))

        acc = torch.mean((torch.maximum(torch.norm(true*pred_top_k,dim=-1), torch.tensor(1e-7))/ torch.maximum(torch.norm(true*true_top_k,dim=-1), torch.tensor(1e-7)))[...,(buckets_minimum)*bucket_size:])
        acc_history.append(acc.item())

print(f'attn_pred_model.alpha: {attn_pred_model.alpha}')
print(f'attn_pred_model.beta: {attn_pred_model.beta}')
print(f'attn_pred_model.positional_bias_forward_param: {attn_pred_model.positional_bias_forward_param}')
print(f'attn_pred_model.positional_bias_backward_param: {attn_pred_model.positional_bias_backward_param}')
print(f'acc_history.mean: {np.mean(acc_history)}')

torch.save(attn_pred_model.state_dict(), "attn_pred_model.pt")
        
