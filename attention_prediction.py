import torch

class Attn_Pred_Model(torch.nn.Module):
    def __init__(self, attn_block_size, num_buckets, bucket_size, past_steps, buckets_minimum):
        super(Attn_Pred_Model, self).__init__()

        self.positional_bias_forward_param = torch.nn.Parameter(torch.zeros(1,num_buckets), requires_grad=True)
        self.positional_bias_backward_param = torch.nn.Parameter(torch.zeros(1,num_buckets), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.alpha = torch.nn.Parameter(torch.ones(1), requires_grad=True)

        indices = torch.arange(attn_block_size)
        arange1 = torch.arange(attn_block_size).view((attn_block_size, 1)).repeat((1, attn_block_size))
        arange2 = (arange1 - indices) % attn_block_size
        arange2 = (arange2/num_buckets).to(torch.int)
        self.register_buffer('arange2', arange2[..., torch.arange(num_buckets)*num_buckets].to(torch.int64))
        self.register_buffer('ones_vec', torch.torch.ones((attn_block_size,1)))
        mask_tensor=torch.tril(torch.ones(attn_block_size,attn_block_size), diagonal=-bucket_size)[:,torch.arange(num_buckets)*num_buckets] # diagonal=-bucket_size because we always take the last bucket
        mask_tensor[:buckets_minimum*bucket_size]=0
        self.register_buffer('mask', mask_tensor)
        self.past_steps=past_steps

    def forward(self, x, past_weighted_attn=None, num_active_buckets=None):
        if self.training:
            result = torch.zeros_like(x)
            for i in range(self.past_steps):
                result+=self.alpha * (self.beta**i)*torch.nn.functional.pad(x[...,:-(i+1),:], pad=(0,0,i+1,0)) # should be i+1 at both locations
            
            positional_bias_forward = self.ones_vec@self.positional_bias_forward_param
            result+=positional_bias_forward

            positional_bias_backward = self.ones_vec@self.positional_bias_backward_param
            positional_bias_backward = torch.gather(positional_bias_backward, dim=-1, index=self.arange2)
            result+=positional_bias_backward

            result*=self.mask

            return result
        else:
            if past_weighted_attn is None or num_active_buckets is None:
                raise Exception("past_weighted_attn and num_active_buckets are required during training")

            weighted_attn = (x + self.beta * past_weighted_attn)
            result = self.alpha * weighted_attn
            
            result+=torch.squeeze(self.positional_bias_forward_param)

            result+=torch.nn.functional.pad(torch.squeeze(self.positional_bias_backward_param[-num_active_buckets:]), pad=(0, result.shape[-1]-num_active_buckets))

            result*=self.mask

            return result, self.beta*weighted_attn
            


