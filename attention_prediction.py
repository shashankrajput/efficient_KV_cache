import torch

class Attn_Pred_Model(torch.nn.Module):
    def __init__(self, num_buckets, attn_block_size):
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
        self.register_buffer('mask', torch.tril(torch.ones(attn_block_size,attn_block_size))[:,torch.arange(num_buckets)*num_buckets])

    def forward(self, x, past_steps):
        result = torch.zeros_like(x)
        for i in range(past_steps):
            result+=self.alpha * (self.beta**i)*torch.nn.functional.pad(x[...,:-(i+1),:], pad=(0,0,i+1,0)) # should be i+1 at both locations
        
        positional_bias_forward = self.ones_vec@self.positional_bias_forward_param
        result+=positional_bias_forward

        positional_bias_backward = self.ones_vec@self.positional_bias_backward_param
        positional_bias_backward = torch.gather(positional_bias_backward, dim=-1, index=self.arange2)
        result+=positional_bias_backward

        result*=self.mask

        return result
