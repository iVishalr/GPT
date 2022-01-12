import random
import numpy as np
from numpy.core.shape_base import block
import torch
import torch.nn as nn
from torch.nn import functional as F

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def tok_k_logits(logits, k):
    v, ix = torch.topk(logits,k)
    out = logits.clone()
    out[out < v[:,[-1]]] = -float('inf')
    return out

@torch.no_grad()
def sample_context(model, x, steps, temperature=1.0, sample=True, top_k=10):
    block_size = model.block_size
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
        _, logits = model(x_cond)
        # take the logits at the final time step and scale by temperature
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            logits = tok_k_logits(logits, top_k)
        
        probs = F.softmax(logits, dim=-1)

        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _,ix = torch.topk(probs, k=1, dim=-1)
        x = torch.cat((x,ix),dim=1)
    return x