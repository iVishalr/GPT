class GPTconfig:
    resid_drop = 0.1
    attn_drop = 0.1
    pos_drop = 0.1
    block_size = 512
    vocab_size = 65

    def __init__(self, **kwargs) -> None:
        for k,v in kwargs.items():
            setattr(self,k,v)

class GPT1config(GPTconfig):
    n_heads = 12
    num_layers = 12
    embd_size = 768
    
class GPT2config(GPTconfig):
    n_heads = 12
    num_layers = 12
    embd_size = 768
    vocab_size = 50257
    block_size = 1024