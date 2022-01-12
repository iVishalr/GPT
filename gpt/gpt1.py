import torch
import torch.nn as nn
import torch.nn.functional as F
from .sublayers import SelfAttention, PointWiseFeedForward
from .utils.gpt_config import GPT1config

class GPT1Block(nn.Module):

    def __init__(self, config: GPT1config) -> None:
        super().__init__()

        self.attn = SelfAttention(config)

        self.ln1 = nn.LayerNorm(config.embd_size)
        self.ln2 = nn.LayerNorm(config.embd_size)

        self.mlp = PointWiseFeedForward(config)

    def forward(self, x) -> torch.Tensor:
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.mlp(x))
        return x

class GPT1(nn.Module):
    def __init__(self, config: GPT1config) -> None:
        super().__init__()

        self.tok_embd = nn.Embedding(config.vocab_size,config.embd_size)
        self.pos_embd = nn.Parameter(torch.zeros(1,config.block_size, config.embd_size))
        self.embd_drop = nn.Dropout(config.pos_drop)

        self.blocks = nn.Sequential(*[GPT1Block(config) for _ in range(config.num_layers)])

        self.head = nn.Linear(config.embd_size, config.vocab_size)

        self.apply(self._init_weights)
        self.block_size = config.block_size

        print("Number of Trainable Parameters : ", sum([p.numel() for p in self.parameters()]))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(0,0.02)
            if isinstance(module, (nn.Linear)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm)):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def forward(self, x, targets=None):

        B, T = x.size()
        tok_embd = self.tok_embd(x)
        pos_embd = self.pos_embd[:, :T, :]
        
        embd = self.embd_drop(tok_embd + pos_embd)
        out = self.blocks(embd)
        logits = self.head(out)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss, logits

    def configure_optimizer(self, config) -> torch.optim.AdamW:
        decay = set()
        no_decay = set()

        for pn, p in self.named_parameters():
            if pn.endswith('bias'):
                no_decay.add(pn)
            elif pn.endswith('weight') and (".ln" in pn or ".tok_embd" in pn):
                no_decay.add(pn)
            else:
                decay.add(pn)

        params = {pn:p for pn,p in self.named_parameters()}

        optim_groups = [
            {"params": [params[pn] for pn in sorted(list(decay))], "weight_decay": config.weight_decay},
            {"params": [params[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=config.lr, betas=config.betas, eps=config.epsilon)
        return optimizer