import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

class TrainingConfig:
    max_epochs = 100
    lr = 3e-4
    betas = (0.9,0.95)
    weight_decay = 0.1
    epsilon = 10e-8
    batch_size = 64
    grad_norm_clip = 1.0
    lr_decay = True
    num_workers = 8
    warmup_tokens = 375e6
    final_tokens = 260e9
    shuffle = True
    pin_memory = True
    device = "cuda"
    ckpt_path = "./transformers.pt"

    def __init__(self, **kwargs) -> None:
        for k,v in kwargs.items():
            setattr(self,k,v)

class Trainer:
    def __init__(self, model, train_set, test_set, configs) -> None:
        self.model = model
        self.train_set = train_set
        self.test_set = test_set
        self.configs = configs

        self.device = torch.device("cpu")

        if self.configs.device == "cuda" and torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = nn.DataParallel(self.model).to(self.device)
            # self.model.to(self.device)
        
    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        print(f"saving model at {self.configs.ckpt_path}.")
        torch.save(raw_model.state_dict(), self.configs.ckpt_path)

    def train(self):
        model, config = self.model, self.configs
        raw_model = self.model.module if hasattr(self.model,"module") else self.model
        optimizer = raw_model.configure_optimizer(config)

        def run_epoch(split):
            is_train = split=="train"
            if is_train:
                model.train()
            else:
                model.eval()
            
            data = self.train_set if is_train else self.test_set
            loader = DataLoader(data, batch_size=config.batch_size, 
                                num_workers=config.num_workers, 
                                shuffle=config.shuffle, 
                                pin_memory=config.pin_memory)
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            losses = []
            for ix, data in pbar:
                x, y = data
                x = x.to(self.device)
                y = y.to(self.device)

                loss, logits = model(x,y)
                loss = loss.mean()
                losses.append(loss.item())

                if is_train:
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    if config.lr_decay:
                        self.tokens+= (y>=0).sum()
                        if self.tokens < config.warmup_tokens:
                            lr_multiplier = float(self.tokens) / float(max(1,config.warmup_tokens))
                        else:
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_multiplier = max(0.1, 0.5*(1.0+math.cos(math.pi * progress)))
                        lr = config.lr * lr_multiplier
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.lr
                    pbar.set_description(f"epoch {epoch+1} it: {ix+1} | loss: {loss.item():.5f} lr: {lr:e}")
                if not is_train:
                    test_loss = float(np.mean(losses))
                    print("test loss : ", test_loss)
                    return test_loss

        best_loss = float('inf')
        test_loss = float('inf')
        self.tokens = 0
        for epoch in range(self.configs.max_epochs):
            run_epoch('train')

            if self.test_set is not None:
                test_loss = run_epoch('test')
            
            good_model = self.test_set is None or test_loss < best_loss
            if self.configs.ckpt_path is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint()