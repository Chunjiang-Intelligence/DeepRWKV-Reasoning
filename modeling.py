import torch
import torch.nn as nn
import os
from rwkv.model import RWKV

class ValueHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh() # Output range [-1, 1]
        )

    def forward(self, x):
        return self.net(x)

class DeepRWKVWrapper:
    def __init__(self, config):
        print(f"[DeepRWKV] Initializing v7 Kernel on A100...")
        os.environ["RWKV_V7_ON"] = "1"
        os.environ["RWKV_JIT_ON"] = "1"
        os.environ["RWKV_CUDA_ON"] = "1"
        
        self.model = RWKV(model=config.model_path, strategy=config.strategy)
        self.config = config
        self.args = self.model.args
        
        self.value_head = None
        if config.use_learned_value and os.path.exists(config.value_head_path):
            print(f"[DeepRWKV] Loading Value Head from {config.value_head_path}")
            self.value_head = ValueHead(self.args.n_embd).to("cuda").to(dtype=torch.float16)
            self.value_head.load_state_dict(torch.load(config.value_head_path))
            self.value_head.eval()
        else:
            print("[DeepRWKV] No Value Head found/enabled. Using Heuristic Engine.")

    def forward_with_hidden(self, tokens, state=None):
        with torch.no_grad():
            logits, new_state = self.model.forward(tokens, state)
            hidden_proxy = logits
        return logits, new_state, hidden_proxy

    def estimate_value(self, hidden_proxy):
        if self.value_head is None:
            return None
        if hidden_proxy.shape[-1] != self.args.n_embd:
            return 0.0 
             
        return self.value_head(hidden_proxy).item()