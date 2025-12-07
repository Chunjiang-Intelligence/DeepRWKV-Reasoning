import os
import torch
import torch.nn as nn
from rwkv.model import RWKV
from rwkv.utils import PIPELINE
from deep_rwkv.config import ProjectConfig

class ValueHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.LayerNorm(config.input_dim),
            nn.Linear(config.input_dim, config.hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class DeepRWKV(nn.Module):
    def __init__(self, config: ProjectConfig):
        super().__init__()
        self.config = config
        
        self._setup_environment()
        
        self.backbone = RWKV(model=config.model.model_path, strategy=config.model.strategy)
        self.pipeline = PIPELINE(self.backbone, config.model.vocab_path)
        self.args = self.backbone.args
        self.config.value_head.input_dim = self.args.n_embd
        
        self.value_head = ValueHead(config.value_head)
        
        if config.mcts.use_learned_value:
            self._load_value_head()

    def _setup_environment(self):
        os.environ["RWKV_V7_ON"] = "1"
        os.environ["RWKV_JIT_ON"] = "1"
        os.environ["RWKV_CUDA_ON"] = "1"
    
    def _load_value_head(self):
        path = self.config.value_head.checkpoint_path
        if path and os.path.exists(path):
            print(f"[DeepRWKV] Loading Value Head from: {path}")
            self.value_head.load_state_dict(torch.load(path, map_location=self.backbone.device))
        else:
            print(f"[DeepRWKV] Warning: Value Head checkpoint not found at {path}. Using random initialization.")

    def forward_policy(self, tokens, states):
        return self.backbone.forward(tokens, states)

    def forward_value(self, states):
        last_ffn_state = states[-1]  # This is a strong assumption and may need tuning.
        return self.value_head(last_ffn_state)