from dataclasses import dataclass

@dataclass
class DeepConfig:
    model_path: str = "rwkv7-g0a3-7.2b-20251029-ctx8192.pth"
    strategy: str = "cuda fp16"

    mcts_simulations: int = 128   # 搜索次数
    batch_size: int = 16          # 并行叶子节点扩展数
    depth_limit: int = 32         # 搜索深度限制
    
    use_uncertainty_penalty: bool = True  # 启用熵惩罚 (UES)
    uncertainty_lambda: float = 0.8       # 熵惩罚系数
    
    value_head_path: str = "checkpoints/value_head_latest.pth"
    use_learned_value: bool = False