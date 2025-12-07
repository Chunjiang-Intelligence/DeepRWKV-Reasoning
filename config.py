from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    model_path: str = "rwkv7-g0a3-7.2b-20251029-ctx8192.pth"
    strategy: str = "cuda bf16"
    vocab_path: str = "rwkv_vocab_v20230424"

@dataclass
class ValueHeadConfig:
    input_dim: int = 4096  # Should match RWKV n_embd
    hidden_dim: int = 1024
    checkpoint_path: Optional[str] = "checkpoints/value_head_latest.pth"

@dataclass
class MCTSConfig:
    simulations_per_step: int = 128
    parallel_leaves: int = 16  # A100: 16-64
    max_depth: int = 40
    puct_c: float = 1.2
    dirichlet_alpha: float = 0.25
    dirichlet_epsilon: float = 0.25
    
    use_learned_value: bool = True
    learned_value_weight: float = 0.8 # Hybrid mode weight

@dataclass
class HeuristicConfig:
    rollout_depth: int = 15
    confidence_weight: float = 0.6
    reflection_weight: float = 0.4
    reflection_prompt: str = "\nIs the reasoning so far correct? Answer Yes or No:"

@dataclass
class TrainingConfig:
    dataset_path: str = "data/prm_dataset.jsonl"
    save_path: str = "checkpoints/value_head_latest.pth"
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

@dataclass
class ProjectConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    value_head: ValueHeadConfig = field(default_factory=ValueHeadConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    heuristic: HeuristicConfig = field(default_factory=HeuristicConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)