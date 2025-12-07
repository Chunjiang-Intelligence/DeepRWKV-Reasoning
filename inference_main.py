from deep_rwkv.config import DeepConfig
from deep_rwkv.modeling import DeepRWKVWrapper
from deep_rwkv.mcts_engine import FluxMCTS
from rwkv.utils import PIPELINE

def main():
    cfg = DeepConfig()
    cfg.use_learned_value = False
    
    wrapper = DeepRWKVWrapper(cfg)
    pipeline = PIPELINE(wrapper.model, "rwkv_vocab_v20230424")
    
    engine = FluxMCTS(wrapper, pipeline, cfg)
    
    problem = "Let S be the set of integers n such that n^2 is a multiple of n+15. Max(S)?"
    prompt = f"User: {problem}\n\nAssistant: Let's think step by step."
    
    print(f"\n[DeepRWKV] Problem: {problem}")
    engine.run(prompt)

if __name__ == "__main__":
    main()