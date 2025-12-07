import asyncio
from deep_rwkv.config import ProjectConfig
from deep_rwkv.modeling import DeepRWKV
from deep_kv.mcts_engine import MCTSEngine

async def main():
    config = ProjectConfig()
    model = DeepRWKV(config)
    engine = MCTSEngine(model, config)

    problem = "Let S be the set of all positive integers n such that n^2 is a multiple of n + 15. Find the largest integer in S."
    prompt = f"User: {problem}\n\nAssistant: Let's think step by step."

    print(f"--- Prompt ---\n{prompt}")
    print("\n--- DeepRWKV Response ---")
    
    response = await engine.run(prompt, max_new_tokens=300)
    
    print(f"\n\n--- Final Answer ---\n{response}")

if __name__ == "__main__":
    asyncio.run(main())