import asyncio
import time
from deep_rwkv.config import ProjectConfig
from deep_rwkv.mcts_engine import ParallelMCTSEngine

async def main():
    config = ProjectConfig()
    config.mcts.simulations_per_step = 128 
    
    engine = ParallelMCTSEngine(config, num_universes=3)

    problem = "Let $S$ be the set of all positive integers $n$ such that $n^2$ is a multiple of $n + 15$. Find the largest integer in $S$."
    prompt = f"User: {problem}\n\nAssistant: Let's think step by step to find the solution."

    print(f"--- Prompt ---\n{prompt}")
    print("\n--- DeepRWKV Parallel Universe Search Initialized ---")
    
    start_time = time.time()
    results = await engine.run(prompt, max_new_tokens=300)
    end_time = time.time()
    
    print("\n\n" + "="*50)
    print("           FINAL RESULTS")
    print("="*50)
    
    print("\n--- Consensus Response ---")
    print(results["consensus_response"])
    
    print("\n--- Individual Universe Trajectories ---")
    for i, resp in results["universe_responses"].items():
        print(f"Universe {i} (Strategy: {engine.universes[i].strategy}): {resp}")
    
    print("\n" + "="*50)
    print(f"Search completed in {end_time - start_time:.2f} seconds.")
    print("Visualization server is still running. Press Ctrl+C to exit.")
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting.")

if __name__ == "__main__":
    try:
        import eventlet
        eventlet.monkey_patch()
    except ImportError:
        pass
    
    asyncio.run(main())