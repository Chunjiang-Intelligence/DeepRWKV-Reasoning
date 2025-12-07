# DeepRWKV-Reasoning

**DeepRWKV-Reasoning is a framework designed to enhance the reasoning capabilities of Large Language Models through sophisticated test-time search algorithms.This project leverages the unique architectural advantages of the RWKV model, combining it with a high-performance, asynchronous Monte Carlo Tree Search engine to create a powerful System 2 thinking machine.

The core of this project is the **Parallel Universe Search** mechanism, an advanced MCTS implementation where multiple search strategies explore the problem space concurrently, converging on a robust solution through a consensus mechanism.

## Key Features

- Parallel Universe MCTS: Deploys multiple MCTS engines, each with a unique search strategy (PUCT, KL-UCB, Thompson Sampling), to explore diverse reasoning paths simultaneously and improve solution robustness.
- Probabilistically Advanced Search: Integrates information-theoretic principles like **KL-UCB** for optimal exploration and **Thompson Sampling** (via Beta distributions) for value estimation, moving beyond simple mean-based metrics.
- Learned Value Head & Heuristic Fallback: Supports a fine-tunable MLP `ValueHead` to predict the potential of a reasoning path, seamlessly falling back to a sophisticated heuristic evaluator that combines model confidence and self-reflection.
- Live Visualization: Features a built-in web server with a `D3.js` frontend to provide a real-time, interactive visualization of the MCTS search tree, offering unparalleled insight into the model's "thought process".
- Asynchronous & High-Performance: Built on `asyncio` to handle thousands of concurrent simulations, with a design optimized for high-throughput hardware like NVIDIA A100 GPUs.
- Modular & Extensible: Cleanly architected with a clear separation of concerns, making it easy to add new search algorithms, value functions, or models.

## Why RWKV is Uniquely Suited for MCTS

While MCTS can be applied to any autoregressive model, the RWKV architecture offers fundamental advantages that make it an exceptionally powerful and efficient backbone for tree-based search.

#### 1. Constant-Cost State Representation
- Transformer Limitation: In Transformer-based models (like GPT or Llama), the "state" of the model at any given point is its entire KV Cache.This cache grows linearly with sequence length, making it massive (many gigabytes).Storing, cloning, or switching between the states of thousands of MCTS nodes is prohibitively expensive in terms of memory and computation.
- RWKV Advantage: As a Recurrent Neural Network (RNN), RWKV encapsulates its entire context into a **fixed-size state tensor**.This state is small (megabytes), independent of sequence length, and acts as a compressed representation of the entire history.

#### 2. Efficient Node Management
- Cloning States: In our MCTS engine, creating a new search branch requires cloning the parent node's state.For a Transformer, this means copying a huge KV Cache.For RWKV, it's a fast and cheap clone of a small, fixed-size tensor.This allows our engine to maintain hundreds of thousands of nodes in memory, a feat virtually impossible with Transformers without complex engineering like vLLM's PagedAttention.
- State Switching: Traversing the search tree involves constantly switching between different states.With RWKV, this is as simple as swapping a pointer to a different state tensor.Transformers require complex KV Cache management to avoid re-computation, adding significant overhead.

#### 3. Natural Fit for State-Space Search
MCTS is fundamentally a state-space search algorithm.The RWKV state is a true "state" in the state-space sense—it's all you need to predict the future.This clean mapping from the model's internal state to a search node's state makes the implementation elegant and performant.Transformers, by contrast, require their entire history (the full sequence of tokens) to be conceptually part of the state, complicating the search process.

In summary, **RWKV transforms the problem of LLM tree search from a big data management challenge into a far more tractable algorithmic problem**, making it the ideal architecture for deep, expansive reasoning tasks.

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.0+ with CUDA support
- An NVIDIA GPU with 24GB+ VRAM (A100 recommended)
- A pre-trained RWKV model

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Chunjiang-Intelligence/DeepRWKV-Reasoning.git
    cd DeepRWKV-Reasoning
    ```

2.  Install required Python packages:
    ```bash
    pip install torch rwkv numpy flask flask-socketio eventlet
    ```

3.  Download an RWKV model and place it in a `models/` directory (or update the path in `deep_rwkv/config.py`).

### Running Inference

To run the full Parallel Universe Search with real-time visualization:

```bash
python inference_main.py
```

- Terminal Output: Observe the step-by-step reasoning, including the votes from each universe and the final consensus.
- Web Visualization: Open your browser and navigate to `http://localhost:5001` to see the live MCTS search tree of the primary universe.

### Training the Value Head (Optional)

The performance of the MCTS engine can be significantly boosted by a trained Value Head.

1.  **Prepare a Dataset**: Create a dataset in `data/prm_dataset.jsonl` format. Each line should be a JSON object containing `{"tokens": [list_of_token_ids], "reward": float_value}`, where the reward is typically `1.0` for steps leading to a correct solution and `-1.0` for incorrect steps.

2.  **Run Training**:
    ```bash
    python train_value_head.py
    ```
    This will save the trained model to `checkpoints/value_head_latest.pth`. The inference engine will automatically detect and use it if `use_learned_value` is enabled in the config.

## Future Work

- [ ] **Advanced Voting Mechanisms**: Implement confidence-weighted voting instead of simple majority rule.
- [ ] **Graph Deduplication**: Use a transposition table to merge identical states reached via different paths, turning the search tree into a graph.
- [ ] **Distributed Execution**: Scale the engine to run universes across multiple GPUs or machines.
