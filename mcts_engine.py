import torch
import torch.nn.functional as F
import math
import numpy as np
from copy import deepcopy

class FluxNode:
    def __init__(self, token_id, state, parent=None, prior=0.0):
        self.token_id = token_id
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior
        self.is_terminal = False
        self.policy_entropy = 0.0 

    def value(self):
        return self.value_sum / (self.visits + 1e-6)

    def uct(self, c_puct=1.2, lambda_ues=0.5):
        # Uncertainty-Entropy Scaling
        # 如果该节点产生时的策略熵很高，说明模型很困惑，我们降低其 UCT 分数
        q_score = self.value()
        
        # Standard PUCT
        u_score = c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        
        # Entropy Penalty
        penalty = lambda_ues * self.policy_entropy
        
        return q_score + u_score - penalty

class FluxMCTS:
    def __init__(self, wrapper, pipeline, config):
        self.wrapper = wrapper
        self.pipeline = pipeline
        self.cfg = config

    def batch_search(self, root_node):
        for _ in range(self.cfg.mcts_simulations // self.cfg.batch_size):
            leaves = []
            
            # 由于 Python GIL，我们顺序选取 B 个叶子，但这依然比顺序推理快
            for _ in range(self.cfg.batch_size):
                node = root_node
                depth = 0
                while node.children and depth < self.cfg.depth_limit:
                    node = max(node.children.values(), key=lambda n: n.uct(
                        lambda_ues=self.cfg.uncertainty_lambda if self.cfg.use_uncertainty_penalty else 0
                    ))
                    depth += 1
                leaves.append(node)

            for leaf in leaves:
                if leaf.visits > 0 and not leaf.is_terminal:
                    self._expand_and_evaluate(leaf)
                else:
                    self._expand_and_evaluate(leaf)

    def _expand_and_evaluate(self, node):
        # Forward Pass
        # 注意：这里需要深拷贝 state，因为我们要基于它生成
        # RWKV forward
        logits, new_state, hidden = self.wrapper.forward_with_hidden([node.token_id], deepcopy(node.state))
        
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        
        v = self.wrapper.estimate_value(hidden)
        
        if v is None:
            # 使用置信度作为 Value
            max_prob = torch.max(probs).item()
            v = max_prob * 2 - 1 # Map [0,1] -> [-1, 1]
        
        # Backprop
        curr = node
        while curr:
            curr.visits += 1
            curr.value_sum += v
            curr = curr.parent
            
        # Expansion (Only expand if visits > threshold to save memory, dynamic pruning)
        if node.visits >= 1:
            topk = torch.topk(probs, k=3)
            for val, idx in zip(topk.values, topk.indices):
                tid = idx.item()
                if tid not in node.children:
                    _, child_state, _ = self.wrapper.forward_with_hidden([tid], deepcopy(new_state))
                    child = FluxNode(tid, child_state, parent=node, prior=val.item())
                    child.policy_entropy = entropy
                    node.children[tid] = child

    def run(self, prompt):
        input_ids = self.pipeline.encode(prompt)
        # Prefill
        _, state, _ = self.wrapper.forward_with_hidden(input_ids, None)
        root = FluxNode(input_ids[-1], state)
        
        print(f"Flux-MCTS: Thinking with {self.cfg.mcts_simulations} sims/token...")
        
        generated = []
        for _ in range(200): # Max tokens
            self.batch_search(root)
            
            if not root.children: break
            
            # Select best action
            best_child = max(root.children.values(), key=lambda n: n.visits)
            token = best_child.token_id
            
            # Output
            word = self.pipeline.decode([token])
            print(word, end="", flush=True)
            generated.append(token)
            
            # Prune and Move
            root = best_child
            root.parent = None # Detach to free memory
            
            if "\n\n" in word: break
            
        return self.pipeline.decode(generated)