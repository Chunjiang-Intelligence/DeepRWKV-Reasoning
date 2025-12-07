import torch
import torch.nn.functional as F
import numpy as np
import asyncio
from typing import List, Dict, Optional, Tuple
from deep_rwkv.config import ProjectConfig
from deep_rwkv.modeling import DeepRWKV
from deep_kv.utils import clone_state, get_special_tokens, calculate_entropy, add_dirichlet_noise

class MCTSNode:
    def __init__(self, parent: Optional['MCTSNode'], prior: float):
        self.parent = parent
        self.children: Dict[int, 'MCTSNode'] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.state = None

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def select_child(self, puct_c: float) -> Tuple[int, 'MCTSNode']:
        best_score = -float('inf')
        best_action = -1
        best_child = None
        for action, child in self.children.items():
            score = child.value + puct_c * child.prior * (np.sqrt(self.visit_count) / (1 + child.visit_count))
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

class MCTSEngine:
    def __init__(self, model: DeepRWKV, config: ProjectConfig):
        self.model = model
        self.config = config
        self.special_tokens = get_special_tokens(model.pipeline)

    async def run(self, prompt: str, max_new_tokens: int = 256):
        root_tokens = self.model.pipeline.encode(prompt)
        _, root_state = self.model.forward_policy(root_tokens, None)
        
        root_node = MCTSNode(parent=None, prior=1.0)
        root_node.state = root_state

        await self._expand(root_node, add_noise=True)

        generated_tokens = []
        for _ in range(max_new_tokens):
            for _ in range(self.config.mcts.simulations_per_step // self.config.mcts.parallel_leaves):
                await self._run_batch_simulations(root_node)

            action = self._select_action(root_node)
            if action == self.special_tokens["newline"]:
                break
            
            generated_tokens.append(action)
            print(self.model.pipeline.decode([action]), end="", flush=True)

            root_node = root_node.children[action]
            root_node.parent = None
            if not root_node.children:
                await self._expand(root_node, add_noise=False)

        return self.model.pipeline.decode(generated_tokens)

    async def _run_batch_simulations(self, root: MCTSNode):
        paths, leaves = [], []
        for _ in range(self.config.mcts.parallel_leaves):
            path, leaf = self._select_leaf(root)
            paths.append(path)
            leaves.append(leaf)

        values = await self._evaluate_leaves(leaves)

        for path, value in zip(paths, values):
            self._backpropagate(path, value)

    def _select_leaf(self, root: MCTSNode) -> Tuple[List[MCTSNode], MCTSNode]:
        path = [root]
        node = root
        while node.children:
            if not node.state: # This indicates a pruned or abstract node
                break
            _, node = node.select_child(self.config.mcts.puct_c)
            path.append(node)
        return path, node

    async def _evaluate_leaves(self, leaves: List[MCTSNode]) -> List[float]:
        tasks = [self._expand_and_get_value(leaf) for leaf in leaves]
        values = await asyncio.gather(*tasks)
        return values

    async def _expand_and_get_value(self, node: MCTSNode) -> float:
        if not node.children:
            await self._expand(node, add_noise=False)
        
        value = 0.0
        if self.config.mcts.use_learned_value:
            learned_value = self.model.forward_value(node.state).item()
            value += self.config.mcts.learned_value_weight * learned_value

        if self.config.mcts.learned_value_weight < 1.0:
            heuristic_value = await self._heuristic_rollout(node)
            value += (1.0 - self.config.mcts.learned_value_weight) * heuristic_value
        
        return value

    async def _expand(self, node: MCTSNode, add_noise: bool):
        logits, _ = self.model.forward_policy([0], clone_state(node.state))
        probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
        
        if add_noise:
            probs = add_dirichlet_noise(probs, self.config.mcts.dirichlet_alpha, self.config.mcts.dirichlet_epsilon)

        for token_id, prob in enumerate(probs):
            if prob > 1e-5:
                node.children[token_id] = MCTSNode(parent=node, prior=prob)

    async def _heuristic_rollout(self, node: MCTSNode) -> float:
        current_state = clone_state(node.state)
        last_token = 0
        total_confidence = 0
        
        for _ in range(self.config.heuristic.rollout_depth):
            logits, current_state = self.model.forward_policy([last_token], current_state)
            probs = F.softmax(logits, dim=-1)
            confidence, next_token = torch.max(probs, dim=-1)
            
            total_confidence += confidence.item()
            last_token = next_token.item()
            if last_token == self.special_tokens["newline"]:
                break
        
        avg_confidence = total_confidence / self.config.heuristic.rollout_depth
        return avg_confidence * 2 - 1

    def _backpropagate(self, path: List[MCTSNode], value: float):
        for node in reversed(path):
            node.value_sum += value
            node.visit_count += 1

    def _select_action(self, root: MCTSNode) -> int:
        visit_counts = {action: child.visit_count for action, child in root.children.items()}
        return max(visit_counts, key=visit_counts.get)