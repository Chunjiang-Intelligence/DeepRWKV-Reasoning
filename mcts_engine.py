import torch
import torch.nn.functional as F
import numpy as np
import asyncio
import math
from typing import List, Dict, Optional, Tuple
from deep_rwkv.config import ProjectConfig
from deep_rwkv.modeling import DeepRWKV
from deep_rwkv.utils import clone_state, get_special_tokens, calculate_entropy, add_dirichlet_noise
from web_visualizer.server import VisualizationServer

class MCTSNode:
    def __init__(self, parent: Optional['MCTSNode'], prior: float):
        self.parent = parent
        self.children: Dict[int, 'MCTSNode'] = {}
        
        self.wins = 0.0
        self.losses = 0.0
        
        self.prior = prior
        self.state = None
        self.action_token = None

    @property
    def visit_count(self) -> int:
        return int(self.wins + self.losses)
    
    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return (self.wins - self.losses) / self.visit_count

    def sample_value(self) -> float:
        sampled_q = np.random.beta(self.wins + 1, self.losses + 1)
        return sampled_q * 2 - 1

    def kl_divergence(self, p, q):
        if p == 0: return math.log(1 / (1 - q))
        if p == 1: return math.log(1 / q)
        return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

    def select_child_kl_ucb(self) -> Tuple[int, 'MCTSNode']:
        if not self.children:
            return -1, None

        log_parent_visits = math.log(self.visit_count)
        best_action = -1
        best_node = None
        max_ucb = -float('inf')

        for action, child in self.children.items():
            if child.visit_count == 0:
                return action, child
            
            p = (child.wins + 1) / (child.visit_count + 2)
            target = (log_parent_visits - math.log(child.visit_count)) / child.visit_count
            
            q = p
            high = 1.0
            for _ in range(8):
                mid = (q + high) / 2
                if self.kl_divergence(p, mid) < target:
                    q = mid
                else:
                    high = mid
            
            if q > max_ucb:
                max_ucb = q
                best_action = action
                best_node = child
                
        return best_action, best_node

class MCTSEngine:
    def __init__(self, model: DeepRWKV, config: ProjectConfig, universe_id: int, strategy: str, visualizer: Optional[VisualizationServer] = None):
        self.model = model
        self.config = config
        self.universe_id = universe_id
        self.strategy = strategy
        self.visualizer = visualizer
        self.special_tokens = get_special_tokens(model.pipeline)
        self.root = None

    async def initialize(self, prompt: str):
        root_tokens = self.model.pipeline.encode(prompt)
        _, root_state = self.model.forward_policy(root_tokens, None)
        self.root = MCTSNode(parent=None, prior=1.0)
        self.root.state = root_state
        await self._expand(self.root, add_noise=True)

    async def search_step(self):
        for _ in range(self.config.mcts.simulations_per_step // self.config.mcts.parallel_leaves):
            await self._run_batch_simulations(self.root)
        
        if self.visualizer and self.universe_id == 0:
             tree_data = self.get_tree_visualization_data(self.root)
             self.visualizer.update_tree_data({
                 "universe_id": self.universe_id, 
                 "strategy": self.strategy,
                 "tree": tree_data
             })

    def advance_root(self):
        action = self._select_action(self.root)
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
            return action
        return None

    async def _run_batch_simulations(self, root: MCTSNode):
        paths, leaves = [], []
        for _ in range(self.config.mcts.parallel_leaves):
            path, leaf = self._select_leaf(root)
            if leaf:
                paths.append(path)
                leaves.append(leaf)
        
        if not leaves:
            return

        values = await self._evaluate_leaves(leaves)
        for path, value in zip(paths, values):
            self._backpropagate(path, value)

    def _select_leaf(self, root: MCTSNode) -> Tuple[List[MCTSNode], MCTSNode]:
        path = [root]
        node = root
        while node.children:
            if not node.state:
                break
            
            if self.strategy == 'KL_UCB':
                action, next_node = node.select_child_kl_ucb()
            elif self.strategy == 'Thompson':
                children_values = {a: c.sample_value() for a, c in node.children.items()}
                action = max(children_values, key=children_values.get)
                next_node = node.children[action]
            else: # PUCT
                action, next_node = max(
                    node.children.items(), 
                    key=lambda item: item[1].value + self.config.mcts.puct_c * item[1].prior * (np.sqrt(node.visit_count) / (1 + item[1].visit_count))
                )
            
            if next_node is None:
                return path, node

            # Critical: Compute state for the child node on-the-fly if it doesn't exist
            if next_node.state is None:
                _, next_node.state = self.model.forward_policy([action], clone_state(node.state))
            
            next_node.action_token = action
            node = next_node
            path.append(node)
        return path, node

    async def _evaluate_leaves(self, leaves: List[MCTSNode]):
        tasks = [self._expand_and_get_value(leaf) for leaf in leaves]
        return await asyncio.gather(*tasks)

    async def _expand_and_get_value(self, node: MCTSNode) -> float:
        # If the node hasn't been expanded, expand it first.
        if not node.children:
            is_terminal = await self._expand(node, add_noise=False)
            if is_terminal:
                # If a terminal token like <|endoftext|> is generated, assign a neutral value.
                return 0.0
        
        value = 0.0
        # Combine learned value and heuristic value
        if self.config.mcts.use_learned_value:
            try:
                learned_value = self.model.forward_value(node.state).item()
                value += self.config.mcts.learned_value_weight * learned_value
            except Exception: # Fallback if value head fails
                value += 0.0

        if self.config.mcts.learned_value_weight < 1.0:
            heuristic_value = await self._heuristic_rollout(node)
            value += (1.0 - self.config.mcts.learned_value_weight) * heuristic_value
        
        return value

    async def _expand(self, node: MCTSNode, add_noise: bool) -> bool:
        logits, next_state = self.model.forward_policy([0], clone_state(node.state))
        
        # Check for termination
        if torch.argmax(logits) == 0: # end of text token
            return True

        probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
        
        if add_noise:
            probs = add_dirichlet_noise(probs, self.config.mcts.dirichlet_alpha, self.config.mcts.dirichlet_epsilon)

        for token_id, prob in enumerate(probs):
            if prob > 1e-4:
                child = MCTSNode(parent=node, prior=prob)
                node.children[token_id] = child
        return False

    async def _heuristic_rollout(self, node: MCTSNode) -> float:
        rollout_state = clone_state(node.state)
        # The last token that led to this node's state
        last_token = node.action_token if node.action_token is not None else 0
        
        total_log_prob = 0
        rollout_tokens = []
        
        # 1. Confidence-based greedy rollout
        for _ in range(self.config.heuristic.rollout_depth):
            logits, rollout_state = self.model.forward_policy([last_token], rollout_state)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Use greedy decoding for a stable rollout
            max_log_prob, next_token_tensor = torch.max(log_probs, dim=-1)
            
            total_log_prob += max_log_prob.item()
            next_token = next_token_tensor.item()
            
            rollout_tokens.append(next_token)
            last_token = next_token

            # Stop at sentence-like boundaries or end of text
            if last_token == self.special_tokens["newline"] or last_token == 0:
                break
        
        # Normalize confidence score to [-1, 1]
        avg_log_prob = total_log_prob / (len(rollout_tokens) if rollout_tokens else 1)
        # Heuristic mapping: e.g., avg_log_prob of -1 is decent, -5 is bad
        confidence_score = math.tanh(avg_log_prob / 3.0) # Map to [-1, 1]

        # 2. Self-reflection (if enabled)
        reflection_prompt_tokens = self.model.pipeline.encode(self.config.heuristic.reflection_prompt)
        # Append reflection prompt to the state after the rollout
        logits, _ = self.model.forward_policy(reflection_prompt_tokens, rollout_state)
        
        probs = F.softmax(logits, dim=-1)
        p_yes = probs[0, self.special_tokens["yes"]].item()
        p_no = probs[0, self.special_tokens["no"]].item()
        
        reflection_score = (p_yes - p_no) / (p_yes + p_no + 1e-6)

        # 3. Combine scores
        final_value = (self.config.heuristic.confidence_weight * confidence_score +
                       self.config.heuristic.reflection_weight * reflection_score)
        
        return final_value

    def _backpropagate(self, path: List[MCTSNode], value: float):
        for node in reversed(path):
            node.wins += (value + 1) / 2
            node.losses += (1 - value) / 2

    def _select_action(self, root: MCTSNode) -> int:
        if not root.children:
            return 0 # Should not happen if expanded
        return max(root.children.items(), key=lambda item: item[1].visit_count)[0]

    def get_tree_visualization_data(self, node, max_depth=3):
        if max_depth == 0 or not node.children:
            return {
                "name": self.model.pipeline.decode([node.action_token]) if node.action_token else "ROOT",
                "value": node.value,
                "visits": node.visit_count,
                "children": []
            }
        
        children_nodes = list(node.children.values())
        children_nodes.sort(key=lambda x: x.visit_count, reverse=True)

        children_data = [self.get_tree_visualization_data(child, max_depth - 1) for child in children_nodes[:5]]

        return {
            "name": self.model.pipeline.decode([node.action_token]) if node.action_token else "ROOT",
            "value": node.value,
            "visits": node.visit_count,
            "children": children_data
        }