import torch
import torch.nn.functional as F
import numpy as np
import asyncio
import math
from collections import Counter
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
        return sampled_q * 2 - 1 # Scale from [0, 1] to [-1, 1]

    def kl_divergence(self, p: float, q: float) -> float:
        if p == 0: return math.log(1 / (1 - q)) if q < 1 else float('inf')
        if p == 1: return math.log(1 / q) if q > 0 else float('inf')
        if q <= 0 or q >= 1: return float('inf')
        return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

    def select_child_kl_ucb(self) -> Tuple[int, 'MCTSNode']:
        if not self.children:
            return -1, None
            
        log_parent_visits = math.log(self.visit_count + 1) # Add 1 to avoid log(0)
        best_action, best_node, max_ucb = -1, None, -float('inf')

        for action, child in self.children.items():
            if child.visit_count == 0:
                return action, child
            
            # Smoothed win rate for the child
            p = (child.wins + 1) / (child.visit_count + 2)
            target = (log_parent_visits - math.log(child.visit_count)) / child.visit_count
            
            # Binary search to find the UCB value q
            q, high = p, 1.0
            for _ in range(8): # 8 iterations provide sufficient precision
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

class _SingleUniverse:
    def __init__(self, model: DeepRWKV, config: ProjectConfig, universe_id: int, strategy: str, visualizer: Optional[VisualizationServer] = None):
        self.model = model
        self.config = config
        self.universe_id = universe_id
        self.strategy = strategy
        self.visualizer = visualizer
        self.special_tokens = get_special_tokens(model.pipeline)
        self.root: Optional[MCTSNode] = None

    async def initialize(self, root_state):
        self.root = MCTSNode(parent=None, prior=1.0)
        self.root.state = clone_state(root_state)
        await self._expand(self.root, add_noise=True)

    async def search_step(self):
        for _ in range(self.config.mcts.simulations_per_step // self.config.mcts.parallel_leaves):
            await self._run_batch_simulations(self.root)
        
        if self.visualizer and self.universe_id == 0:
             tree_data = self._get_tree_visualization_data(self.root)
             self.visualizer.update_tree_data({
                 "universe_id": self.universe_id, 
                 "strategy": self.strategy,
                 "tree": tree_data
             })

    def advance_root(self) -> Optional[int]:
        action = self._select_action(self.root)
        if action is not None and action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None # Prune the tree
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
            if not node.state: # Should not happen in this implementation
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
                return path, node # Reached a dead end

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
        if not node.children:
            is_terminal = await self._expand(node, add_noise=False)
            if is_terminal:
                return 0.0 # Neutral value for terminal state
        
        value = 0.0
        if self.config.mcts.use_learned_value:
            try:
                learned_value = self.model.forward_value(node.state).item()
                value += self.config.mcts.learned_value_weight * learned_value
            except Exception:
                value += 0.0

        if self.config.mcts.learned_value_weight < 1.0:
            heuristic_value = await self._heuristic_rollout(node)
            value += (1.0 - self.config.mcts.learned_value_weight) * heuristic_value
        
        return value

    async def _expand(self, node: MCTSNode, add_noise: bool) -> bool:
        logits, _ = self.model.forward_policy([0], clone_state(node.state))
        
        if torch.argmax(logits) == 0:
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
        last_token = node.action_token if node.action_token is not None else 0
        
        total_log_prob = 0
        rollout_tokens = []
        
        for _ in range(self.config.heuristic.rollout_depth):
            logits, rollout_state = self.model.forward_policy([last_token], rollout_state)
            log_probs = F.log_softmax(logits, dim=-1)
            
            max_log_prob, next_token_tensor = torch.max(log_probs, dim=-1)
            
            total_log_prob += max_log_prob.item()
            next_token = next_token_tensor.item()
            
            rollout_tokens.append(next_token)
            last_token = next_token

            if last_token == self.special_tokens["newline"] or last_token == 0:
                break
        
        avg_log_prob = total_log_prob / (len(rollout_tokens) if rollout_tokens else 1)
        confidence_score = math.tanh(avg_log_prob / 3.0)

        reflection_prompt_tokens = self.model.pipeline.encode(self.config.heuristic.reflection_prompt)
        logits, _ = self.model.forward_policy(reflection_prompt_tokens, rollout_state)
        
        probs = F.softmax(logits, dim=-1)
        p_yes = probs[0, self.special_tokens["yes"]].item()
        p_no = probs[0, self.special_tokens["no"]].item()
        reflection_score = (p_yes - p_no) / (p_yes + p_no + 1e-6)

        final_value = (self.config.heuristic.confidence_weight * confidence_score +
                       self.config.heuristic.reflection_weight * reflection_score)
        
        return final_value

    def _backpropagate(self, path: List[MCTSNode], value: float):
        for node in reversed(path):
            node.wins += (value + 1) / 2
            node.losses += (1 - value) / 2

    def _select_action(self, root: MCTSNode) -> Optional[int]:
        if not root.children: return None
        return max(root.children, key=lambda action: root.children[action].visit_count)

    def _get_tree_visualization_data(self, node, max_depth=3):
        if max_depth == 0 or not node.children:
            return {"name": self.model.pipeline.decode([node.action_token]) if node.action_token else "ROOT", "value": node.value, "visits": node.visit_count, "children": []}
        
        children_nodes = list(node.children.values())
        children_nodes.sort(key=lambda x: x.visit_count, reverse=True)
        children_data = [self._get_tree_visualization_data(child, max_depth - 1) for child in children_nodes[:5]]
        return {"name": self.model.pipeline.decode([node.action_token]) if node.action_token else "ROOT", "value": node.value, "visits": node.visit_count, "children": children_data}


class ParallelMCTSEngine:
    def __init__(self, config: ProjectConfig, num_universes: int = 3):
        self.config = config
        self.num_universes = num_universes
        
        print("[DeepRWKV] Initializing DeepRWKV Model...")
        self.model = DeepRWKV(config)
        
        print("[DeepRWKV] Initializing Visualization Server...")
        self.visualizer = VisualizationServer()
        
        self.universes: List[_SingleUniverse] = []
        
        strategies = ['PUCT', 'KL_UCB', 'Thompson']
        for i in range(num_universes):
            strategy = strategies[i % len(strategies)]
            print(f"[Universe {i}] Assigning search strategy: {strategy}")
            self.universes.append(_SingleUniverse(self.model, config, i, strategy, self.visualizer))

    async def run(self, prompt: str, max_new_tokens: int = 256) -> Dict:
        self.visualizer.run()
        
        print("[DeepRWKV] Bootstrapping all universes...")
        prompt_tokens = self.model.pipeline.encode(prompt)
        _, root_state = self.model.forward_policy(prompt_tokens, None)
        
        init_tasks = [u.initialize(root_state) for u in self.universes]
        await asyncio.gather(*init_tasks)

        full_responses = {i: [] for i in range(self.num_universes)}
        consensus_tokens = []
        
        for step in range(max_new_tokens):
            print(f"\n--- Reasoning Step {step + 1} ---")
            
            search_tasks = [u.search_step() for u in self.universes]
            await asyncio.gather(*search_tasks)
            
            actions = [u.advance_root() for u in self.universes]
            valid_actions = [a for a in actions if a is not None]

            if not valid_actions:
                print("[DeepRWKV] All universes failed to select an action. Terminating.")
                break

            consensus_action = Counter(valid_actions).most_common(1)[0][0]
            consensus_tokens.append(consensus_action)
            
            vote_log = {f"U{i} ({u.strategy})": f"'{self.model.pipeline.decode([a])}'" if a else "FAIL" for i, (u, a) in enumerate(zip(self.universes, actions))}
            print(f"[Voting] Votes: {vote_log} -> Consensus: '{self.model.pipeline.decode([consensus_action])}'")
            
            sync_tasks = []
            for i, (universe, action) in enumerate(zip(self.universes, actions)):
                full_responses[i].append(action if action is not None else 0)
                if action != consensus_action:
                    if universe.root.parent and consensus_action in universe.root.parent.children:
                        universe.root = universe.root.parent.children[consensus_action]
                        universe.root.parent = None
                    else:
                        print(f"[Sync Error] Universe {i} cannot be synced.")

                if universe.root and not universe.root.children:
                    sync_tasks.append(universe._expand(universe.root, add_noise=True))

            if sync_tasks: await asyncio.gather(*sync_tasks)

            if consensus_action == 0 or self.model.pipeline.decode([consensus_action]) == "\n\n":
                print("\n[DeepRWKV] Termination condition met.")
                break
        
        return {
            "consensus_response": self.model.pipeline.decode(consensus_tokens),
            "universe_responses": {i: self.model.pipeline.decode(resp) for i, resp in full_responses.items()}
        }