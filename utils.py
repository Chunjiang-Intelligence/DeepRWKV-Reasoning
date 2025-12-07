import torch
import torch.nn.functional as F
import numpy as np

def clone_state(state):
    if state is None:
        return None
    return [t.clone() for t in state]

def get_special_tokens(pipeline):
    return {
        "yes": pipeline.encode(" Yes")[0],
        "no": pipeline.encode(" No")[0],
        "newline": pipeline.encode("\n")[0]
    }

def calculate_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).item()

def add_dirichlet_noise(priors, alpha, epsilon):
    noise = np.random.dirichlet([alpha] * len(priors))
    return (1 - epsilon) * priors + epsilon * noise