import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
from deep_rwkv.config import ProjectConfig
from deep_rwkv.modeling import DeepRWKV

class PRMDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.data = []
        with open(config.training.dataset_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = torch.tensor(item['tokens'], dtype=torch.long)
        reward = torch.tensor([item['reward']], dtype=torch.float32)
        return tokens, reward

def collate_fn(batch, model_instance):
    tokens_list, rewards = zip(*batch)
    
    hidden_states = []
    with torch.no_grad():
        for tokens in tokens_list:
            tokens = tokens.to(model_instance.backbone.device)
            # We train the value head on the state AFTER the sequence
            _, state = model_instance.backbone.forward(tokens, None)
            last_ffn_state = state[-1].detach()
            hidden_states.append(last_ffn_state)

    return torch.stack(hidden_states), torch.cat(rewards)

def train():
    config = ProjectConfig()
    model = DeepRWKV(config)
    
    for param in model.backbone.parameters():
        param.requires_grad = False
        
    dataset = PRMDataset(config)
    data_loader = DataLoader(
        dataset, 
        batch_size=config.training.batch_size, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, model)
    )

    optimizer = optim.AdamW(model.value_head.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    criterion = nn.MSELoss()
    
    model.value_head.train()
    
    for epoch in range(config.training.epochs):
        print(f"Epoch {epoch + 1}/{config.training.epochs}")
        for hidden_states, rewards in tqdm(data_loader):
            hidden_states = hidden_states.to(model.backbone.device)
            rewards = rewards.to(model.backbone.device)
            
            optimizer.zero_grad()
            
            predicted_values = model.value_head(hidden_states)
            loss = criterion(predicted_values, rewards)
            
            loss.backward()
            optimizer.step()

    torch.save(model.value_head.state_dict(), config.training.save_path)
    print(f"Value Head saved to {config.training.save_path}")

if __name__ == "__main__":
    train()