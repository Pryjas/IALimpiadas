import torch
import torch.nn as nn
import torch.nn.functional as F
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PongNet(nn.Module):
    
    def __init__(self):
        super(PongNet, self).__init__()
        
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def save(self, filepath='model/pong_model.pth'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.state_dict(), filepath)
        
    def load(self, filepath='model/pong_model.pth'):
        self.load_state_dict(torch.load(filepath, map_location=device))
        self.eval()


class SimpleTrainer:
    
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=1e-5
        )
        self.loss_fn = nn.MSELoss()
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
    def train_step(self, states, actions, rewards, next_states, dones):
        gamma = 0.95
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            next_q = self.model(next_states).max(1)[0]
            target_q = rewards + gamma * next_q * (1 - dones)
        
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
