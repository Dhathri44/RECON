import random
import numpy as np
import torch
from torch import nn
from src.replay_buffer import ReplayBuffer

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim,128), nn.ReLU(),
                                 nn.Linear(128,128), nn.ReLU(),
                                 nn.Linear(128,action_dim))
    def forward(self,x): return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim=3, lr=1e-3, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995,
                 target_update=100, device=None, seed=42):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(seed); random.seed(seed)
        self.q = QNetwork(state_dim,action_dim).to(self.device)
        self.target = QNetwork(state_dim,action_dim).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.optimizer = torch.optim.Adam(self.q.parameters(),lr=lr)
        self.buffer = ReplayBuffer()
        self.gamma, self.epsilon = gamma, epsilon
        self.epsilon_min, self.epsilon_decay = epsilon_min, epsilon_decay
        self.action_dim, self.steps, self.target_update = action_dim, 0, target_update

    def act(self,state,training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            x=torch.tensor(state,dtype=torch.float32,device=self.device).unsqueeze(0)
            return int(self.q(x).argmax(dim=1).item())

    def learn(self,batch_size=64):
        if len(self.buffer)<batch_size: return None
        s,a,r,ns,d=self.buffer.sample(batch_size)
        s=torch.tensor(s,device=self.device); a=torch.tensor(a,device=self.device)
        r=torch.tensor(r,device=self.device); ns=torch.tensor(ns,device=self.device)
        d=torch.tensor(d,device=self.device)
        q=self.q(s).gather(1,a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target=r+self.gamma*(1-d)*self.target(ns).max(1).values
        loss=nn.functional.smooth_l1_loss(q,target)
        self.optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(),5.0)
        self.optimizer.step()
        self.steps+=1
        if self.steps%self.target_update==0:
            self.target.load_state_dict(self.q.state_dict())
        self.epsilon=max(self.epsilon_min,self.epsilon*self.epsilon_decay)
        return float(loss.item())

    def save(self,path):
        torch.save({"q":self.q.state_dict(),"state_dim":self.q.net[0].in_features,
                    "action_dim":self.action_dim,"state_version":2},path)

    def load(self,path):
        data=torch.load(path,map_location=self.device)
        if int(data.get("state_dim", -1)) != self.q.net[0].in_features:
            raise ValueError("Checkpoint state dimension does not match agent")
        if int(data.get("action_dim", -1)) != self.action_dim:
            raise ValueError(f"Checkpoint has {data.get('action_dim')} actions; agent expects {self.action_dim}. Retrain the 3-action model.")
        if int(data.get("state_version", 1)) != 2:
            raise ValueError("Legacy DQN checkpoint uses the old receiver-state definition. Retrain with the updated RECON environment before using this checkpoint.")
        self.q.load_state_dict(data["q"]); self.target.load_state_dict(data["q"])
