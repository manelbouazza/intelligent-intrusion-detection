import joblib
import numpy as np
import torch

# ---------- Load models ----------
supervised_model = joblib.load("models/supervised_model.pkl")
unsupervised_model = joblib.load("models/unsupervised_model.pkl")

# ---------- RL (Q-learning) ----------
q_data = joblib.load("models/q_learning.pkl")
Q = q_data["Q"]
bins_config = q_data["bins_config"]
n_actions = q_data["n_actions"]

# ---------- DRL (DQN) ----------
from torch import nn

class DQN(nn.Module):
    def __init__(self, input_dim=5, n_actions=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.fc(x)


input_dim = 5  # nombre de features utilisé à l’entraînement
n_actions = 2
policy_net = DQN(input_dim=input_dim, n_actions=n_actions)
policy_net.load_state_dict(torch.load("models/dqn_model.pth", map_location="cpu"))
policy_net.eval()


