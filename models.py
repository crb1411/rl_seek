import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        self.net_value = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        self.policy_head = nn.Linear(64, act_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x_policy = self.net(x)
        logits = self.policy_head(x_policy)
        value = self.get_value(x)
        return logits, value

    def get_value(self, x):
        """Evaluate only the critic, without running the policy network."""
        x_value = self.net_value(x)
        value = self.value_head(x_value)
        return value.squeeze(-1)

    def get_action(self, obs):
        logits, value = self.forward(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value

    def evaluate_actions(self, obs, act):
        logits, value = self.forward(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(act)
        entropy = dist.entropy()
        return log_prob, entropy, value
