import random
from dataclasses import dataclass
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym


# =========================
# 1) Q 网络：输入 s，输出每个动作的 Q(s, a)
# =========================
class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, s):
        return self.net(s)  # [B, action_dim]


# =========================
# 2) ReplayBuffer：存 transition (s,a,r,s2,done)
# =========================
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, done = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(s2, dtype=np.float32),
            np.array(done, dtype=np.float32),  # 0/1
        )

    def __len__(self):
        return len(self.buf)


@dataclass
class CFG:
    env_id: str = "CartPole-v1"
    seed: int = 0
    gamma: float = 0.99
    lr: float = 1e-3

    total_steps: int = 50_000
    start_learning: int = 1_000       # buffer 里有一定数据后再训练
    batch_size: int = 64

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000 # 线性退火到 epsilon_end

    target_update_every: int = 1_000  # 每 N step 同步一次 target 网络
    replay_capacity: int = 50_000


def linear_epsilon(step, cfg: CFG):
    # 线性从 eps_start -> eps_end
    t = min(step / cfg.epsilon_decay_steps, 1.0)
    return cfg.epsilon_start + t * (cfg.epsilon_end - cfg.epsilon_start)


def main():
    cfg = CFG()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = gym.make(cfg.env_id)
    obs, _ = env.reset(seed=cfg.seed)
    state_dim = obs.shape[0]
    action_dim = env.action_space.n

    q_net = QNet(state_dim, action_dim)
    target_net = QNet(state_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())  # 初始同步
    target_net.eval()

    optim = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)
    replay = ReplayBuffer(cfg.replay_capacity)

    episode_reward = 0.0
    ep = 0

    for step in range(1, cfg.total_steps + 1):
        # =========================
        # A) 交互：用 ε-greedy 选动作，得到 transition
        # =========================
        eps = linear_epsilon(step, cfg)

        if random.random() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                s_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # [1, state_dim]
                q_values = q_net(s_t)  # [1, action_dim]
                a = int(torch.argmax(q_values, dim=1).item())

        obs2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        # 存入 replay：这就是“训练数据”的来源
        replay.push(obs, a, r, obs2, float(done))

        obs = obs2
        episode_reward += r

        if done:
            obs, _ = env.reset()
            ep += 1
            if ep % 10 == 0:
                print(f"step={step:6d}  ep={ep:4d}  last_ep_reward={episode_reward:6.1f}  eps={eps:.3f}  replay={len(replay)}")
            episode_reward = 0.0

        # =========================
        # B) 学习：从 replay 采样 batch，构造 loss 并更新
        # =========================
        if len(replay) >= cfg.start_learning:
            s, a, r, s2, done_b = replay.sample(cfg.batch_size)

            s = torch.tensor(s, dtype=torch.float32)      # [B, state_dim]
            a = torch.tensor(a, dtype=torch.int64)        # [B]
            r = torch.tensor(r, dtype=torch.float32)      # [B]
            s2 = torch.tensor(s2, dtype=torch.float32)    # [B, state_dim]
            done_b = torch.tensor(done_b, dtype=torch.float32)  # [B]

            # ---- 1) 当前估计 q = Q_theta(s,a) ----
            q_all = q_net(s)  # [B, action_dim]
            q = q_all.gather(1, a.unsqueeze(1)).squeeze(1)  # [B]

            # ---- 2) TD target: y = r + gamma*(1-done)*max_a Q_target(s2,a) ----
            with torch.no_grad():
                q2_all = target_net(s2)          # [B, action_dim]
                q2_max = q2_all.max(dim=1).values  # [B]
                y = r + cfg.gamma * (1.0 - done_b) * q2_max

            # ---- 3) loss：Huber(q - y) ----
            loss = F.smooth_l1_loss(q, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

        # =========================
        # C) target 网络同步：每隔 N step
        # =========================
        if step % cfg.target_update_every == 0:
            target_net.load_state_dict(q_net.state_dict())

    env.close()


if __name__ == "__main__":
    main()
