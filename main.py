import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- 1. Actor-Critic 网络 ---
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh()
        )
        self.net_value = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh()
        )
        self.policy_head = nn.Linear(64, act_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x_policy = self.net(x)
        logits = self.policy_head(x_policy)
        x_value = self.net_value(x)
        value = self.value_head(x_value)
        return logits, value.squeeze(-1)

    def get_action(self, obs):
        logits, value = self.forward(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action.item(), logp, value

    def evaluate_actions(self, obs, act):
        logits, value = self.forward(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        logp = dist.log_prob(act)
        entropy = dist.entropy()
        return logp, entropy, value

# --- 2. Rollout Buffer ---
class RolloutBuffer:
    def __init__(self, size, obs_dim):
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.actions = np.zeros(size, np.int32)
        self.logprobs = np.zeros(size, np.float32)
        self.rewards = np.zeros(size, np.float32)
        self.dones = np.zeros(size, np.float32)
        self.values = np.zeros(size, np.float32)
        self.advantages = np.zeros(size, np.float32)
        self.returns = np.zeros(size, np.float32)
        self.ptr = 0
        self.max_size = size

    def store(self, obs, action, logp, reward, done, value):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.logprobs[self.ptr] = logp
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.ptr += 1

    def finish_path(self, gamma=0.99, lam=0.95, strategy: str = "advantage"):
        """
        Compute G_t and the chosen policy target (G_list).
        strategy in {"return", "advantage", "td_error", "ppo_gae"} maps to:
        - return: G_t
        - advantage: G_t - V(s_t)
        - td_error: r_t + gamma * V(s_{t+1}) - V(s_t)
        - ppo_gae: sum_k (gamma*lam)^k * delta_{t+k}, where delta_t is TD error
        """
        n = self.ptr
        if n == 0:
            return

        returns = np.zeros(n, dtype=np.float32)
        g_next = 0.0
        for i in reversed(range(n)):
            if self.dones[i]:
                g_next = 0.0
            g_next = self.rewards[i] + gamma * g_next
            returns[i] = g_next

        self.returns[:n] = returns

        next_values = np.concatenate(
            [self.values[1:n], np.array([0.0], dtype=np.float32)]
        )

        if strategy == "return":
            g_list = returns
        elif strategy == "advantage":
            g_list = returns - self.values[:n]
        elif strategy == "td_error":
            g_list = (
                self.rewards[:n]
                + gamma * next_values * (1 - self.dones[:n])
                - self.values[:n]
            )
        elif strategy == "ppo_gae":
            gae = 0.0
            adv = np.zeros(n, dtype=np.float32)
            for i in reversed(range(n)):
                delta = (
                    self.rewards[i]
                    + gamma * next_values[i] * (1 - self.dones[i])
                    - self.values[i]
                )
                gae = delta + gamma * lam * (1 - self.dones[i]) * gae
                adv[i] = gae
            g_list = adv
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        self.advantages[:n] = g_list

    def get(self):
        return (
            torch.tensor(self.obs[:self.ptr], dtype=torch.float32),
            torch.tensor(self.actions[:self.ptr], dtype=torch.long),
            torch.tensor(self.logprobs[:self.ptr], dtype=torch.float32),
            torch.tensor(self.returns[:self.ptr], dtype=torch.float32),
            torch.tensor(self.advantages[:self.ptr], dtype=torch.float32),
        )

    def reset(self):
        self.ptr = 0
        

# --- 3. 训练主循环 ---
def ppo_train(env_name='CartPole-v1', steps_per_epoch=2048, epochs=50,
              gamma=0.99, lam=0.95, pi_lr=3e-4,
              train_iters=10, batch_size=64, render_test=False,
              policy_target: str = "advantage"):

    env = gym.make(env_name)
    spec = getattr(env, "spec", None)
    env_max_steps = getattr(spec, "max_episode_steps", None)
    if env_max_steps is None:
        env_max_steps = getattr(env, "_max_episode_steps", None)
    if env_max_steps is None:
        env_max_steps = steps_per_epoch
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    ac = ActorCritic(obs_dim, act_dim)
    optimizer = optim.Adam(ac.parameters(), lr=pi_lr)

    valid_policy_targets = {"return", "advantage", "td_error", "ppo_gae"}
    if policy_target not in valid_policy_targets:
        raise ValueError(f"policy_target must be one of {valid_policy_targets}")

    buf = RolloutBuffer(steps_per_epoch + env_max_steps, obs_dim)

    for epoch in range(epochs):
        buf.reset()
        steps_collected = 0
        while steps_collected < steps_per_epoch:
            obs, _ = env.reset()
            done = False
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                with torch.no_grad():
                    action, logp, value = ac.get_action(obs_tensor)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                buf.store(obs, action, logp.item(), reward, done, value.item())
                obs = next_obs
                steps_collected += 1
                if done:
                    break
        # 采样完成，计算G_list
        buf.finish_path(gamma=gamma, lam=lam, strategy=policy_target)

        # 开始训练（使用 -log_prob * G_list）
        obs_buf, act_buf, _, ret_buf, g_buf = buf.get()
        n = obs_buf.shape[0]
        idx = np.arange(n)

        for _ in range(train_iters):
            np.random.shuffle(idx)
            for start in range(0, n, batch_size):
                end = start + batch_size
                mb_idx = idx[start:end]
                logp, entropy, values = ac.evaluate_actions(
                    obs_buf[mb_idx], act_buf[mb_idx]
                )
                policy_loss = -(logp * g_buf[mb_idx]).mean()
                critic_loss = ((ret_buf[mb_idx] - values) ** 2).mean()
                loss = policy_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 简单评估（每轮打印一次）
        test_reward = evaluate_policy(ac, env, render=render_test)
        print(f"Epoch {epoch+1}: TestReward={test_reward:.1f}")

# --- 4. 推理代码 ---
def evaluate_policy(ac, env, episodes=3, render=False):
    total_reward = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                logits, _ = ac(obs_tensor)
                probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(probs).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if render:
                env.render()
    return total_reward / episodes

# --- 5. 运行训练与推理 ---
if __name__ == "__main__":
    ppo_train(epochs=20)   # 可调大epochs训练更好
    # 推理展示（训练结束后手动调用）
    # env = gym.make("CartPole-v1", render_mode='human')
    # ac = ... # 你的模型
    # evaluate_policy(ac, env, episodes=5, render=True)
