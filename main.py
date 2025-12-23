import os
import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from dataclasses import asdict
from typing import List, Optional

from advantage_normalizer import AdvantageNormalizer
from models import ActorCritic
from training_utils import init_wandb, save_checkpoint, load_checkpoint
from inference import evaluate_policy
from config import TrainingConfig

def format_head_tail(values: List[float], n: int = 5) -> str:
    """Return first/last n values (ellipsis in the middle if truncated)."""
    if not values:
        return "[]"
    if len(values) <= 2 * n:
        parts = [f"{v:.3f}" for v in values]
    else:
        parts = [f"{v:.3f}" for v in values[:n]] + ["..."] + [f"{v:.3f}" for v in values[-n:]]
    return "[" + ", ".join(parts) + "]"


def select_device(pref: str) -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)

# --- 2. Rollout Buffer ---
class RolloutBuffer:
    def __init__(self, size, obs_dim, normalizer: AdvantageNormalizer | None = None):
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.actions = np.zeros(size, np.int32)
        self.logprobs = np.zeros(size, np.float32)
        self.rewards = np.zeros(size, np.float32)
        self.dones = np.zeros(size, np.float32)
        self.timeouts = np.zeros(size, np.float32)
        self.values = np.zeros(size, np.float32)
        self.advantages = np.zeros(size, np.float32)
        self.returns = np.zeros(size, np.float32)
        self.ptr = 0
        self.max_size = size
        self.normalizer = normalizer

    def store(self, obs, action, logp, reward, done, value, timeout=False):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.logprobs[self.ptr] = logp
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.timeouts[self.ptr] = float(timeout)
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
                if self.timeouts[i]:
                    # g_next = 1.0 / (1.0 - gamma)
                    g_next = (self.values[i] - 1.0) / gamma
                else:
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

        if self.normalizer is not None:
            g_torch = torch.tensor(g_list, dtype=torch.float32)
            g_list = self.normalizer.normalize(g_torch).cpu().numpy()

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
def ppo_train(config: TrainingConfig):
    save_root = Path(config.save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(config.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = save_root / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_path = Path(config.resume_path) if config.resume_path else None
    env = gym.make(config.env_name)
    spec = getattr(env, "spec", None)
    env_max_steps = getattr(spec, "max_episode_steps", None)
    if env_max_steps is None:
        env_max_steps = getattr(env, "_max_episode_steps", None)
    if env_max_steps is None:
        env_max_steps = config.steps_per_epoch
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    device = select_device(config.device)
    ac = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(ac.parameters(), lr=config.pi_lr)
    adv_normalizer = AdvantageNormalizer()
    training_config = asdict(config)
    training_config["save_root"] = str(save_root)
    training_config["checkpoint_dir"] = str(checkpoint_dir)
    training_config["resume_path"] = str(resume_path) if resume_path else None

    valid_policy_targets = {"return", "advantage", "td_error", "ppo_gae"}
    if config.policy_target not in valid_policy_targets:
        raise ValueError(f"policy_target must be one of {valid_policy_targets}")

    buf = RolloutBuffer(config.steps_per_epoch + env_max_steps, obs_dim, normalizer=adv_normalizer)

    start_epoch = 0
    if resume_path and resume_path.exists():
        last_epoch, _ = load_checkpoint(resume_path, ac, optimizer, adv_normalizer)
        start_epoch = last_epoch + 1
        print(f"Resumed from {resume_path}, starting at epoch {start_epoch}")
    elif resume_path:
        print(f"Resume path {resume_path} not found. Starting fresh.")
    else:
        print("Starting fresh.")

        run = init_wandb(
            config.use_wandb,
            save_root=save_root,
            config=training_config,
            run_name=config.run_name,
            resume_id=(resume_path.stem + "_train") if resume_path else None,
            entity=config.wandb_entity,
            project=config.wandb_project,
        )
        
        run_rollout = init_wandb(
            config.use_wandb,
            save_root=save_root,
            run_name=config.run_name + "_rollout",
            resume_id=(resume_path.stem + "_rollout") if resume_path else None,
            entity=config.wandb_entity,
            project=config.wandb_project,
        )

    for epoch in range(start_epoch, config.epochs):
        buf.reset()
        steps_collected = 0
        while steps_collected < config.steps_per_epoch:
            obs, _ = env.reset()
            done = False
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    action, logp, value = ac.get_action(obs_tensor)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                buf.store(
                    obs, action, logp.item(), reward, done, value.item(), timeout=truncated
                )
                obs = next_obs
                steps_collected += 1
                if done:
                    break
        # 采样完成，计算G_list
        buf.finish_path(gamma=config.gamma, lam=config.lam, strategy=config.policy_target)
        
        if run_rollout is not None:
            run_rollout.log(
                {
                    
                }
            )

        # 开始训练（使用 -log_prob * G_list 或 PPO-Clip）
        obs_buf, act_buf, logp_buf, ret_buf, advantages_buf = buf.get()
        obs_buf = obs_buf.to(device)
        act_buf = act_buf.to(device)
        logp_buf = logp_buf.to(device)
        ret_buf = ret_buf.to(device)
        advantages_buf = advantages_buf.to(device)
        n = obs_buf.shape[0]
        idx = np.arange(n)
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropies = []

        for _ in range(config.train_iters):
            np.random.shuffle(idx)
            for start in range(0, n, config.batch_size):
                end = start + config.batch_size
                mb_idx = idx[start:end]
                logp, entropy, values = ac.evaluate_actions(
                    obs_buf[mb_idx], act_buf[mb_idx]
                )
                if config.use_clip:
                    ratio = torch.exp(logp - logp_buf[mb_idx])
                    unclipped = ratio * advantages_buf[mb_idx]
                    clipped = torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio) * advantages_buf[mb_idx]
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    print(f'ratio: {ratio.mean().item()}')
                else:
                    policy_loss = -(logp * advantages_buf[mb_idx]).mean()
                critic_loss = ((ret_buf[mb_idx] - values) ** 2).mean()
                entropy_coef = max(0.02 * (1 - epoch / config.epochs), 0.0)
                loss = policy_loss + 0.5 * critic_loss - entropy_coef * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(critic_loss.item() * 0.5)
                epoch_entropies.append(entropy.mean().item() * entropy_coef)

        # 简单评估（每轮打印一次）
        test_reward = evaluate_policy(ac, env, render=config.render_test, device=device)
        mean_policy_loss = float(np.mean(epoch_policy_losses)) if epoch_policy_losses else 0.0
        mean_value_loss = float(np.mean(epoch_value_losses)) if epoch_value_losses else 0.0
        mean_entropy = float(np.mean(epoch_entropies)) if epoch_entropies else 0.0
        current_policy_loss = epoch_policy_losses[-1] if epoch_policy_losses else 0.0
        current_value_loss = epoch_value_losses[-1] if epoch_value_losses else 0.0
        current_entropy = epoch_entropies[-1] if epoch_entropies else 0.0
        current_epoch = epoch + 1
        print(
            f"Epoch {current_epoch}: TestReward={test_reward:.1f}\n"
            f"pi_loss={current_policy_loss:.4f} vf_loss={current_value_loss:.4f} ent={current_entropy:.4f}\n"
            f"mean_pi_loss={mean_policy_loss:.4f} mean_vf_loss={mean_value_loss:.4f} mean_ent={mean_entropy:.4f}\n\n"
        )

        save_checkpoint(checkpoint_dir / "latest.pt", ac, optimizer, adv_normalizer, epoch, training_config)
        if config.checkpoint_freq > 0 and current_epoch % config.checkpoint_freq == 0:
            ckpt_path = checkpoint_dir / f"epoch_{current_epoch}.pt"
            save_checkpoint(ckpt_path, ac, optimizer, adv_normalizer, epoch, training_config)

        if run is not None:
            run.log({
                "epoch": current_epoch,
                "test_reward": test_reward,
                "policy_loss": current_policy_loss,
                "value_loss": current_value_loss,
                "entropy": current_entropy,
            })

    if run is not None:
        run.finish()

# --- 5. 运行训练与推理 ---
if __name__ == "__main__":
    train_config = TrainingConfig(
                        epochs=100,
                        use_wandb=True,
                    )
    save_root = train_config.save_root if train_config.save_root else "/mnt/seek/rundata/1223"
    from utils import creat_subdir
    train_config.save_root = creat_subdir(
                                base_dir=save_root, 
                                prefix=f"{TrainingConfig.policy_target}", 
                                create=True,
                                time=True,
                            )
    train_config.run_name = f"{train_config.policy_target}_with_clip" if train_config.use_clip else f"{train_config.policy_target}_no_clip"
    train_config.checkpoint_dir = os.path.join(train_config.save_root, "checkpoints")
    print(train_config)
    ppo_train(train_config)   # 可调大epochs训练更好
    # 推理展示（训练结束后手动调用）
    # env = gym.make("CartPole-v1", render_mode='human')
    # ac = ... # 你的模型
    # evaluate_policy(ac, env, episodes=5, render=True)
