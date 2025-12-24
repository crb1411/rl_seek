import os

import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from dataclasses import asdict
from typing import List, Optional
import logging

from advantage_normalizer import AdvantageNormalizer
from models import ActorCritic
from training_utils import init_wandb, save_checkpoint, load_checkpoint
from inference import evaluate_policy
from config import TrainingConfig, Advantage_Policy
from utils import creat_subdir, format_rollout_log_str, format_head_tail, setup_logger

logger = logging.getLogger("rl.training")



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

    def finish_path(self, gamma=0.99, lam=0.95, strategy: Advantage_Policy=Advantage_Policy.PPO_GAE):
        """
        Compute G_t and the chosen policy target (G_list).
        strategy in Advantage. 0,1,2,3 maps to:
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

        if strategy == Advantage_Policy.RETURN:
            g_list = returns
        elif strategy == Advantage_Policy.ADVANTAGE:
            g_list = returns - self.values[:n]
        elif strategy == Advantage_Policy.TD_ERROR:
            g_list = (
                self.rewards[:n]
                + gamma * next_values * (1 - self.dones[:n])
                - self.values[:n]
            )
        elif strategy == Advantage_Policy.PPO_GAE:
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
        elif strategy == Advantage_Policy.ADVANTAGE_DISCOUNTED:
            adv_list = returns - self.values[:n]
            for i in reversed(range(n)):
                if i == n - 1:
                    continue
                adv_list[i] = adv_list[i] + gamma * lam * adv_list[i + 1] * (1 - self.dones[i])
            g_list = adv_list
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
    adv_normalizer = AdvantageNormalizer(momentum=0) # 等于 0 退化成不参考历史的 mean, std
    policy_name = (
        config.policy_target.name
        if isinstance(config.policy_target, Advantage_Policy)
        else str(config.policy_target)
    )
    training_config = asdict(config)
    training_config["policy_target"] = policy_name
    training_config["save_root"] = str(save_root)
    training_config["checkpoint_dir"] = str(checkpoint_dir)
    training_config["resume_path"] = str(resume_path) if resume_path else None

    valid_policy_targets = set(Advantage_Policy)
    if config.policy_target not in valid_policy_targets:
        raise ValueError(f"policy_target must be one of {list(Advantage_Policy)}")

    buf = RolloutBuffer(config.steps_per_epoch + env_max_steps, obs_dim, normalizer=adv_normalizer)

    start_epoch = 0
    run = None
    if resume_path and resume_path.exists():
        last_epoch, _ = load_checkpoint(resume_path, ac, optimizer, adv_normalizer)
        start_epoch = last_epoch + 1
        logger.info(f"Resumed from {resume_path}, starting at epoch {start_epoch}")
    elif resume_path:
        logger.info(f"Resume path {resume_path} not found. Starting fresh.")
    else:
        logger.info("Starting fresh.")

    # =========================
    # Train run
    # =========================
    base_name = config.run_name or f"{policy_name}_{'with_clip' if config.use_clip else 'no_clip'}"
    if config.run_name is None:
        config.run_name = base_name
    training_config["run_name"] = base_name

    run = init_wandb(
        config.use_wandb,
        save_root=save_root,
        config=training_config,
        run_name=base_name,
        resume_id=(resume_path.stem + "_train") if resume_path else None,
        entity=config.wandb_entity,
        project=config.wandb_project,
    )
    logger.info(f"[wandb] train run   : {base_name}")

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
        
        if run is not None:
            first_done_indices = np.where(buf.dones[:buf.ptr] > 0)[0]
            if first_done_indices.size > 0:
                first_end = int(first_done_indices[0])
            else:
                first_end = min(buf.ptr, 10) - 1
            first_end = max(first_end, -1)
            first_slice = slice(0, first_end + 1)
            first_len = first_slice.stop - first_slice.start

            episodes = int(np.count_nonzero(buf.dones[:buf.ptr]))
            episodes = max(episodes, 1)
            rollout_log = {
                "rollout/epoch": epoch + 1,
                "rollout/steps": int(buf.ptr),
                "rollout/episodes": episodes,
                "rollout/avg_steps_per_episode": float(buf.ptr / episodes),
                "rollout/first_episode_len": int(first_len),
                "rollout/first_values": format_head_tail(buf.values[first_slice].tolist()),
                "rollout/first_returns": format_head_tail(buf.returns[first_slice].tolist()),
                "rollout/first_advantages": format_head_tail(buf.advantages[first_slice].tolist()),
            }
            logger.info(f"\n{format_rollout_log_str(rollout_log)}")
            # run.log(
            #     rollout_log
            # )
            
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
                adv = advantages_buf[mb_idx].detach()
                if config.use_clip:
                    ratio = torch.exp(logp - logp_buf[mb_idx])
                    unclipped = ratio * adv
                    clipped = torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio) * adv
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    # logger.info(f'ratio: {ratio.mean().item()}')
                else:
                    policy_loss = -(logp * adv).mean()
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
        logger.info(
            f"Epoch {current_epoch}: TestReward={test_reward:.1f}\n"
            f"pi_loss={current_policy_loss:.4f} vf_loss={current_value_loss:.4f} ent={current_entropy:.4f}\n"
            f"{"*" * 52}\n"
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
                        policy_target=Advantage_Policy.ADVANTAGE_DISCOUNTED,
                        use_wandb=True,
                        train_iters=10,
                        use_clip=True,
                        save_root="/data/seek/rl_rundata/logs_ac"
                    )
    
    save_root = train_config.save_root if train_config.save_root else "/mnt/seek/rundata/1223"
    policy_str = train_config.policy_target.name if isinstance(train_config.policy_target, Advantage_Policy) else str(train_config.policy_target)
    prefix = f"{policy_str}_with_clip" if train_config.use_clip else f"{policy_str}_no_clip"
    train_config.save_root = creat_subdir(
                                base_dir=save_root, 
                                prefix=prefix, 
                                create=True,
                                time=True,
                            )
    suffix = train_config.save_root[-9:] if len(train_config.save_root) >= 9 else train_config.save_root
    train_config.run_name = (
        f"{policy_str}_with_clip_{suffix}"
        if train_config.use_clip
        else f"{policy_str}_no_clip_{suffix}"
    )
    train_config.checkpoint_dir = os.path.join(train_config.save_root, "checkpoints")
    logger = setup_logger(name='rl.training', log_dir=train_config.save_root, filename='training.log')
    logger.info(train_config)
    ppo_train(train_config)     # 训练
    # 
    """
    我发现训练后期模型会忘记坏状态下的策略， 
    因为考虑如果（极限）优化后期大部分状态都是好的，所以优化到后面，模型会将坏的结果可能也打一个很高的 value，
    那么此时模型输入一个坏结果，就会干扰到整个模型的训练，让极其不稳定。 
    这是极限情况，可以说明模型训练不鲁棒，反之如果是最优点的话，不会出现扰动，
    
    后续可以针对此，可以记录下一定的坏状态，增加到value的训练中
    """
