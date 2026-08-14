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
        self.old_log_probs = np.zeros(size, np.float32)
        self.rewards = np.zeros(size, np.float32)
        self.terminated = np.zeros(size, np.float32)
        self.truncated = np.zeros(size, np.float32)
        self.old_values = np.zeros(size, np.float32)
        # Only needed when an episode is cut off by the environment time limit.
        # For ordinary transitions, V_old(s_{t+1}) is old_values[t + 1].
        self.timeout_bootstrap_values = np.zeros(size, np.float32)
        # Unnormalized actor signal, e.g. GAE's A_hat_t.
        self.raw_advantages = np.zeros(size, np.float32)
        # Actor signal after optional normalization.
        self.advantages = np.zeros(size, np.float32)
        # Critic regression target. In standard PPO this is A_hat_t + V_old(s_t).
        self.returns = np.zeros(size, np.float32)
        self.ptr = 0
        self.max_size = size
        self.normalizer = normalizer

    def store(
        self,
        obs,
        action,
        old_log_prob,
        reward,
        terminated,
        truncated,
        old_value,
        timeout_bootstrap_value=0.0,
    ):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.old_log_probs[self.ptr] = old_log_prob
        self.rewards[self.ptr] = reward
        self.terminated[self.ptr] = float(terminated)
        self.truncated[self.ptr] = float(truncated)
        self.old_values[self.ptr] = old_value
        self.timeout_bootstrap_values[self.ptr] = timeout_bootstrap_value
        self.ptr += 1

    def _next_state_values(self, n: int) -> np.ndarray:
        """Build V_old(s_{t+1}) without another forward on normal steps."""
        next_values = np.zeros(n, dtype=np.float32)
        if n > 1:
            next_values[:-1] = self.old_values[1:n]

        # old_values[i + 1] after an episode end belongs to the next reset
        # episode. A pure time-limit truncation instead bootstraps from its
        # actual next_obs. If both flags are true, termination takes priority.
        timeout_mask = (
            self.truncated[:n].astype(bool)
            & ~self.terminated[:n].astype(bool)
        )
        next_values[timeout_mask] = self.timeout_bootstrap_values[:n][timeout_mask]
        return next_values

    def compute_returns_and_advantages(
        self,
        gamma=0.99,
        lam=0.95,
        strategy: Advantage_Policy = Advantage_Policy.PPO_GAE,
    ):
        """Compute actor advantages and critic returns from one rollout.

        Notation used throughout this method:
        - ``discounted_returns[t]``: bootstrapped discounted return G_t.
        - ``td_errors[t]``: one-step Bellman residual delta_t.
        - ``raw_advantages[t]``: unnormalized actor advantage A_hat_t.
        - ``self.advantages[t]``: actor advantage after normalization.
        - ``self.returns[t]``: critic regression target R_t.

        For STANDARD_PPO, A_hat_t is GAE and R_t = A_hat_t + V_old(s_t).
        Legacy strategies keep using Monte Carlo G_t as the critic target.
        """
        n = self.ptr
        if n == 0:
            return

        next_values = self._next_state_values(n)
        episode_ends = np.maximum(
            self.terminated[:n], self.truncated[:n]
        )
        bootstrap_mask = 1.0 - self.terminated[:n]

        discounted_returns = np.zeros(n, dtype=np.float32)
        next_return = 0.0
        for i in reversed(range(n)):
            if episode_ends[i]:
                # A timeout is not a true terminal state, so bootstrap from
                # V_old(next_obs). A true termination has no future value.
                is_timeout = self.truncated[i] and not self.terminated[i]
                next_return = next_values[i] if is_timeout else 0.0

            next_return = self.rewards[i] + gamma * next_return
            discounted_returns[i] = next_return

        # Unless STANDARD_PPO overrides it below, the critic fits Monte Carlo G_t.
        self.returns[:n] = discounted_returns

        if strategy == Advantage_Policy.RETURN:
            # Legacy REINFORCE-style actor weight; it is not baseline-centered.
            raw_advantages = discounted_returns
        elif strategy == Advantage_Policy.ADVANTAGE:
            raw_advantages = discounted_returns - self.old_values[:n]
        elif strategy == Advantage_Policy.TD_ERROR:
            raw_advantages = (
                self.rewards[:n]
                + gamma * bootstrap_mask * next_values
                - self.old_values[:n]
            )
        elif strategy == Advantage_Policy.PPO_GAE:
            # TD errors are independent across transitions, so compute them in
            # one vectorized operation. GAE itself remains a backward scan
            # because A_hat_t depends on A_hat_{t+1}.
            td_errors = (
                self.rewards[:n]
                + gamma * bootstrap_mask * next_values
                - self.old_values[:n]
            )
            next_gae = 0.0
            raw_advantages = np.zeros(n, dtype=np.float32)
            for i in reversed(range(n)):
                next_gae = (
                    td_errors[i]
                    + gamma * lam * (1 - episode_ends[i]) * next_gae
                )
                raw_advantages[i] = next_gae
        elif strategy == Advantage_Policy.STANDARD_PPO:
            # Bootstrap through a time-limit truncation, but never let GAE flow
            # into the next reset episode. A true termination has no bootstrap.
            td_errors = (
                self.rewards[:n]
                + gamma * bootstrap_mask * next_values
                - self.old_values[:n]
            )
            next_gae = 0.0
            raw_advantages = np.zeros(n, dtype=np.float32)
            for i in reversed(range(n)):
                next_gae = (
                    td_errors[i]
                    + gamma * lam * (1.0 - episode_ends[i]) * next_gae
                )
                raw_advantages[i] = next_gae

            # GAE's matching critic target is the lambda-return R_t.
            self.returns[:n] = raw_advantages + self.old_values[:n]
        elif strategy == Advantage_Policy.ADVANTAGE_DISCOUNTED:
            raw_advantages = discounted_returns - self.old_values[:n]
            for i in reversed(range(n)):
                if i == n - 1:
                    continue
                raw_advantages[i] += (
                    gamma
                    * lam
                    * raw_advantages[i + 1]
                    * (1 - episode_ends[i])
                )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        self.raw_advantages[:n] = raw_advantages
        if self.normalizer is not None:
            advantage_tensor = torch.tensor(raw_advantages, dtype=torch.float32)
            actor_advantages = (
                self.normalizer.normalize(advantage_tensor).cpu().numpy()
            )
        else:
            actor_advantages = raw_advantages

        self.advantages[:n] = actor_advantages

    def get(self):
        return (
            torch.tensor(self.obs[:self.ptr], dtype=torch.float32),
            torch.tensor(self.actions[:self.ptr], dtype=torch.long),
            torch.tensor(self.old_log_probs[:self.ptr], dtype=torch.float32),
            torch.tensor(self.returns[:self.ptr], dtype=torch.float32),
            torch.tensor(self.advantages[:self.ptr], dtype=torch.float32),
            torch.tensor(self.old_values[:self.ptr], dtype=torch.float32),
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
    print(f'device: {device}')
    ac = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(ac.parameters(), lr=config.pi_lr)
    adv_normalizer = (
        AdvantageNormalizer(momentum=0)
        if config.use_adv_normalizer
        else None
    )
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

    def train_one_epoch(epoch: int) -> None:
        """Collect one rollout, update PPO, evaluate, and save the epoch."""
        buf.reset()
        steps_collected = 0
        while steps_collected < config.steps_per_epoch:
            obs, _ = env.reset()
            while True:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    action, old_log_prob, old_value = ac.get_action(obs_tensor)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                timeout_bootstrap_value = 0.0
                if truncated and not terminated:
                    # next_obs will not become the next stored observation
                    # because the environment is reset after a timeout.
                    with torch.no_grad():
                        next_obs_tensor = torch.tensor(
                            next_obs, dtype=torch.float32, device=device
                        )
                        timeout_value_tensor = ac.get_value(next_obs_tensor)
                        timeout_bootstrap_value = timeout_value_tensor.item()
                buf.store(
                    obs,
                    action,
                    old_log_prob.item(),
                    reward,
                    terminated,
                    truncated,
                    old_value.item(),
                    timeout_bootstrap_value=timeout_bootstrap_value,
                )
                obs = next_obs
                steps_collected += 1
                if terminated or truncated:
                    break
        # Rollout is complete: compute critic returns and actor advantages.
        buf.compute_returns_and_advantages(
            gamma=config.gamma,
            lam=config.lam,
            strategy=config.policy_target,
        )
        
        if run is not None:
            episode_ends = np.maximum(
                buf.terminated[:buf.ptr], buf.truncated[:buf.ptr]
            )
            first_done_indices = np.where(episode_ends > 0)[0]
            if first_done_indices.size > 0:
                first_end = int(first_done_indices[0])
            else:
                first_end = min(buf.ptr, 10) - 1
            first_end = max(first_end, -1)
            first_slice = slice(0, first_end + 1)
            first_len = first_slice.stop - first_slice.start

            episodes = int(np.count_nonzero(episode_ends))
            episodes = max(episodes, 1)
            rollout_log = {
                "rollout/epoch": epoch + 1,
                "rollout/steps": int(buf.ptr),
                "rollout/episodes": episodes,
                "rollout/avg_steps_per_episode": float(buf.ptr / episodes),
                "rollout/first_episode_len": int(first_len),
                "rollout/first_old_values": format_head_tail(
                    buf.old_values[first_slice].tolist()
                ),
                "rollout/first_returns": format_head_tail(
                    buf.returns[first_slice].tolist()
                ),
                "rollout/first_raw_advantages": format_head_tail(
                    buf.raw_advantages[first_slice].tolist()
                ),
                "rollout/first_advantages": format_head_tail(
                    buf.advantages[first_slice].tolist()
                ),
            }
            logger.info(f"\n{format_rollout_log_str(rollout_log)}")
            # run.log(
            #     rollout_log
            # )
            
        # Actor uses advantages; critic regresses toward returns (value targets).
        (
            obs_buf,
            actions_buf,
            old_log_probs_buf,
            returns_buf,
            advantages_buf,
            old_values_buf,
        ) = buf.get()
        obs_buf = obs_buf.to(device)
        actions_buf = actions_buf.to(device)
        old_log_probs_buf = old_log_probs_buf.to(device)
        returns_buf = returns_buf.to(device)
        advantages_buf = advantages_buf.to(device)
        old_values_buf = old_values_buf.to(device)
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
                new_log_probs, entropy, new_values = ac.evaluate_actions(
                    obs_buf[mb_idx], actions_buf[mb_idx]
                )
                advantages = advantages_buf[mb_idx].detach()
                if (
                    config.use_clip
                    or config.policy_target == Advantage_Policy.STANDARD_PPO
                ):
                    ratio = torch.exp(
                        new_log_probs - old_log_probs_buf[mb_idx]
                    )
                    surrogate = ratio * advantages
                    clipped_surrogate = torch.clamp(
                        ratio,
                        1 - config.clip_ratio,
                        1 + config.clip_ratio,
                    ) * advantages
                    policy_loss = -torch.minimum(
                        surrogate, clipped_surrogate
                    ).mean()
                    # logger.info(f'ratio: {ratio.mean().item()}')
                else:
                    policy_loss = -(new_log_probs * advantages).mean()
                if config.policy_target == Advantage_Policy.STANDARD_PPO:
                    old_values = old_values_buf[mb_idx]
                    values_clipped = old_values + torch.clamp(
                        new_values - old_values,
                        -config.value_clip_range,
                        config.value_clip_range,
                    )
                    value_loss_unclipped = (
                        new_values - returns_buf[mb_idx]
                    ) ** 2
                    value_loss_clipped = (
                        values_clipped - returns_buf[mb_idx]
                    ) ** 2
                    critic_loss = torch.maximum(
                        value_loss_unclipped, value_loss_clipped
                    ).mean()
                    entropy_coef = config.entropy_coef
                    value_loss_coef = config.value_loss_coef
                else:
                    critic_loss = (
                        (returns_buf[mb_idx] - new_values) ** 2
                    ).mean()
                    entropy_coef = max(0.02 * (1 - epoch / config.epochs), 0.0)
                    value_loss_coef = 0.5

                loss = (
                    policy_loss
                    + value_loss_coef * critic_loss
                    - entropy_coef * entropy.mean()
                )

                optimizer.zero_grad()
                loss.backward()
                if config.policy_target == Advantage_Policy.STANDARD_PPO:
                    torch.nn.utils.clip_grad_norm_(ac.parameters(), config.max_grad_norm)
                optimizer.step()
                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(critic_loss.item() * value_loss_coef)
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

    for epoch in range(start_epoch, config.epochs):
        train_one_epoch(epoch)

    if run is not None:
        run.finish()

# --- 5. 运行训练与推理 ---
if __name__ == "__main__":
    train_config = TrainingConfig(
                        epochs=100,
                        policy_target=Advantage_Policy.STANDARD_PPO,
                        use_wandb=True,
                        train_iters=10,
                        use_clip=True,
                        save_root="./logs_ac"
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
