import random
from datetime import datetime
from functools import partial

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
from inference import evaluate_policy, record_policy_videos
from config import TrainingConfig, Advantage_Policy
from utils import (
    format_head_tail,
    format_rollout_log_str,
    path_component,
    resolve_output_root,
    setup_logger,
    training_run_dir,
)

logger = logging.getLogger("rl.training")



def select_device(pref: str) -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def make_env(env_name: str):
    """Top-level factory so subprocess vector workers can pickle it."""
    return gym.make(env_name)


def compute_policy_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    use_probability_ratio: bool,
    use_clip: bool,
    clip_ratio: float,
) -> torch.Tensor:
    """Compute the actor objective used by the selected training strategy.

    Standard PPO always uses the importance-sampling ratio. ``use_clip`` only
    decides whether that ratio is clipped; disabling clipping therefore gives
    the unclipped surrogate ``-mean(ratio * advantage)``.

    Legacy, non-PPO strategies retain their historical log-probability loss
    when neither ratio weighting nor clipping is requested.
    """
    if use_probability_ratio or use_clip:
        ratio = torch.exp(new_log_probs - old_log_probs)
        surrogate = ratio * advantages
        if use_clip:
            clipped_surrogate = torch.clamp(
                ratio,
                1.0 - clip_ratio,
                1.0 + clip_ratio,
            ) * advantages
            surrogate = torch.minimum(surrogate, clipped_surrogate)
        return -surrogate.mean()

    return -(new_log_probs * advantages).mean()

# --- 2. Rollout Buffer ---
class RolloutBuffer:
    def __init__(
        self,
        size,
        obs_dim,
        normalizer: AdvantageNormalizer | None = None,
        num_envs: int = 1,
    ):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.actions = np.zeros(size, np.int32)
        self.old_log_probs = np.zeros(size, np.float32)
        self.rewards = np.zeros(size, np.float32)
        self.terminated = np.zeros(size, np.float32)
        self.truncated = np.zeros(size, np.float32)
        self.old_values = np.zeros(size, np.float32)
        # Only needed when an episode is cut off by the environment time limit.
        # Normal next values come from the next vector time step.
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
        self.num_envs = num_envs

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

    def store_batch(
        self,
        obs,
        actions,
        old_log_probs,
        rewards,
        terminated,
        truncated,
        old_values,
        timeout_bootstrap_values,
    ) -> None:
        """Store one time step from all vector environments in env order."""
        batch_size = len(rewards)
        if batch_size != self.num_envs:
            raise ValueError(
                f"expected {self.num_envs} transitions, got {batch_size}"
            )
        end = self.ptr + batch_size
        if end > self.max_size:
            raise RuntimeError("rollout buffer capacity exceeded")
        target = slice(self.ptr, end)
        self.obs[target] = obs
        self.actions[target] = actions
        self.old_log_probs[target] = old_log_probs
        self.rewards[target] = rewards
        self.terminated[target] = terminated
        self.truncated[target] = truncated
        self.old_values[target] = old_values
        self.timeout_bootstrap_values[target] = timeout_bootstrap_values
        self.ptr = end

    def _next_state_values(
        self,
        n: int,
        last_values: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build V_old(s_{t+1}) independently for each environment."""
        if n % self.num_envs != 0:
            raise ValueError("stored transitions must form complete vector steps")
        time_steps = n // self.num_envs
        values = self.old_values[:n].reshape(time_steps, self.num_envs)
        next_values = np.zeros_like(values)
        if time_steps > 1:
            next_values[:-1] = values[1:]
        if last_values is not None:
            last_values = np.asarray(last_values, dtype=np.float32)
            if last_values.shape != (self.num_envs,):
                raise ValueError(
                    f"last_values must have shape ({self.num_envs},)"
                )
            next_values[-1] = last_values

        # At an episode end, the following stored value belongs to the reset
        # episode. A time-limit truncation instead bootstraps from final_obs.
        timeout_mask = (
            self.truncated[:n].astype(bool)
            & ~self.terminated[:n].astype(bool)
        ).reshape(time_steps, self.num_envs)
        timeout_values = self.timeout_bootstrap_values[:n].reshape(
            time_steps, self.num_envs
        )
        next_values[timeout_mask] = timeout_values[timeout_mask]
        return next_values.reshape(-1)

    def compute_returns_and_advantages(
        self,
        gamma=0.99,
        lam=0.95,
        strategy: Advantage_Policy = Advantage_Policy.PPO_GAE,
        last_values: np.ndarray | None = None,
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

        if n % self.num_envs != 0:
            raise ValueError("stored transitions must form complete vector steps")
        time_steps = n // self.num_envs
        shape = (time_steps, self.num_envs)
        next_values = self._next_state_values(n, last_values).reshape(shape)
        rewards = self.rewards[:n].reshape(shape)
        old_values = self.old_values[:n].reshape(shape)
        terminated = self.terminated[:n].reshape(shape)
        truncated = self.truncated[:n].reshape(shape)
        episode_ends = np.maximum(terminated, truncated)
        bootstrap_mask = 1.0 - terminated

        if last_values is None:
            next_return = np.zeros(self.num_envs, dtype=np.float32)
        else:
            next_return = np.asarray(last_values, dtype=np.float32).copy()
        discounted_returns = np.zeros(shape, dtype=np.float32)
        for i in reversed(range(time_steps)):
            if np.any(episode_ends[i]):
                # A timeout is not a true terminal state, so bootstrap from
                # V_old(next_obs). A true termination has no future value.
                is_timeout = (truncated[i] > 0) & (terminated[i] == 0)
                episode_bootstrap = np.where(
                    is_timeout, next_values[i], 0.0
                )
                next_return = np.where(
                    episode_ends[i] > 0, episode_bootstrap, next_return
                )

            next_return = rewards[i] + gamma * next_return
            discounted_returns[i] = next_return

        # Unless STANDARD_PPO overrides it below, the critic fits Monte Carlo G_t.
        self.returns[:n] = discounted_returns.reshape(-1)

        if strategy == Advantage_Policy.RETURN:
            # Legacy REINFORCE-style actor weight; it is not baseline-centered.
            raw_advantages = discounted_returns
        elif strategy == Advantage_Policy.ADVANTAGE:
            raw_advantages = discounted_returns - old_values
        elif strategy == Advantage_Policy.TD_ERROR:
            raw_advantages = (
                rewards
                + gamma * bootstrap_mask * next_values
                - old_values
            )
        elif strategy == Advantage_Policy.PPO_GAE:
            # TD errors are independent across transitions, so compute them in
            # one vectorized operation. GAE itself remains a backward scan
            # because A_hat_t depends on A_hat_{t+1}.
            td_errors = (
                rewards
                + gamma * bootstrap_mask * next_values
                - old_values
            )
            next_gae = np.zeros(self.num_envs, dtype=np.float32)
            raw_advantages = np.zeros(shape, dtype=np.float32)
            for i in reversed(range(time_steps)):
                next_gae = (
                    td_errors[i]
                    + gamma * lam * (1 - episode_ends[i]) * next_gae
                )
                raw_advantages[i] = next_gae
        elif strategy == Advantage_Policy.STANDARD_PPO:
            # Bootstrap through a time-limit truncation, but never let GAE flow
            # into the next reset episode. A true termination has no bootstrap.
            td_errors = (
                rewards
                + gamma * bootstrap_mask * next_values
                - old_values
            )
            next_gae = np.zeros(self.num_envs, dtype=np.float32)
            raw_advantages = np.zeros(shape, dtype=np.float32)
            for i in reversed(range(time_steps)):
                next_gae = (
                    td_errors[i]
                    + gamma * lam * (1.0 - episode_ends[i]) * next_gae
                )
                raw_advantages[i] = next_gae

            # GAE's matching critic target is the lambda-return R_t.
            self.returns[:n] = (raw_advantages + old_values).reshape(-1)
        elif strategy == Advantage_Policy.ADVANTAGE_DISCOUNTED:
            raw_advantages = discounted_returns - old_values
            for i in reversed(range(time_steps)):
                if i == time_steps - 1:
                    continue
                raw_advantages[i] += (
                    gamma
                    * lam
                    * raw_advantages[i + 1]
                    * (1 - episode_ends[i])
                )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        raw_advantages = raw_advantages.reshape(-1)
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
    global logger
    if config.num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if config.steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")
    if config.steps_per_epoch % config.num_envs != 0:
        raise ValueError(
            "steps_per_epoch must be divisible by num_envs; "
            f"got {config.steps_per_epoch} and {config.num_envs}"
        )
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    valid_policy_targets = set(Advantage_Policy)
    if config.policy_target not in valid_policy_targets:
        raise ValueError(f"policy_target must be one of {list(Advantage_Policy)}")
    policy_name = (
        config.policy_target.name
        if isinstance(config.policy_target, Advantage_Policy)
        else str(config.policy_target)
    )
    if config.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        clip_label = "clip" if config.use_clip else "no-clip"
        config.run_name = (
            f"{timestamp}__{policy_name.lower()}__{clip_label}"
            f"__seed-{config.seed}"
        )
    base_name = config.run_name

    if config.save_root is None:
        save_root = training_run_dir(
            output_root=config.output_root,
            env_name=config.env_name,
            experiment_name=config.experiment_name,
            run_name=base_name,
        )
    else:
        save_root = Path(config.save_root)
        if not save_root.is_absolute():
            save_root = Path(__file__).resolve().parent / save_root
        output_root = resolve_output_root(config.output_root).resolve()
        if not save_root.resolve().is_relative_to(output_root):
            raise ValueError(
                "save_root must be inside output_root; set output_root to "
                "the desired common output directory"
            )
    save_root.mkdir(parents=True, exist_ok=True)
    logger_dir = getattr(logger, "log_dir", None)
    if logger_dir is None or Path(logger_dir).resolve() != save_root.resolve():
        logger = setup_logger(
            name=f"rl.training.{path_component(base_name)}",
            log_dir=str(save_root),
            filename="training.log",
        )
    logger.info(config)
    checkpoint_dir = Path(config.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = save_root / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    video_subdir = Path(config.video_dir)
    if (
        video_subdir == Path(".")
        or video_subdir.is_absolute()
        or ".." in video_subdir.parts
    ):
        raise ValueError(
            "video_dir must be a relative subdirectory inside save_root"
        )
    video_dir = save_root / video_subdir
    resume_path = Path(config.resume_path) if config.resume_path else None
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    env_fns = [partial(make_env, config.env_name) for _ in range(config.num_envs)]
    vector_env_class = (
        gym.vector.SyncVectorEnv
        if config.num_envs == 1
        else gym.vector.AsyncVectorEnv
    )
    rollout_env = vector_env_class(
        env_fns,
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    rollout_obs, _ = rollout_env.reset(seed=config.seed)
    rollout_env.action_space.seed(config.seed)

    eval_env = gym.make(config.env_name)
    eval_env.reset(seed=config.seed + 50_000)
    eval_env.action_space.seed(config.seed + 50_000)
    obs_dim = rollout_env.single_observation_space.shape[0]
    act_dim = rollout_env.single_action_space.n
    device = select_device(config.device)
    print(f'device: {device}')
    ac = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(ac.parameters(), lr=config.pi_lr)
    adv_normalizer = (
        AdvantageNormalizer(momentum=0)
        if config.use_adv_normalizer
        else None
    )
    training_config = asdict(config)
    training_config["policy_target"] = policy_name
    training_config["output_root"] = str(config.output_root)
    training_config["save_root"] = str(save_root)
    training_config["checkpoint_dir"] = str(checkpoint_dir)
    training_config["video_dir"] = str(video_dir)
    training_config["resume_path"] = str(resume_path) if resume_path else None

    rollout_horizon = config.steps_per_epoch // config.num_envs
    buf = RolloutBuffer(
        config.steps_per_epoch,
        obs_dim,
        normalizer=adv_normalizer,
        num_envs=config.num_envs,
    )

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
    training_config["run_name"] = base_name

    run = init_wandb(
        config.use_wandb,
        save_root=save_root,
        config=training_config,
        run_name=base_name,
        resume_id=(resume_path.stem + "_train") if resume_path else None,
        entity=config.wandb_entity,
        project=config.wandb_project,
        group=config.wandb_group,
    )
    logger.info(f"[wandb] train run   : {base_name}")
    if run is not None:
        logger.info(f"[wandb] run url     : {run.url}")

    def train_one_epoch(epoch: int) -> None:
        """Collect one rollout, update PPO, evaluate, and save the epoch."""
        nonlocal rollout_obs
        buf.reset()
        for _ in range(rollout_horizon):
            obs_tensor = torch.as_tensor(
                rollout_obs, dtype=torch.float32, device=device
            )
            with torch.no_grad():
                actions, old_log_probs, old_values = ac.get_actions(obs_tensor)
            next_obs, rewards, terminated, truncated, infos = rollout_env.step(
                actions.cpu().numpy()
            )

            timeout_bootstrap_values = np.zeros(
                config.num_envs, dtype=np.float32
            )
            timeout_mask = np.asarray(truncated, dtype=bool) & ~np.asarray(
                terminated, dtype=bool
            )
            if np.any(timeout_mask):
                final_obs = infos.get("final_obs")
                if final_obs is None:
                    raise RuntimeError(
                        "vector environment did not provide final_obs for a timeout"
                    )
                timeout_obs_tensor = torch.as_tensor(
                    np.stack(final_obs[timeout_mask]),
                    dtype=torch.float32,
                    device=device,
                )
                with torch.no_grad():
                    timeout_values = ac.get_value(timeout_obs_tensor)
                timeout_bootstrap_values[timeout_mask] = (
                    timeout_values.cpu().numpy()
                )

            buf.store_batch(
                rollout_obs,
                actions.cpu().numpy(),
                old_log_probs.cpu().numpy(),
                rewards,
                terminated,
                truncated,
                old_values.cpu().numpy(),
                timeout_bootstrap_values,
            )
            rollout_obs = next_obs

        # Only the final vector step needs an extra critic call. For all prior
        # normal transitions, V(s_{t+1}) is the next stored old value.
        with torch.no_grad():
            last_values = ac.get_value(
                torch.as_tensor(
                    rollout_obs, dtype=torch.float32, device=device
                )
            ).cpu().numpy()
        # Rollout is complete: compute critic returns and actor advantages.
        buf.compute_returns_and_advantages(
            gamma=config.gamma,
            lam=config.lam,
            strategy=config.policy_target,
            last_values=last_values,
        )
        
        if run is not None:
            episode_ends = np.maximum(
                buf.terminated[:buf.ptr], buf.truncated[:buf.ptr]
            )
            env_zero_indices = np.arange(0, buf.ptr, config.num_envs)
            env_zero_ends = episode_ends[env_zero_indices]
            first_done_positions = np.where(env_zero_ends > 0)[0]
            if first_done_positions.size > 0:
                first_end = int(first_done_positions[0])
            else:
                first_end = min(rollout_horizon, 10) - 1
            first_end = max(first_end, -1)
            first_indices = env_zero_indices[: first_end + 1]
            first_len = len(first_indices)

            episodes = int(np.count_nonzero(episode_ends))
            episodes = max(episodes, 1)
            rollout_log = {
                "rollout/epoch": epoch + 1,
                "rollout/steps": int(buf.ptr),
                "rollout/num_envs": config.num_envs,
                "rollout/steps_per_env": rollout_horizon,
                "rollout/episodes": episodes,
                "rollout/avg_steps_per_episode": float(buf.ptr / episodes),
                "rollout/first_episode_len": int(first_len),
                "rollout/first_old_values": format_head_tail(
                    buf.old_values[first_indices].tolist()
                ),
                "rollout/first_returns": format_head_tail(
                    buf.returns[first_indices].tolist()
                ),
                "rollout/first_raw_advantages": format_head_tail(
                    buf.raw_advantages[first_indices].tolist()
                ),
                "rollout/first_advantages": format_head_tail(
                    buf.advantages[first_indices].tolist()
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
        value_visit_count = 0
        value_clipped_visits = torch.zeros((), dtype=torch.long, device=device)
        value_gradient_blocked_visits = torch.zeros(
            (), dtype=torch.long, device=device
        )

        with torch.no_grad():
            target_variance = returns_buf.var(unbiased=False)
            residual_variance = (returns_buf - old_values_buf).var(unbiased=False)
            explained_variance = (
                1.0 - residual_variance / target_variance
                if target_variance > 1e-8
                else torch.tensor(float("nan"), device=device)
            )
            rollout_value_mse = ((old_values_buf - returns_buf) ** 2).mean()
            rollout_value_bias = (old_values_buf - returns_buf).mean()
            rollout_target_mean = returns_buf.mean()
            rollout_old_value_mean = old_values_buf.mean()

        for _ in range(config.train_iters):
            np.random.shuffle(idx)
            for start in range(0, n, config.batch_size):
                end = start + config.batch_size
                mb_idx = idx[start:end]
                new_log_probs, entropy, new_values = ac.evaluate_actions(
                    obs_buf[mb_idx], actions_buf[mb_idx]
                )
                advantages = advantages_buf[mb_idx].detach()
                policy_loss = compute_policy_loss(
                    new_log_probs,
                    old_log_probs_buf[mb_idx],
                    advantages,
                    use_probability_ratio=(
                        config.policy_target == Advantage_Policy.STANDARD_PPO
                    ),
                    use_clip=config.use_clip,
                    clip_ratio=config.clip_ratio,
                )
                if config.policy_target == Advantage_Policy.STANDARD_PPO:
                    old_values = old_values_buf[mb_idx]
                    value_loss_unclipped = (
                        new_values - returns_buf[mb_idx]
                    ) ** 2
                    if config.use_value_clip:
                        value_change = new_values - old_values
                        values_clipped = old_values + torch.clamp(
                            value_change,
                            -config.value_clip_range,
                            config.value_clip_range,
                        )
                        value_loss_clipped = (
                            values_clipped - returns_buf[mb_idx]
                        ) ** 2
                        selected_value_losses = torch.maximum(
                            value_loss_unclipped, value_loss_clipped
                        )
                        value_clipped = (
                            value_change.abs() > config.value_clip_range
                        )
                        gradient_blocked = value_clipped & (
                            value_loss_clipped > value_loss_unclipped
                        )
                        with torch.no_grad():
                            value_clipped_visits += value_clipped.sum()
                            value_gradient_blocked_visits += (
                                gradient_blocked.sum()
                            )
                    else:
                        selected_value_losses = value_loss_unclipped
                    value_visit_count += new_values.numel()
                    critic_loss = selected_value_losses.mean()
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
        test_reward = evaluate_policy(
            ac,
            eval_env,
            render=config.render_test,
            device=device,
        )
        mean_policy_loss = float(np.mean(epoch_policy_losses)) if epoch_policy_losses else 0.0
        mean_value_loss = float(np.mean(epoch_value_losses)) if epoch_value_losses else 0.0
        mean_entropy = float(np.mean(epoch_entropies)) if epoch_entropies else 0.0
        current_policy_loss = epoch_policy_losses[-1] if epoch_policy_losses else 0.0
        current_value_loss = epoch_value_losses[-1] if epoch_value_losses else 0.0
        current_entropy = epoch_entropies[-1] if epoch_entropies else 0.0
        value_clip_visit_rate = (
            value_clipped_visits.item() / value_visit_count
            if value_visit_count
            else 0.0
        )
        value_gradient_blocked_rate = (
            value_gradient_blocked_visits.item() / value_visit_count
            if value_visit_count
            else 0.0
        )
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
                "value_loss_mean": mean_value_loss,
                "entropy": current_entropy,
                "value/clip_visit_rate": value_clip_visit_rate,
                "value/gradient_blocked_rate": value_gradient_blocked_rate,
                "value/rollout_mse": rollout_value_mse.item(),
                "value/rollout_bias": rollout_value_bias.item(),
                "value/explained_variance": explained_variance.item(),
                "value/target_mean": rollout_target_mean.item(),
                "value/old_value_mean": rollout_old_value_mean.item(),
            })

    try:
        for epoch in range(start_epoch, config.epochs):
            train_one_epoch(epoch)

        if config.video_episodes > 0:
            logger.info(
                "Recording %d final-policy episodes in %s",
                config.video_episodes,
                video_dir,
            )
            video_records = record_policy_videos(
                ac=ac,
                env_name=config.env_name,
                episodes=config.video_episodes,
                video_dir=video_dir,
                name_prefix="final-policy",
                device=device,
                base_seed=config.seed + 100_000,
            )
            for record in video_records:
                logger.info(
                    "Video episode %d: seed=%d reward=%.2f steps=%d path=%s",
                    record["episode"],
                    record["seed"],
                    record["reward"],
                    record["steps"],
                    record["video"],
                )
            average_video_reward = float(
                np.mean([record["reward"] for record in video_records])
            )
            logger.info(
                "Final-policy video average reward: %.2f",
                average_video_reward,
            )
            if run is not None:
                run.summary["video/episodes"] = len(video_records)
                run.summary["video/average_reward"] = average_video_reward
    finally:
        rollout_env.close()
        eval_env.close()
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
        experiment_name="manual",
    )
    ppo_train(train_config)
    # 
    """
    我发现训练后期模型会忘记坏状态下的策略， 
    因为考虑如果（极限）优化后期大部分状态都是好的，所以优化到后面，模型会将坏的结果可能也打一个很高的 value，
    那么此时模型输入一个坏结果，就会干扰到整个模型的训练，让极其不稳定。 
    这是极限情况，可以说明模型训练不鲁棒，反之如果是最优点的话，不会出现扰动，
    
    后续可以针对此，可以记录下一定的坏状态，增加到value的训练中
    """
