import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import logging
try:
    import torch.distributed as dist
except Exception:
    dist = None
    
def next_log_dir(base_dir: str, prefix: str = "log", create: bool = False) -> str:
    """
    在 base_dir 下找到下一个自增的日志目录，例如 log1、log2、log3...
    返回新目录的绝对路径。
    
    - base_dir: 日志根目录
    - prefix:   目录前缀（默认 "log"）
    - create:   是否实际创建目录（默认 True）
    """
    base = Path(base_dir).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)

    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    max_n = 0
    for p in base.iterdir():
        if p.is_dir():
            m = pat.match(p.name)
            if m:
                max_n = max(max_n, int(m.group(1)))

    n = max_n + 1
    new_dir = base / f"{prefix}{n}"
    if create:
        new_dir.mkdir(parents=True, exist_ok=False)
    return str(new_dir)




def create_timestamped_subfolder(base_dir: str, prefix: str, create: bool = False) -> str:
    """
    在 base_dir 下创建一个以 prefix + 当前日期时间 为名字的子文件夹

    Args:
        base_dir (str): 基础文件夹路径
        prefix (str): 子文件夹前缀

    Returns:
        str: 创建好的子文件夹路径
    """
    # 格式化当前日期时间
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    folder_name = f"{prefix}_{timestamp}"
    folder_path = os.path.join(base_dir, folder_name)

    # 创建文件夹
    if create:
        os.makedirs(folder_path, exist_ok=True)
    return folder_path


def creat_subdir(base_dir: str, prefix: str = "log", create: bool = False, time: bool = False) -> str:
    """Create either an incrementing or timestamped log subdirectory."""
    if time:
        return create_timestamped_subfolder(base_dir=base_dir, prefix=prefix, create=create)
    return next_log_dir(base_dir=base_dir, prefix=prefix, create=create)


def format_rollout_log_str(rollout_log: dict) -> str:
    lines = []

    sep = "=" * 52
    lines.append(sep)
    lines.append("Rollout Summary".center(52))
    lines.append(sep)

    # -------- 标量项 --------
    scalar_keys = [
        "rollout/epoch",
        "rollout/steps",
        "rollout/episodes",
        "rollout/avg_steps_per_episode",
        "rollout/first_episode_len",
    ]

    for k in scalar_keys:
        if k not in rollout_log:
            continue
        name = k.replace("rollout/", "")
        v = rollout_log[k]
        if isinstance(v, float):
            lines.append(f"{name:<28}: {v:.2f}")
        else:
            lines.append(f"{name:<28}: {v}")

    # -------- 序列项 --------
    seq_keys = [
        "rollout/first_values",
        "rollout/first_returns",
        "rollout/first_advantages",
    ]

    for k in seq_keys:
        if k not in rollout_log:
            continue
        name = k.replace("rollout/", "")
        v = rollout_log[k]
        lines.append("")  # 空行
        lines.append(f"{name}:")
        lines.append(f"  head: {v.get('head', [])}")
        lines.append(f"  tail: {v.get('tail', [])}")

    lines.append(sep)
    return "\n".join(lines)


def format_head_tail(values: List[float], n: int = 5):
    """Return head/tail slices for inspection."""
    if not values:
        return {"head": [], "tail": []}
    head = [f"{float(v):.3f}" for v in values[:n]]
    tail = [f"{float(v):.3f}" for v in values[-n:]] if len(values) > n else []
    return {"head": head, "tail": tail}




base_log_dir = Path(__file__).resolve().parents[1] / "logs"


def setup_logger(
    name: str = "rl",
    log_dir: str = None,
    filename: str = "training.log",
    level: int = logging.INFO,
    formatter: Optional[logging.Formatter] = None,
) -> logging.Logger:
    """
    Initialize a rank-aware logger with console + file handlers.

    Console logs only from rank 0 to avoid clutter; each rank writes to its own log file.
    """
    if log_dir is None:
        log_dir = creat_subdir(base_log_dir, prefix=name, create=True, time=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        if not hasattr(logger, "log_dir") and log_dir is not None:
            logger.log_dir = log_dir
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    rank = get_rank()

    os.makedirs(log_dir, exist_ok=True)
    prefix, ext = os.path.splitext(filename)
    log_path = os.path.join(log_dir, f"{prefix}_rank{rank}{ext or '.log'}")

    fmt = formatter or logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(fmt)
        logger.addHandler(console_handler)

    logger.debug("Logger initialized", extra={"rank": rank, "log_path": log_path})
    logger.log_dir = log_dir
    return logger


def get_rank() -> int:
    """Best-effort rank detection for multi-process (DDP/multi-node) setups."""
    if dist is not None:
        try:
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank()
            else:
                return 0
        except Exception:
            pass

    for env_key in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        if env_key in os.environ:
            try:
                return int(os.environ[env_key])
            except ValueError:
                break

    return 0