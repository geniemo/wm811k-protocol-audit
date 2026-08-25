"""Fixed optimisation budget: same number of gradient steps for every cell."""
import random
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .preprocess import one_hot_maps


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 8000
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    log_every: int = 1000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _batches(n: int, batch_size: int, generator: torch.Generator):
    """Infinite stream of index batches; reshuffles when an epoch is exhausted, drops the remainder."""
    while True:
        perm = torch.randperm(n, generator=generator)
        for s in range(0, n - batch_size + 1, batch_size):
            yield perm[s:s + batch_size]


def train_fixed(model, maps_u8: torch.Tensor, labels: torch.Tensor, train_idx: np.ndarray, cfg: TrainConfig,
                seed: int, device, eval_fn: Optional[Callable] = None) -> list[dict]:
    set_seed(seed)
    model.to(device).train()
    idx_t = torch.as_tensor(np.asarray(train_idx), dtype=torch.long, device=device)
    n = len(idx_t)
    bs = min(cfg.batch_size, n)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)
    stream = _batches(n, bs, torch.Generator().manual_seed(seed))
    log, running, count, t0 = [], 0.0, 0, time.time()
    for step in range(1, cfg.steps + 1):
        b = idx_t[next(stream).to(device)]
        x = one_hot_maps(maps_u8[b.to(maps_u8.device)].to(device))
        y = labels[b.to(labels.device)].to(device)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        running += float(loss.detach())
        count += 1
        if step % cfg.log_every == 0 or step == cfg.steps:
            row = dict(step=step, loss=running / count, lr=float(sched.get_last_lr()[0]), seconds=time.time() - t0)
            if eval_fn is not None:
                model.eval()
                row.update(eval_fn(model))
                model.train()
            log.append(row)
            running, count = 0.0, 0
    model.eval()
    return log


@torch.no_grad()
def predict(model, maps_u8: torch.Tensor, idx: np.ndarray, device, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    idx_t = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=maps_u8.device)
    out = []
    for s in range(0, len(idx_t), batch_size):
        x = one_hot_maps(maps_u8[idx_t[s:s + batch_size]].to(device))
        out.append(model(x).argmax(dim=1).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0, dtype=np.int64)
