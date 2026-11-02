from typing import Optional
import os
import torch
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    decay_rate: float = 0.1,        
    decay_epoch: int = 50,          
    warmup_epochs: int = 0,
    warmup_start_factor: float = 0.1,
    num_epochs: Optional[int] = None,   
    eta_min: float = 1e-6,              
    use_warm_restarts: bool = False,    
    T_mult: int = 1,                    
):
    if use_warm_restarts:
        T_0 = decay_epoch if (num_epochs is None) else max(1, num_epochs - warmup_epochs)
        main_sched = CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )  

        if warmup_epochs > 0:
            warmup = LinearLR(optimizer, start_factor=warmup_start_factor, total_iters=warmup_epochs)
            scheduler = SequentialLR(optimizer, [warmup, main_sched], milestones=[warmup_epochs])
        else:
            scheduler = main_sched
        return scheduler

    
    T_max = (decay_epoch if num_epochs is None else max(1, num_epochs - warmup_epochs))
    cosine = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)  

    if warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=warmup_start_factor, total_iters=warmup_epochs)  
        scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = cosine
    return scheduler


def step_scheduler(scheduler) -> None:
    scheduler.step()  


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def save_train_state(path: str, model: torch.nn.Module,
                     optimizer: torch.optim.Optimizer,
                     scheduler, epoch: int) -> None:
    pkg = {
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(pkg, path)

def load_train_state(path: str, model: torch.nn.Module,
                     optimizer: Optional[torch.optim.Optimizer] = None,
                     scheduler=None,
                     map_location="cpu") -> int:
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        (model.module if hasattr(model, "module") else model).load_state_dict(state, strict=False)
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        return int(ckpt.get("epoch", 0)) + 1
    else:
        state = ckpt
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        (model.module if hasattr(model, "module") else model).load_state_dict(state, strict=False)
        return 1