"""
模型保存/加载工具
"""

import os
import torch
from typing import Optional


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
):
    """
    保存模型 checkpoint。

    Args:
        path: 保存路径
        model: 模型
        optimizer: 优化器（可选）
        scaler: AMP scaler（可选）
        epoch: 当前 epoch
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
    }
    if optimizer is not None:
        state['optimizer'] = optimizer.state_dict()
    if scaler is not None:
        state['scaler'] = scaler.state_dict()
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    strict: bool = True,
) -> int:
    """
    加载 checkpoint，返回 epoch。

    Args:
        path: checkpoint 路径
        model: 模型
        optimizer: 优化器（可选）
        scaler: AMP scaler（可选）
        strict: 是否严格匹配 state_dict

    Returns:
        epoch: 恢复的 epoch
    """
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=strict)
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scaler is not None and 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
    return ckpt.get('epoch', 0) + 1
