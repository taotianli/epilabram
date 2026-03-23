"""
端到端小规模测试：用几个EDF文件跑完整的 Stage1 forward pass
验证数据加载 → 预处理 → 模型前向 → 损失计算 全链路
"""

import sys
import os

# 确保 epilabram 目录优先，避免误导入 LaBraM/utils.py
_HERE = os.path.dirname(os.path.abspath(__file__))
# 移除可能混入的 LaBraM 路径
sys.path = [p for p in sys.path if 'LaBraM' not in p]
sys.path.insert(0, _HERE)

import torch
import torch.nn.functional as F

from data.tuar_edf_dataset import TUAREDFDataset
from models.epilabram import build_epilabram
from training.masking import PathologyAwareDynamicMasking
from training.losses import MaskedEEGModelingLoss, HierarchicalConsistencyLoss
from utils.seed import set_seed


def test_data_loading(data_root: str, max_files: int = 3):
    print("\n" + "="*60)
    print("[1] 数据加载测试")
    print("="*60)
    ds = TUAREDFDataset(data_root, window_sec=10.0, stride_sec=10.0, max_files=max_files)
    print(f"  数据集大小: {len(ds)} 个片段")

    eeg, label = ds[0]
    print(f"  样本 shape: {eeg.shape}  (期望: [23, 2000])")
    print(f"  label: {label}  (0=normal, 1=abnormal)")
    print(f"  数值范围: [{eeg.min():.3f}, {eeg.max():.3f}]")
    assert eeg.shape == (23, 2000), f"shape错误: {eeg.shape}"
    assert label in (0, 1)
    print("  ✓ 数据加载通过")
    return ds


def test_masking(ds):
    print("\n" + "="*60)
    print("[2] PathologyAwareDynamicMasking 测试")
    print("="*60)
    masker = PathologyAwareDynamicMasking(sample_rate=200.0, base_mask_ratio=0.5)
    eeg, _ = ds[0]  # (23, 2000)
    mask, sym_mask = masker(eeg, patch_size=200)
    n_patches = 23 * (2000 // 200)
    print(f"  patch总数: {n_patches}  mask shape: {mask.shape}")
    print(f"  被掩码比例: {mask.float().mean():.3f}  (期望约0.5-0.8)")
    assert mask.shape == (n_patches,)
    assert (mask == ~sym_mask).all(), "对称掩码不互补"
    print("  ✓ 掩码生成通过")
    return masker


def test_model_forward(ds, masker, device):
    print("\n" + "="*60)
    print("[3] 模型前向传播测试")
    print("="*60)

    model = build_epilabram(backbone_size='base').to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  模型参数量: {n_params:.1f}M")

    # 构建 batch
    batch_size = 2
    eegs, labels = [], []
    for i in range(batch_size):
        eeg, label = ds[i % len(ds)]
        eegs.append(eeg)
        labels.append(label)

    eeg_batch = torch.stack(eegs).to(device)   # (B, 23, 2000)
    label_batch = torch.tensor(labels, device=device)

    B, C, T = eeg_batch.shape
    patch_size = 200
    A = T // patch_size
    eeg_4d = eeg_batch.reshape(B, C, A, patch_size)  # (B, 23, 10, 200)
    print(f"  输入 shape: {eeg_4d.shape}")

    # --- Stage1 forward ---
    print("\n  [Stage1] Masked EEG Modeling...")
    masks, sym_masks = [], []
    for b in range(B):
        m, sm = masker(eeg_batch[b], patch_size)
        masks.append(m)
        sym_masks.append(sm)
    mask = torch.stack(masks).to(device)       # (B, C*A)
    sym_mask = torch.stack(sym_masks).to(device)

    with torch.no_grad():
        targets = model.tokenizer.get_codebook_indices(eeg_4d)  # (B, C*A)
        logits, sym_logits = model.forward_stage1(eeg_4d, mask, sym_mask)

    print(f"  logits shape: {logits.shape}  (期望: [{B}, {C*A}, 8192])")
    assert logits.shape == (B, C * A, 8192)

    criterion = MaskedEEGModelingLoss()
    loss, log = criterion(logits, sym_logits, targets, mask, sym_mask)
    print(f"  Stage1 loss: {loss.item():.4f}")
    print(f"  mask accuracy: {log['metric/mask_acc'].item():.4f}")
    assert not torch.isnan(loss), "loss为NaN"
    print("  ✓ Stage1 forward 通过")

    # --- Stage2 forward ---
    print("\n  [Stage2] 多任务微调...")
    task_ids = torch.zeros(B, dtype=torch.long, device=device)  # TUAB task
    with torch.no_grad():
        results = model.forward_stage2(eeg_4d, task_ids)

    _, logits_s2 = results['TUAB']
    print(f"  Stage2 logits shape: {logits_s2.shape}  (期望: [{B}, 2])")
    assert logits_s2.shape == (B, 2)

    hier_loss_fn = HierarchicalConsistencyLoss()
    loss_s2, log_s2 = hier_loss_fn(logits_s2, None, None, label_batch)
    print(f"  Stage2 loss: {loss_s2.item():.4f}")
    assert not torch.isnan(loss_s2)
    print("  ✓ Stage2 forward 通过")

    return model


def test_backward(ds, masker, model, device):
    print("\n" + "="*60)
    print("[4] 反向传播测试")
    print("="*60)

    eeg, label = ds[0]
    eeg_batch = eeg.unsqueeze(0).to(device)   # (1, 23, 2000)
    B, C, T = eeg_batch.shape
    A = T // 200
    eeg_4d = eeg_batch.reshape(B, C, A, 200)

    mask, sym_mask = masker(eeg_batch[0], 200)
    mask = mask.unsqueeze(0).to(device)
    sym_mask = sym_mask.unsqueeze(0).to(device)

    with torch.no_grad():
        targets = model.tokenizer.get_codebook_indices(eeg_4d)

    model.train()
    logits, sym_logits = model.forward_stage1(eeg_4d, mask, sym_mask)
    criterion = MaskedEEGModelingLoss()
    loss, _ = criterion(logits, sym_logits, targets, mask, sym_mask)
    loss.backward()

    # 检查梯度
    grad_norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.norm().item())
    print(f"  有梯度的参数组数: {len(grad_norms)}")
    print(f"  梯度范数 mean={sum(grad_norms)/len(grad_norms):.4f}  max={max(grad_norms):.4f}")
    assert len(grad_norms) > 0, "没有梯度"
    assert not any(torch.isnan(torch.tensor(g)) for g in grad_norms), "梯度含NaN"
    print("  ✓ 反向传播通过")


def test_memory(device):
    print("\n" + "="*60)
    print("[5] 显存占用估算")
    print("="*60)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        model = build_epilabram(backbone_size='base').to(device)
        masker = PathologyAwareDynamicMasking()
        criterion = MaskedEEGModelingLoss()

        # batch_size=8 模拟训练
        B, C, A, T = 8, 23, 10, 200
        eeg = torch.randn(B, C, A, T, device=device)
        mask = torch.rand(B, C * A, device=device) > 0.5
        sym_mask = ~mask

        with torch.no_grad():
            targets = model.tokenizer.get_codebook_indices(eeg)
        logits, sym_logits = model.forward_stage1(eeg, mask, sym_mask)
        loss, _ = criterion(logits, sym_logits, targets, mask, sym_mask)
        loss.backward()

        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"  batch_size=8, 峰值显存: {peak_mb:.0f} MB ({peak_mb/1024:.1f} GB)")
        print(f"  推荐显卡: {'RTX 3090/4090 (24GB)' if peak_mb < 20000 else 'A100 (40/80GB)'}")
    else:
        print("  (CPU模式，跳过显存测试)")


def main():
    set_seed(42)
    data_root = '/home/taotl/Desktop/TUAR/v3.0.1/edf'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    ds = test_data_loading(data_root, max_files=5)
    masker = test_masking(ds)
    model = test_model_forward(ds, masker, device)
    test_backward(ds, masker, model, device)
    test_memory(device)

    print("\n" + "="*60)
    print("✓ 所有测试通过！")
    print("="*60)


if __name__ == '__main__':
    main()
