"""
诊断脚本：逐步测试全量数据集 + trainer 的各个阶段
"""
import sys, os
sys.path = [p for p in sys.path if 'LaBraM' not in p]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import copy
from data.tuar_edf_dataset import TUAREDFDataset
from models.epilabram import build_epilabram
from utils.seed import set_seed

set_seed(42)
device = torch.device('cuda')

print('[1] loading full dataset...')
ds = TUAREDFDataset('/home/taotl/Desktop/TUAR/v3.0.1/edf', window_sec=10.0, stride_sec=5.0)
print(f'    ok: {len(ds)} segments')

print('[2] building model...')
model = build_epilabram(
    'base',
    pretrained_path='/home/taotl/Desktop/LaBraM/checkpoints/labram-base.pth'
).to(device)
print(f'    ok: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params')

print('[3] deepcopy for EMA...')
ema = copy.deepcopy(model)
print('    ok')

print('[4] WeightedRandomSampler with full dataset...')
from torch.utils.data import WeightedRandomSampler, ConcatDataset, DataLoader
weights = torch.ones(len(ds), dtype=torch.float)
sampler = WeightedRandomSampler(weights=weights, num_samples=len(ds), replacement=True)
loader = DataLoader(ds, batch_size=128, sampler=sampler, num_workers=0,
                    pin_memory=True, drop_last=True)
print(f'    ok: {len(loader)} batches/epoch')

print('[5] first batch forward...')
from training.masking import PathologyAwareDynamicMasking
from training.losses import MaskedEEGModelingLoss
from torch.amp import GradScaler, autocast
import torch.nn as nn

masker = PathologyAwareDynamicMasking()
criterion = MaskedEEGModelingLoss()
scaler = GradScaler('cuda')
opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
model.train()

batch = next(iter(loader))
eeg = batch[0].to(device)
B, C, T = eeg.shape; A = T // 200
eeg4d = eeg.reshape(B, C, A, 200)
masks, syms = [], []
for b in range(B):
    m, sm = masker(eeg[b].cpu(), 200)
    masks.append(m); syms.append(sm)
mask = torch.stack(masks).to(device)
sym  = torch.stack(syms).to(device)

with torch.no_grad():
    targets = model.tokenizer.get_codebook_indices(eeg4d)

with autocast('cuda'):
    logits, sym_logits = model.forward_stage1(eeg4d, mask, sym)
    loss, log = criterion(logits, sym_logits, targets, mask, sym)

scaler.scale(loss / 4).backward()
nn.utils.clip_grad_norm_(model.parameters(), 3.0)
scaler.step(opt); scaler.update(); opt.zero_grad()
print(f'    ok: loss={loss.item():.4f}')

print('[6] EMA update...')
with torch.no_grad():
    for ep, p in zip(ema.parameters(), model.parameters()):
        ep.data.mul_(0.996).add_(p.data, alpha=0.004)
print('    ok')

print('\nAll checks passed!')
