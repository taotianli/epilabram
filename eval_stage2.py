#!/usr/bin/env python3
"""
Stage2 评估：5类伪迹分类准确率、F1、混淆矩阵
"""
import sys, os, faulthandler
faulthandler.enable(file=open('/tmp/eval_fault.log', 'w'), all_threads=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if 'LaBraM' not in p]
sys.path.insert(0, _HERE)

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast
from sklearn.metrics import (
    classification_report, confusion_matrix,
    balanced_accuracy_score, f1_score
)

from models.epilabram import build_epilabram
from data.tuar_edf_dataset import TUAREDFDataset, ARTIFACT_CLASS_NAMES

def main():
    device = torch.device('cuda')

    print('[1] loading model...')
    model = build_epilabram(
        backbone_size='base',
        vqnsp_path='/home/taotl/Desktop/LaBraM/checkpoints/vqnsp.pth',
        task_mode='artifact',
        n_classes=5,
    ).to(device)

    ckpt_path = 'experiments/stage2_full/stage2_best.pth'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('model', ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f'    loaded {ckpt_path}')

    print('[2] loading dataset (eval split, stride=10s)...')
    # 用 stride=10s（不重叠）做评估，避免数据泄露
    ds = TUAREDFDataset(
        '/home/taotl/Desktop/TUAR/v3.0.1/edf',
        window_sec=10.0,
        stride_sec=10.0,
        mode='artifact',
    )
    print(f'    {len(ds)} segments')

    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

    print('[3] inference...')
    all_preds, all_labels = [], []
    all_probs = []

    with torch.no_grad():
        for eeg, labels in loader:
            eeg = eeg.to(device)
            B, C, T = eeg.shape
            A = T // 200
            eeg4d = eeg.reshape(B, C, A, 200)
            task_ids = torch.zeros(B, dtype=torch.long, device=device)

            with autocast('cuda'):
                results = model.forward_stage2(eeg4d, task_ids)
            _, logits = results['TUAB']
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.append(probs)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.concatenate(all_probs, axis=0)

    print('\n' + '='*60)
    print('Evaluation Results')
    print('='*60)

    # 整体指标
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    wf1     = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    mf1     = f1_score(all_labels, all_preds, average='macro',    zero_division=0)
    acc     = (all_preds == all_labels).mean()

    print(f'Accuracy:          {acc:.4f}')
    print(f'Balanced Accuracy: {bal_acc:.4f}')
    print(f'Weighted F1:       {wf1:.4f}')
    print(f'Macro F1:          {mf1:.4f}')

    # 每类报告
    print('\nPer-class Report:')
    present = sorted(np.unique(all_labels).tolist())
    names   = [ARTIFACT_CLASS_NAMES[i] for i in present]
    print(classification_report(
        all_labels, all_preds,
        labels=present, target_names=names,
        zero_division=0
    ))

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=present)
    print('Confusion Matrix (rows=true, cols=pred):')
    header = '       ' + '  '.join(f'{n:>6}' for n in names)
    print(header)
    for i, row in enumerate(cm):
        print(f'{names[i]:>6} ' + '  '.join(f'{v:>6}' for v in row))

    # 标签分布
    print('\nLabel distribution:')
    for i in present:
        n = (all_labels == i).sum()
        print(f'  {ARTIFACT_CLASS_NAMES[i]:>6}: {n:5d} ({n/len(all_labels)*100:.1f}%)')

if __name__ == '__main__':
    main()
