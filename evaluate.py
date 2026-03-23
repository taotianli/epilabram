"""
评估入口
"""

import argparse
import os
import sys
import yaml
import torch

sys.path.insert(0, os.path.dirname(__file__))

from models.epilabram import build_epilabram
from data.tuh_dataset import TUABDataset, TUSZDataset, TUEVDataset, TUEPDataset
from data.preprocessing import EEGPreprocessor
from evaluation.evaluator import Evaluator
from evaluation.metrics import MetricTracker
from evaluation.visualization import AttentionVisualizer, FrequencyBandAnalyzer, tSNEVisualizer
from utils.seed import set_seed
from utils.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='EpiLaBraM Evaluation')
    parser.add_argument('--config', type=str, default='configs/stage2_mtpct.yaml')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='experiments/eval')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--tasks', nargs='+', default=['TUAB', 'TUSZ', 'TUEV', 'TUEP'])
    parser.add_argument('--save_vis', action='store_true', help='Save visualizations')
    parser.add_argument('--split', type=str, default='test')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_epilabram(
        backbone_size=cfg.get('model', {}).get('backbone_size', 'base'),
        n_prompt=cfg.get('model', {}).get('n_prompt_tokens', 10),
    ).to(device)
    load_checkpoint(args.ckpt, model, strict=False)
    model.eval()

    data_cfg = cfg.get('data', {})
    preprocessor = EEGPreprocessor(target_fs=data_cfg.get('sample_rate', 200))
    window_sec = data_cfg.get('window_sec', 10.0)
    stride_sec = data_cfg.get('stride_sec', 5.0)

    task_ds_map = {
        'TUAB': (TUABDataset, 'tuab_path'),
        'TUSZ': (TUSZDataset, 'tusz_path'),
        'TUEV': (TUEVDataset, 'tuev_path'),
        'TUEP': (TUEPDataset, 'tuep_path'),
    }

    datasets = {}
    for task in args.tasks:
        cls, path_key = task_ds_map[task]
        path = data_cfg.get(path_key)
        if path and os.path.exists(path):
            datasets[task] = cls(path, window_sec=window_sec, stride_sec=stride_sec,
                                 preprocessor=preprocessor, split=args.split)

    os.makedirs(args.output_dir, exist_ok=True)

    evaluator = Evaluator(model, device)
    all_results = evaluator.evaluate_all(datasets, batch_size=args.batch_size)

    # 打印结果
    tracker = MetricTracker()
    for task, ds in datasets.items():
        import numpy as np
        # 重新填充 tracker 用于格式化输出
        # (evaluator 内部已计算，这里直接打印 all_results)
        pass

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for task, metrics in all_results.items():
        print(f"\n[{task}]")
        for k, v in metrics.items():
            print(f"  {k:25s}: {v:.4f}")

    # 保存结果到文件
    result_path = os.path.join(args.output_dir, 'results.txt')
    with open(result_path, 'w') as f:
        for task, metrics in all_results.items():
            f.write(f"[{task}]\n")
            for k, v in metrics.items():
                f.write(f"  {k}: {v:.4f}\n")
            f.write("\n")
    print(f"\nResults saved to {result_path}")

    # 可视化
    if args.save_vis and datasets:
        first_task = list(datasets.keys())[0]
        first_ds = datasets[first_task]
        sample_eeg, *_ = first_ds[0]

        attn_vis = AttentionVisualizer(model, device)
        attn_vis.plot_attention_map(
            sample_eeg,
            task_id=list(Evaluator.TASK_ID_MAP.values())[0],
            save_path=os.path.join(args.output_dir, 'attention_map.png'),
        )

        freq_vis = FrequencyBandAnalyzer(model, device)
        freq_vis.plot_band_importance(
            sample_eeg,
            task_id=0,
            save_path=os.path.join(args.output_dir, 'band_importance.png'),
        )

        tsne_vis = tSNEVisualizer(model, device)
        tsne_vis.plot_embedding_space(
            first_ds,
            task_id=0,
            save_path=os.path.join(args.output_dir, 'tsne.png'),
        )
        print(f"Visualizations saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
