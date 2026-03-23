"""
可视化工具：AttentionVisualizer / FrequencyBandAnalyzer / tSNEVisualizer
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, List
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class AttentionVisualizer:
    """提取并可视化 multi-head attention 权重"""

    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device

    def plot_attention_map(
        self,
        eeg_sample: torch.Tensor,
        task_id: int = 0,
        layer_idx: int = -1,
        save_path: Optional[str] = None,
    ):
        """
        提取指定层的 attention 权重，绘制 EEG通道×时间patch 热图。

        Args:
            eeg_sample: (C, T) 单样本
            task_id: 任务ID
            layer_idx: 层索引，-1 表示最后一层
            save_path: 保存路径，None 则 plt.show()
        """
        self.model.eval()
        C, T = eeg_sample.shape
        patch_size = 200
        A = T // patch_size
        eeg = eeg_sample[:, :A * patch_size].reshape(1, C, A, patch_size).to(self.device)
        task_ids = torch.tensor([task_id], device=self.device)

        # 手动 forward 到指定层，提取 attention
        backbone = self.model.backbone
        x = backbone.patch_embed(eeg)
        x = backbone.patch_proj(x)
        cls = backbone.cls_token.expand(1, -1, -1)
        x = torch.cat([cls, x], dim=1)
        pos = backbone.spatial_embed(C, A, batch_size=1)
        x = x + pos
        te = backbone.temporal_embed(x, C, A)
        x[:, 1:] = x[:, 1:] + te
        x = backbone.pos_drop(x)

        # 拼接 prompt tokens
        x = self.model.task_prompts(task_ids, x)

        blocks = backbone.blocks
        if layer_idx < 0:
            layer_idx = len(blocks) + layer_idx

        with torch.no_grad():
            for i, (blk, adapter) in enumerate(zip(blocks, self.model.adapters)):
                if i == layer_idx:
                    attn_weights = blk(x, return_attention=True)  # (1, H, N, N)
                    break
                x = blk(x)
                x = adapter(x)

        # 平均所有 head，取 patch tokens 部分
        n_prompt = self.model.task_prompts.n_prompt
        attn = attn_weights[0].mean(0)  # (N, N)
        # 去掉 prompt 和 CLS
        patch_start = n_prompt + 1
        attn_patch = attn[patch_start:, patch_start:].cpu().numpy()  # (C*A, C*A)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 左：attention 热图
        im = axes[0].imshow(attn_patch, aspect='auto', cmap='hot', interpolation='nearest')
        axes[0].set_title(f'Attention Map (Layer {layer_idx})')
        axes[0].set_xlabel('Key patch (channel × time)')
        axes[0].set_ylabel('Query patch (channel × time)')
        plt.colorbar(im, ax=axes[0])

        # 右：原始 EEG 波形（前8通道）
        eeg_np = eeg_sample.cpu().numpy()
        n_show = min(8, C)
        for ch in range(n_show):
            offset = ch * 2.0
            axes[1].plot(eeg_np[ch] + offset, linewidth=0.5, label=f'Ch{ch}')
        axes[1].set_title('EEG Waveform')
        axes[1].set_xlabel('Time (samples)')
        axes[1].set_ylabel('Channel')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()


class FrequencyBandAnalyzer:
    """使用 Gradient × Input 分析模型对各频段的敏感性"""

    BANDS = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta':  (13, 30),
        'gamma': (30, 75),
    }

    def __init__(self, model, device: torch.device, sample_rate: float = 200.0):
        self.model = model
        self.device = device
        self.fs = sample_rate

    def _band_energy(self, eeg: np.ndarray, low: float, high: float) -> np.ndarray:
        """计算指定频段能量，shape (C,)"""
        T = eeg.shape[1]
        freqs = np.fft.rfftfreq(T, d=1.0 / self.fs)
        fft = np.fft.rfft(eeg, axis=-1)
        mask = (freqs >= low) & (freqs <= high)
        return (np.abs(fft[:, mask]) ** 2).mean(axis=-1)

    def plot_band_importance(
        self,
        eeg_sample: torch.Tensor,
        task_id: int = 0,
        save_path: Optional[str] = None,
    ):
        """
        Gradient × Input 方法分析各频段重要性。

        Args:
            eeg_sample: (C, T)
            task_id: 任务ID
            save_path: 保存路径
        """
        self.model.eval()
        C, T = eeg_sample.shape
        patch_size = 200
        A = T // patch_size
        eeg = eeg_sample[:, :A * patch_size].reshape(1, C, A, patch_size).to(self.device)
        eeg.requires_grad_(True)
        task_ids = torch.tensor([task_id], device=self.device)

        task_name = {0: 'TUAB', 1: 'TUSZ', 2: 'TUEV', 3: 'TUEP'}[task_id]
        results = self.model.forward_stage2(eeg, task_ids)
        _, logits = results[task_name]
        score = logits.max(dim=-1).values.sum()
        score.backward()

        grad = eeg.grad.detach().cpu().numpy()[0]  # (C, A, T)
        eeg_np = eeg_sample.cpu().numpy()          # (C, T)
        grad_flat = grad.reshape(C, A * patch_size)
        saliency = np.abs(grad_flat * eeg_np[:, :A * patch_size])  # (C, T)

        # 按频段计算平均显著性
        band_scores = {}
        for band_name, (low, high) in self.BANDS.items():
            freqs = np.fft.rfftfreq(A * patch_size, d=1.0 / self.fs)
            fft_sal = np.abs(np.fft.rfft(saliency, axis=-1))
            mask = (freqs >= low) & (freqs <= high)
            band_scores[band_name] = float(fft_sal[:, mask].mean())

        # 归一化
        total = sum(band_scores.values()) + 1e-8
        band_scores = {k: v / total for k, v in band_scores.items()}

        fig, ax = plt.subplots(figsize=(7, 4))
        bands = list(band_scores.keys())
        scores = [band_scores[b] for b in bands]
        colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']
        ax.bar(bands, scores, color=colors)
        ax.set_title('Frequency Band Importance (Gradient × Input)')
        ax.set_ylabel('Normalized Importance')
        ax.set_xlabel('Frequency Band')
        for i, (b, s) in enumerate(zip(bands, scores)):
            ax.text(i, s + 0.005, f'{s:.3f}', ha='center', fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
        return band_scores


class tSNEVisualizer:
    """t-SNE 可视化 embedding 空间"""

    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def plot_embedding_space(
        self,
        dataset,
        task_id: int = 0,
        max_samples: int = 2000,
        batch_size: int = 64,
        save_path: Optional[str] = None,
        perplexity: float = 30.0,
    ):
        """
        提取所有样本的 pooled embeddings，用 t-SNE 降维可视化。

        Args:
            dataset: torch Dataset
            task_id: 任务ID
            max_samples: 最多使用样本数
            save_path: 保存路径
        """
        from sklearn.manifold import TSNE
        from torch.utils.data import DataLoader, Subset
        import random

        self.model.eval()
        task_name = {0: 'TUAB', 1: 'TUSZ', 2: 'TUEV', 3: 'TUEP'}[task_id]

        # 随机子集
        indices = list(range(len(dataset)))
        if len(indices) > max_samples:
            indices = random.sample(indices, max_samples)
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

        embeddings, labels_list = [], []

        for batch in loader:
            eeg = batch[0].to(self.device)
            label = batch[-1]
            B, C, T = eeg.shape
            A = T // 200
            eeg = eeg[:, :, :A * 200].reshape(B, C, A, 200)
            task_ids = torch.full((B,), task_id, dtype=torch.long, device=self.device)

            # 提取 pooled features
            backbone = self.model.backbone
            x = backbone.patch_embed(eeg)
            x = backbone.patch_proj(x)
            cls = backbone.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            pos = backbone.spatial_embed(C, A, batch_size=B)
            x = x + pos
            te = backbone.temporal_embed(x, C, A)
            x[:, 1:] = x[:, 1:] + te
            x = backbone.pos_drop(x)
            x = self.model.task_prompts(task_ids, x)
            for blk, adapter in zip(backbone.blocks, self.model.adapters):
                x = blk(x)
                x = adapter(x)
            x = backbone.norm(x)
            n_prompt = self.model.task_prompts.n_prompt
            pooled = x[:, n_prompt + 1:].mean(dim=1).cpu().numpy()

            embeddings.append(pooled)
            labels_list.append(label.numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        labels_arr = np.concatenate(labels_list, axis=0)

        # t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        emb_2d = tsne.fit_transform(embeddings)

        # 绘图
        unique_labels = np.unique(labels_arr)
        colors = cm.tab10(np.linspace(0, 1, len(unique_labels)))
        fig, ax = plt.subplots(figsize=(8, 7))
        for lbl, color in zip(unique_labels, colors):
            mask = labels_arr == lbl
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                       c=[color], label=f'Class {lbl}', alpha=0.6, s=10)
        ax.set_title(f't-SNE Embedding Space ({task_name})')
        ax.set_xlabel('t-SNE dim 1')
        ax.set_ylabel('t-SNE dim 2')
        ax.legend(markerscale=2)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
