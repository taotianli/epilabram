"""
Retrieval-Augmented Generation (RAG) for EEG Foundation Models.
Retrieves similar historical EEG cases to improve diagnosis and prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import faiss
import pickle
from pathlib import Path


class EEGRetriever:
    """
    Retrieves similar EEG cases from a database using FAISS.
    """

    def __init__(
        self,
        embedding_dim: int,
        index_type: str = 'flat',  # 'flat', 'ivf', 'hnsw'
        metric: str = 'l2',  # 'l2' or 'cosine'
        use_gpu: bool = False,
    ):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric = metric
        self.use_gpu = use_gpu

        # Initialize FAISS index
        self.index = self._build_index()
        self.metadata = []  # Store metadata for each embedding

    def _build_index(self) -> faiss.Index:
        """Build FAISS index based on configuration."""
        if self.metric == 'cosine':
            # For cosine similarity, normalize embeddings and use L2
            if self.index_type == 'flat':
                index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product
            elif self.index_type == 'ivf':
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
            elif self.index_type == 'hnsw':
                index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            else:
                raise ValueError(f"Unknown index_type: {self.index_type}")
        else:  # L2
            if self.index_type == 'flat':
                index = faiss.IndexFlatL2(self.embedding_dim)
            elif self.index_type == 'ivf':
                quantizer = faiss.IndexFlatL2(self.embedding_dim)
                index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
            elif self.index_type == 'hnsw':
                index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            else:
                raise ValueError(f"Unknown index_type: {self.index_type}")

        # Move to GPU if requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        return index

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
    ):
        """
        Add embeddings to the index.

        Args:
            embeddings: (N, D) numpy array of embeddings
            metadata: List of metadata dicts (length N)
        """
        assert len(embeddings) == len(metadata)

        # Normalize for cosine similarity
        if self.metric == 'cosine':
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Train index if needed (for IVF)
        if self.index_type == 'ivf' and not self.index.is_trained:
            self.index.train(embeddings)

        # Add to index
        self.index.add(embeddings.astype(np.float32))
        self.metadata.extend(metadata)

    def search(
        self,
        query_embeddings: Union[torch.Tensor, np.ndarray],
        k: int = 5,
        return_metadata: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[List[List[Dict]]]]:
        """
        Search for k nearest neighbors.

        Args:
            query_embeddings: (B, D) query embeddings
            k: Number of neighbors to retrieve
            return_metadata: Whether to return metadata

        Returns:
            distances: (B, k) distances to neighbors
            indices: (B, k) indices of neighbors
            metadata: List of lists of metadata dicts (if return_metadata=True)
        """
        # Convert to numpy if needed
        if isinstance(query_embeddings, torch.Tensor):
            query_embeddings = query_embeddings.detach().cpu().numpy()

        # Normalize for cosine similarity
        if self.metric == 'cosine':
            query_embeddings = query_embeddings / np.linalg.norm(
                query_embeddings, axis=1, keepdims=True
            )

        # Search
        distances, indices = self.index.search(query_embeddings.astype(np.float32), k)

        # Retrieve metadata
        retrieved_metadata = None
        if return_metadata:
            retrieved_metadata = []
            for batch_indices in indices:
                batch_metadata = [self.metadata[idx] for idx in batch_indices]
                retrieved_metadata.append(batch_metadata)

        return distances, indices, retrieved_metadata

    def save(self, save_dir: str):
        """Save index and metadata to disk."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        if self.use_gpu:
            # Move to CPU before saving
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(save_dir / 'index.faiss'))
        else:
            faiss.write_index(self.index, str(save_dir / 'index.faiss'))

        # Save metadata
        with open(save_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(self.metadata, f)

        # Save config
        config = {
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'metric': self.metric,
        }
        with open(save_dir / 'config.pkl', 'wb') as f:
            pickle.dump(config, f)

    @classmethod
    def load(cls, load_dir: str, use_gpu: bool = False) -> 'EEGRetriever':
        """Load index and metadata from disk."""
        load_dir = Path(load_dir)

        # Load config
        with open(load_dir / 'config.pkl', 'rb') as f:
            config = pickle.load(f)

        # Create retriever
        retriever = cls(
            embedding_dim=config['embedding_dim'],
            index_type=config['index_type'],
            metric=config['metric'],
            use_gpu=use_gpu,
        )

        # Load FAISS index
        index = faiss.read_index(str(load_dir / 'index.faiss'))
        if use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        retriever.index = index

        # Load metadata
        with open(load_dir / 'metadata.pkl', 'rb') as f:
            retriever.metadata = pickle.load(f)

        return retriever


class RetrievalFusion(nn.Module):
    """
    Fuses query EEG features with retrieved case features.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Cross-attention to fuse query with retrieved cases
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query_features: torch.Tensor,
        retrieved_features: torch.Tensor,
        retrieved_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuse query with retrieved features.

        Args:
            query_features: (B, D) or (B, 1, D)
            retrieved_features: (B, k, D) retrieved case features
            retrieved_mask: (B, k) mask for valid retrieved cases

        Returns:
            fused_features: (B, D)
        """
        if query_features.dim() == 2:
            query_features = query_features.unsqueeze(1)  # (B, 1, D)

        # Cross-attention: query attends to retrieved cases
        x = query_features
        for attn, norm in zip(self.cross_attn_layers, self.norms):
            attn_out, _ = attn(
                x, retrieved_features, retrieved_features,
                key_padding_mask=~retrieved_mask if retrieved_mask is not None else None
            )
            x = norm(x + attn_out)

        # FFN
        x = self.ffn_norm(x + self.ffn(x))

        # Return single vector
        return x.squeeze(1)  # (B, D)


class RAGEEGModel(nn.Module):
    """
    Retrieval-Augmented EEG model.

    Retrieves similar historical cases and uses them to improve predictions.
    """

    def __init__(
        self,
        backbone: nn.Module,
        retriever: Optional[EEGRetriever] = None,
        fusion_module: Optional[RetrievalFusion] = None,
        num_retrieved: int = 5,
        use_retrieval: bool = True,
        num_classes: int = 2,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.embed_dim
        self.num_retrieved = num_retrieved
        self.use_retrieval = use_retrieval

        # Retriever
        if retriever is None and use_retrieval:
            retriever = EEGRetriever(
                embedding_dim=self.embed_dim,
                index_type='flat',
                metric='cosine',
            )
        self.retriever = retriever

        # Fusion module
        if fusion_module is None and use_retrieval:
            fusion_module = RetrievalFusion(
                embed_dim=self.embed_dim,
                num_heads=8,
                num_layers=2,
            )
        self.fusion = fusion_module

        # Classifier
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def extract_features(self, eeg: torch.Tensor) -> torch.Tensor:
        """Extract features from EEG using backbone."""
        return self.backbone(eeg)

    def forward(
        self,
        eeg: torch.Tensor,
        use_retrieval: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with optional retrieval.

        Args:
            eeg: (B, N, A, T) EEG data
            use_retrieval: Whether to use retrieval (overrides self.use_retrieval)

        Returns:
            logits: (B, num_classes)
            retrieval_info: Dict with retrieval information (if retrieval is used)
        """
        use_retrieval = use_retrieval if use_retrieval is not None else self.use_retrieval

        # Extract query features
        query_features = self.extract_features(eeg)  # (B, D)

        retrieval_info = None

        if use_retrieval and self.retriever is not None:
            # Retrieve similar cases
            distances, indices, metadata = self.retriever.search(
                query_features,
                k=self.num_retrieved,
                return_metadata=True,
            )

            # Get retrieved features from metadata
            # Assume metadata contains 'embedding' and 'label'
            retrieved_features = []
            retrieved_labels = []
            for batch_metadata in metadata:
                batch_features = torch.stack([
                    torch.tensor(m['embedding']) for m in batch_metadata
                ])
                batch_labels = torch.tensor([m['label'] for m in batch_metadata])
                retrieved_features.append(batch_features)
                retrieved_labels.append(batch_labels)

            retrieved_features = torch.stack(retrieved_features).to(eeg.device)  # (B, k, D)
            retrieved_labels = torch.stack(retrieved_labels).to(eeg.device)  # (B, k)

            # Fuse query with retrieved features
            fused_features = self.fusion(query_features, retrieved_features)

            # Store retrieval info
            retrieval_info = {
                'distances': distances,
                'indices': indices,
                'retrieved_labels': retrieved_labels,
                'metadata': metadata,
            }

            # Classify
            logits = self.classifier(fused_features)
        else:
            # Direct classification without retrieval
            logits = self.classifier(query_features)

        return logits, retrieval_info

    def build_database(
        self,
        eeg_data: List[torch.Tensor],
        labels: List[int],
        additional_metadata: Optional[List[Dict]] = None,
        batch_size: int = 32,
        device: str = 'cuda',
    ):
        """
        Build retrieval database from EEG data.

        Args:
            eeg_data: List of EEG tensors
            labels: List of labels
            additional_metadata: Optional additional metadata for each sample
            batch_size: Batch size for feature extraction
            device: Device to use
        """
        self.backbone.eval()

        all_embeddings = []
        all_metadata = []

        with torch.no_grad():
            for i in range(0, len(eeg_data), batch_size):
                batch_eeg = torch.stack(eeg_data[i:i + batch_size]).to(device)
                batch_labels = labels[i:i + batch_size]

                # Extract features
                embeddings = self.extract_features(batch_eeg)
                all_embeddings.append(embeddings.cpu().numpy())

                # Prepare metadata
                for j, (emb, label) in enumerate(zip(embeddings, batch_labels)):
                    metadata = {
                        'embedding': emb.cpu().numpy(),
                        'label': label,
                        'index': i + j,
                    }
                    if additional_metadata is not None:
                        metadata.update(additional_metadata[i + j])
                    all_metadata.append(metadata)

        # Concatenate all embeddings
        all_embeddings = np.concatenate(all_embeddings, axis=0)

        # Add to retriever
        self.retriever.add_embeddings(all_embeddings, all_metadata)

        print(f"Built database with {len(all_metadata)} samples")

    def save_database(self, save_dir: str):
        """Save retrieval database."""
        self.retriever.save(save_dir)

    def load_database(self, load_dir: str, use_gpu: bool = False):
        """Load retrieval database."""
        self.retriever = EEGRetriever.load(load_dir, use_gpu=use_gpu)


class AdaptiveRetrieval(nn.Module):
    """
    Adaptive retrieval that learns when to retrieve and how many cases to retrieve.
    """

    def __init__(
        self,
        embed_dim: int,
        max_retrieved: int = 10,
        min_retrieved: int = 1,
    ):
        super().__init__()
        self.max_retrieved = max_retrieved
        self.min_retrieved = min_retrieved

        # Retrieval decision network
        self.decision_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Number of cases to retrieve
        self.count_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, max_retrieved - min_retrieved + 1),
        )

    def forward(self, query_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decide whether to retrieve and how many cases.

        Args:
            query_features: (B, D)

        Returns:
            should_retrieve: (B,) binary decision
            num_to_retrieve: (B,) number of cases to retrieve
        """
        # Should retrieve?
        retrieve_prob = self.decision_net(query_features).squeeze(-1)  # (B,)
        should_retrieve = (retrieve_prob > 0.5).float()

        # How many to retrieve?
        count_logits = self.count_net(query_features)  # (B, max-min+1)
        count_probs = F.softmax(count_logits, dim=-1)
        num_to_retrieve = count_probs.argmax(dim=-1) + self.min_retrieved  # (B,)

        return should_retrieve, num_to_retrieve


def build_rag_model(
    backbone: nn.Module,
    num_retrieved: int = 5,
    num_classes: int = 2,
    retriever_index_type: str = 'flat',
    retriever_metric: str = 'cosine',
    use_gpu: bool = False,
) -> RAGEEGModel:
    """
    Factory function to build RAG model.

    Args:
        backbone: Pretrained EEG backbone
        num_retrieved: Number of cases to retrieve
        num_classes: Number of output classes
        retriever_index_type: FAISS index type
        retriever_metric: Distance metric
        use_gpu: Whether to use GPU for retrieval

    Returns:
        model: RAGEEGModel
    """
    retriever = EEGRetriever(
        embedding_dim=backbone.embed_dim,
        index_type=retriever_index_type,
        metric=retriever_metric,
        use_gpu=use_gpu,
    )

    fusion = RetrievalFusion(
        embed_dim=backbone.embed_dim,
        num_heads=8,
        num_layers=2,
    )

    model = RAGEEGModel(
        backbone=backbone,
        retriever=retriever,
        fusion_module=fusion,
        num_retrieved=num_retrieved,
        use_retrieval=True,
        num_classes=num_classes,
    )

    return model


class RetrievalAnalyzer:
    """
    Analyzes retrieval quality and provides insights.
    """

    def __init__(self):
        self.retrieval_logs = []

    def log_retrieval(
        self,
        query_label: int,
        retrieved_labels: List[int],
        distances: List[float],
        prediction: int,
        correct: bool,
    ):
        """Log a retrieval event."""
        self.retrieval_logs.append({
            'query_label': query_label,
            'retrieved_labels': retrieved_labels,
            'distances': distances,
            'prediction': prediction,
            'correct': correct,
        })

    def compute_retrieval_precision(self, k: int = 5) -> float:
        """
        Compute retrieval precision@k.
        Percentage of retrieved cases with same label as query.
        """
        precisions = []
        for log in self.retrieval_logs:
            query_label = log['query_label']
            retrieved_labels = log['retrieved_labels'][:k]
            precision = sum(l == query_label for l in retrieved_labels) / k
            precisions.append(precision)

        return np.mean(precisions)

    def compute_retrieval_impact(self) -> Dict[str, float]:
        """
        Compute impact of retrieval on accuracy.
        Compare cases where retrieval helped vs hurt.
        """
        helped = 0
        hurt = 0
        neutral = 0

        for log in self.retrieval_logs:
            query_label = log['query_label']
            retrieved_labels = log['retrieved_labels']
            correct = log['correct']

            # Majority vote from retrieved cases
            retrieved_prediction = max(set(retrieved_labels), key=retrieved_labels.count)

            if correct:
                if retrieved_prediction == query_label:
                    helped += 1
                else:
                    neutral += 1
            else:
                if retrieved_prediction == query_label:
                    hurt += 1
                else:
                    neutral += 1

        total = len(self.retrieval_logs)
        return {
            'helped': helped / total,
            'hurt': hurt / total,
            'neutral': neutral / total,
        }

    def print_statistics(self):
        """Print retrieval statistics."""
        print("\n" + "=" * 60)
        print("Retrieval Statistics")
        print("=" * 60)

        precision = self.compute_retrieval_precision(k=5)
        print(f"Retrieval Precision@5: {precision:.4f}")

        impact = self.compute_retrieval_impact()
        print(f"\nRetrieval Impact:")
        print(f"  Helped: {impact['helped']:.2%}")
        print(f"  Hurt: {impact['hurt']:.2%}")
        print(f"  Neutral: {impact['neutral']:.2%}")

        # Average distance to retrieved cases
        avg_distances = [np.mean(log['distances']) for log in self.retrieval_logs]
        print(f"\nAverage distance to retrieved cases: {np.mean(avg_distances):.4f}")
