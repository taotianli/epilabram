"""
Unit tests for extended features: RoPE, LoRA, Temporal Transformer, and ICL.
"""

import unittest
import torch
import torch.nn as nn

from models.rope import RotaryEmbedding, RoPEAttention, apply_rotary_pos_emb
from models.lora import LoRALayer, LoRALinear, LoRAAttention, add_lora_to_model, get_lora_params
from models.temporal_transformer import TemporalTransformer, build_temporal_transformer
from models.epilabram_extended import build_epilabram_extended
from models.labram_backbone import LaBraMBackbone


class TestRoPE(unittest.TestCase):
    """Test RoPE implementation"""

    def test_rotary_embedding_init(self):
        """Test RotaryEmbedding initialization"""
        rope = RotaryEmbedding(dim=64, max_seq_len=512)
        self.assertEqual(rope.dim, 64)
        self.assertEqual(rope.max_seq_len, 512)

    def test_rotary_embedding_forward(self):
        """Test RotaryEmbedding forward pass"""
        rope = RotaryEmbedding(dim=64, max_seq_len=512)
        cos, sin = rope(seq_len=100)
        self.assertEqual(cos.shape, (100, 64))
        self.assertEqual(sin.shape, (100, 64))

    def test_rotary_embedding_extension(self):
        """Test context length extension"""
        rope = RotaryEmbedding(dim=64, max_seq_len=512)

        # Test with longer sequence
        cos, sin = rope(seq_len=1024)
        self.assertEqual(cos.shape, (1024, 64))
        self.assertEqual(sin.shape, (1024, 64))
        self.assertEqual(rope.max_seq_len, 1024)

    def test_apply_rotary_pos_emb(self):
        """Test applying RoPE to Q and K"""
        B, H, N, D = 2, 8, 100, 64
        q = torch.randn(B, H, N, D)
        k = torch.randn(B, H, N, D)

        rope = RotaryEmbedding(dim=D)
        cos, sin = rope(N)

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)

    def test_rope_attention(self):
        """Test RoPEAttention module"""
        dim = 200
        num_heads = 10
        attn = RoPEAttention(dim, num_heads=num_heads, max_seq_len=512)

        B, N, D = 2, 100, dim
        x = torch.randn(B, N, D)

        out = attn(x)
        self.assertEqual(out.shape, (B, N, D))


class TestLoRA(unittest.TestCase):
    """Test LoRA implementation"""

    def test_lora_layer_init(self):
        """Test LoRALayer initialization"""
        lora = LoRALayer(in_dim=200, out_dim=200, rank=8, alpha=16.0)
        self.assertEqual(lora.rank, 8)
        self.assertEqual(lora.alpha, 16.0)
        self.assertEqual(lora.lora_A.shape, (8, 200))
        self.assertEqual(lora.lora_B.shape, (200, 8))

    def test_lora_layer_forward(self):
        """Test LoRALayer forward pass"""
        lora = LoRALayer(in_dim=200, out_dim=200, rank=8)
        x = torch.randn(2, 100, 200)
        out = lora(x)
        self.assertEqual(out.shape, (2, 100, 200))

    def test_lora_linear(self):
        """Test LoRALinear wrapper"""
        linear = nn.Linear(200, 200)
        lora_linear = LoRALinear(linear, rank=8)

        # Check that original weights are frozen
        self.assertFalse(lora_linear.linear.weight.requires_grad)

        x = torch.randn(2, 100, 200)
        out = lora_linear(x)
        self.assertEqual(out.shape, (2, 100, 200))

    def test_add_lora_to_model(self):
        """Test adding LoRA to a model"""
        backbone = LaBraMBackbone(size='base', use_rope=False)
        original_params = sum(p.numel() for p in backbone.parameters())

        # Add LoRA
        backbone = add_lora_to_model(backbone, rank=8)

        # Check that LoRA parameters were added
        lora_params = get_lora_params(backbone)
        self.assertGreater(len(lora_params), 0)

        # Total parameters should increase
        new_params = sum(p.numel() for p in backbone.parameters())
        self.assertGreater(new_params, original_params)

    def test_lora_parameter_efficiency(self):
        """Test LoRA parameter efficiency"""
        backbone = LaBraMBackbone(size='base', use_rope=False)
        original_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)

        # Add LoRA and freeze backbone
        backbone = add_lora_to_model(backbone, rank=8)
        for name, param in backbone.named_parameters():
            if 'lora' not in name.lower():
                param.requires_grad = False

        lora_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)

        # LoRA should use much fewer parameters
        efficiency = lora_trainable / original_trainable
        self.assertLess(efficiency, 0.1)  # Less than 10%


class TestTemporalTransformer(unittest.TestCase):
    """Test Temporal Transformer implementation"""

    def test_temporal_transformer_init(self):
        """Test TemporalTransformer initialization"""
        model = TemporalTransformer(
            embed_dim=200,
            depth=6,
            num_heads=8,
            num_classes=2,
        )
        self.assertEqual(model.embed_dim, 200)
        self.assertEqual(len(model.blocks), 6)

    def test_temporal_transformer_forward(self):
        """Test standard forward pass"""
        model = TemporalTransformer(embed_dim=200, depth=4, num_heads=8, num_classes=2)

        B, T, D = 2, 16, 200
        x = torch.randn(B, T, D)

        logits = model(x)
        self.assertEqual(logits.shape, (B, 2))

    def test_temporal_transformer_with_mask(self):
        """Test forward pass with sequence mask"""
        model = TemporalTransformer(embed_dim=200, depth=4, num_heads=8, num_classes=2)

        B, T, D = 2, 16, 200
        x = torch.randn(B, T, D)
        mask = torch.ones(B, T).bool()
        mask[0, 12:] = False  # Mask last 4 positions

        logits = model(x, mask=mask)
        self.assertEqual(logits.shape, (B, 2))

    def test_temporal_transformer_icl(self):
        """Test ICL forward pass"""
        model = TemporalTransformer(embed_dim=200, depth=4, num_heads=8, num_classes=2)

        B, n_demo, n_query, D = 2, 4, 3, 200
        demo_emb = torch.randn(B, n_demo, D)
        demo_labels = torch.randint(0, 2, (B, n_demo))
        query_emb = torch.randn(B, n_query, D)

        logits = model.forward_icl(demo_emb, demo_labels, query_emb)
        self.assertEqual(logits.shape, (B, n_query, 2))

    def test_temporal_transformer_icl_single_query(self):
        """Test ICL with single query"""
        model = TemporalTransformer(embed_dim=200, depth=4, num_heads=8, num_classes=2)

        B, n_demo, D = 2, 4, 200
        demo_emb = torch.randn(B, n_demo, D)
        demo_labels = torch.randint(0, 2, (B, n_demo))
        query_emb = torch.randn(B, D)

        logits = model.forward_icl_single_query(demo_emb, demo_labels, query_emb)
        self.assertEqual(logits.shape, (B, 2))

    def test_build_temporal_transformer(self):
        """Test factory function"""
        for size in ['small', 'base', 'large']:
            model = build_temporal_transformer(
                embed_dim=200,
                size=size,
                num_classes=2,
            )
            self.assertIsInstance(model, TemporalTransformer)


class TestEpiLaBraMExtended(unittest.TestCase):
    """Test extended EpiLaBraM model"""

    def test_build_with_rope(self):
        """Test building model with RoPE"""
        model = build_epilabram_extended(
            backbone_size='base',
            use_rope=True,
            use_lora=False,
            use_temporal=False,
        )
        self.assertTrue(model.backbone.use_rope)

    def test_build_with_lora(self):
        """Test building model with LoRA"""
        model = build_epilabram_extended(
            backbone_size='base',
            use_rope=False,
            use_lora=True,
            lora_rank=8,
            use_temporal=False,
        )
        self.assertTrue(model.use_lora)

        # Check LoRA parameters exist
        lora_params = [n for n, p in model.named_parameters() if 'lora' in n.lower()]
        self.assertGreater(len(lora_params), 0)

    def test_build_with_temporal(self):
        """Test building model with temporal transformer"""
        model = build_epilabram_extended(
            backbone_size='base',
            use_rope=True,
            use_lora=False,
            use_temporal=True,
            temporal_size='base',
        )
        self.assertIsNotNone(model.temporal_transformer)

    def test_stage1_forward(self):
        """Test Stage 1 forward pass"""
        model = build_epilabram_extended(
            backbone_size='base',
            use_rope=True,
            use_lora=True,
            use_temporal=False,
        )

        B, N, A, T = 2, 23, 4, 200
        eeg = torch.randn(B, N, A, T)
        mask = torch.rand(B, N * A) > 0.5
        sym_mask = torch.rand(B, N * A) > 0.5

        logits, sym_logits = model.forward_stage1(eeg, mask, sym_mask)

        self.assertEqual(logits.shape, (B, N * A, model.tokenizer.codebook.n_embed))
        self.assertEqual(sym_logits.shape, (B, N * A, model.tokenizer.codebook.n_embed))

    def test_stage2_forward(self):
        """Test Stage 2 forward pass"""
        model = build_epilabram_extended(
            backbone_size='base',
            use_rope=True,
            use_lora=False,
            use_temporal=False,
        )

        B, N, A, T = 2, 23, 4, 200
        eeg = torch.randn(B, N, A, T)
        task_id = torch.tensor([0, 1])

        results = model.forward_stage2(eeg, task_id)

        self.assertIn('TUAB', results)
        self.assertIn('TUSZ', results)

    def test_extract_cls_embeddings(self):
        """Test CLS embedding extraction"""
        model = build_epilabram_extended(
            backbone_size='base',
            use_rope=True,
            use_lora=False,
            use_temporal=False,
        )

        B, N, A, T = 2, 23, 4, 200
        eeg = torch.randn(B, N, A, T)

        cls_emb = model.extract_cls_embeddings(eeg)

        self.assertEqual(cls_emb.shape, (B, model.backbone.embed_dim))

    def test_temporal_forward(self):
        """Test temporal transformer forward"""
        model = build_epilabram_extended(
            backbone_size='base',
            use_rope=True,
            use_lora=False,
            use_temporal=True,
            temporal_size='base',
            temporal_num_classes=2,
        )

        B, T_seq, N, A, T = 2, 8, 23, 4, 200
        eeg_sequence = torch.randn(B, T_seq, N, A, T)
        task_id = torch.tensor([0, 1])

        logits = model.forward_temporal(eeg_sequence, task_id=task_id)

        self.assertEqual(logits.shape, (B, 2))

    def test_icl_forward(self):
        """Test ICL inference"""
        model = build_epilabram_extended(
            backbone_size='base',
            use_rope=True,
            use_lora=False,
            use_temporal=True,
            temporal_size='base',
            temporal_num_classes=2,
        )

        B, n_demo, n_query, N, A, T = 2, 4, 3, 23, 4, 200
        demo_eeg = torch.randn(B, n_demo, N, A, T)
        demo_labels = torch.randint(0, 2, (B, n_demo))
        query_eeg = torch.randn(B, n_query, N, A, T)

        logits = model.forward_icl(demo_eeg, demo_labels, query_eeg)

        self.assertEqual(logits.shape, (B, n_query, 2))

    def test_parameter_management(self):
        """Test parameter freezing and retrieval"""
        model = build_epilabram_extended(
            backbone_size='base',
            use_rope=True,
            use_lora=True,
            use_temporal=True,
        )

        # Test freezing
        model.freeze_backbone()
        backbone_trainable = sum(
            p.numel() for n, p in model.backbone.named_parameters()
            if p.requires_grad and 'lora' not in n.lower()
        )
        self.assertEqual(backbone_trainable, 0)

        # Test unfreezing
        model.unfreeze_backbone()
        backbone_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
        self.assertGreater(backbone_trainable, 0)

        # Test parameter getters
        stage1_params = model.get_stage1_params()
        stage2_params = model.get_stage2_params()
        temporal_params = model.get_temporal_params()

        self.assertGreater(len(stage1_params), 0)
        self.assertGreater(len(stage2_params), 0)
        self.assertGreater(len(temporal_params), 0)

    def test_context_length_extension(self):
        """Test processing different sequence lengths with RoPE"""
        model = build_epilabram_extended(
            backbone_size='base',
            use_rope=True,
            use_lora=False,
            use_temporal=True,
            max_seq_len=512,
        )

        B, N, A, T = 1, 23, 4, 200

        # Test different sequence lengths
        for seq_len in [4, 8, 16, 32]:
            eeg_sequence = torch.randn(B, seq_len, N, A, T)
            logits = model.forward_temporal(eeg_sequence)
            self.assertEqual(logits.shape, (B, 2))


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows"""

    def test_full_training_pipeline(self):
        """Test complete training pipeline"""
        model = build_epilabram_extended(
            backbone_size='base',
            use_rope=True,
            use_lora=True,
            lora_rank=8,
            use_temporal=True,
            temporal_size='base',
        )

        B, N, A, T = 2, 23, 4, 200

        # Stage 1: PADM with LoRA
        model.freeze_backbone()
        eeg = torch.randn(B, N, A, T)
        mask = torch.rand(B, N * A) > 0.5
        sym_mask = torch.rand(B, N * A) > 0.5

        logits, sym_logits = model.forward_stage1(eeg, mask, sym_mask)
        self.assertEqual(logits.shape[0], B)

        # Stage 2: Multi-task fine-tuning
        task_id = torch.tensor([0, 1])
        results = model.forward_stage2(eeg, task_id)
        self.assertGreater(len(results), 0)

        # Temporal modeling
        T_seq = 8
        eeg_sequence = torch.randn(B, T_seq, N, A, T)
        logits_temporal = model.forward_temporal(eeg_sequence, task_id=task_id)
        self.assertEqual(logits_temporal.shape, (B, 2))

        # ICL inference
        n_demo, n_query = 4, 2
        demo_eeg = torch.randn(B, n_demo, N, A, T)
        demo_labels = torch.randint(0, 2, (B, n_demo))
        query_eeg = torch.randn(B, n_query, N, A, T)
        logits_icl = model.forward_icl(demo_eeg, demo_labels, query_eeg)
        self.assertEqual(logits_icl.shape, (B, n_query, 2))

    def test_rope_lora_compatibility(self):
        """Test RoPE and LoRA work together"""
        model = build_epilabram_extended(
            backbone_size='base',
            use_rope=True,
            use_lora=True,
            lora_rank=8,
            use_temporal=False,
        )

        B, N, A, T = 2, 23, 4, 200
        eeg = torch.randn(B, N, A, T)
        mask = torch.rand(B, N * A) > 0.5
        sym_mask = torch.rand(B, N * A) > 0.5

        # Should work without errors
        logits, sym_logits = model.forward_stage1(eeg, mask, sym_mask)
        self.assertEqual(logits.shape[0], B)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRoPE))
    suite.addTests(loader.loadTestsFromTestCase(TestLoRA))
    suite.addTests(loader.loadTestsFromTestCase(TestTemporalTransformer))
    suite.addTests(loader.loadTestsFromTestCase(TestEpiLaBraMExtended))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
