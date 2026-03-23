"""
单元测试：models/labram_backbone.py
"""

import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.labram_backbone import (
    TemporalEncoder, SpatialEmbedding, TemporalPositionEmbedding,
    LaBraMTransformerBlock, LaBraMBackbone, BACKBONE_CONFIGS
)


class TestTemporalEncoder(unittest.TestCase):
    def setUp(self):
        self.B, self.N, self.A, self.T = 2, 23, 4, 200
        self.encoder = TemporalEncoder(in_chans=1, out_chans=8)

    def test_output_shape(self):
        x = torch.randn(self.B, self.N, self.A, self.T)
        out = self.encoder(x)
        self.assertEqual(out.shape[0], self.B)
        self.assertEqual(out.shape[1], self.N * self.A)

    def test_no_nan(self):
        x = torch.randn(self.B, self.N, self.A, self.T)
        out = self.encoder(x)
        self.assertFalse(torch.isnan(out).any())


class TestLaBraMTransformerBlock(unittest.TestCase):
    def setUp(self):
        self.dim = 200
        self.block = LaBraMTransformerBlock(dim=self.dim, num_heads=10)

    def test_output_shape(self):
        B, N = 2, 50
        x = torch.randn(B, N, self.dim)
        out = self.block(x)
        self.assertEqual(out.shape, (B, N, self.dim))

    def test_attention_return(self):
        B, N = 2, 50
        x = torch.randn(B, N, self.dim)
        attn = self.block(x, return_attention=True)
        self.assertEqual(attn.shape[0], B)
        self.assertEqual(attn.shape[-1], N)

    def test_residual_connection(self):
        # With zero weights, output should be close to input
        B, N = 2, 10
        x = torch.randn(B, N, self.dim)
        out = self.block(x)
        self.assertEqual(out.shape, x.shape)


class TestLaBraMBackbone(unittest.TestCase):
    def setUp(self):
        self.B, self.N, self.A, self.T = 2, 23, 4, 200
        self.backbone = LaBraMBackbone(size='base')

    def test_forward_pooled(self):
        x = torch.randn(self.B, self.N, self.A, self.T)
        out = self.backbone(x)
        self.assertEqual(out.shape, (self.B, 200))

    def test_forward_patch_tokens(self):
        x = torch.randn(self.B, self.N, self.A, self.T)
        out = self.backbone(x, return_patch_tokens=True)
        self.assertEqual(out.shape, (self.B, self.N * self.A, 200))

    def test_forward_all_hidden(self):
        x = torch.randn(self.B, self.N, self.A, self.T)
        hidden_states, patch_tokens = self.backbone(x, return_all_hidden=True)
        self.assertEqual(len(hidden_states), 12)  # base depth=12
        self.assertEqual(patch_tokens.shape, (self.B, self.N * self.A, 200))

    def test_no_nan(self):
        x = torch.randn(self.B, self.N, self.A, self.T)
        out = self.backbone(x)
        self.assertFalse(torch.isnan(out).any())

    def test_backbone_configs(self):
        for size in ['base', 'large']:
            cfg = BACKBONE_CONFIGS[size]
            self.assertIn('depth', cfg)
            self.assertIn('hidden', cfg)
            self.assertIn('heads', cfg)

    def test_masked_forward(self):
        x = torch.randn(self.B, self.N, self.A, self.T)
        mask = torch.rand(self.B, self.N * self.A) > 0.5
        mask_token = torch.nn.Parameter(torch.zeros(1, 1, 200))
        out = self.backbone(x, bool_masked_pos=mask, mask_token=mask_token,
                            return_patch_tokens=True)
        self.assertEqual(out.shape, (self.B, self.N * self.A, 200))


if __name__ == '__main__':
    unittest.main()
