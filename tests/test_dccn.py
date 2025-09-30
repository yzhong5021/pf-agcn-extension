"""
test_dccn.py

Tests for DCCN_1D dilated causal convolutional network
"""

import torch
import torch.nn as nn
import pytest
from modules.dccn import DCCN_1D


class TestDCCN1D:
    
    def test_init_default_params(self):
        torch.manual_seed(42)
        model = DCCN_1D(embed_len=256)
        assert model.c == 256
        assert model.k == 3
        assert model.dilations == (1, 2, 4, 8)
        assert len(model.convs) == 4
        assert len(model.norms) == 4
        assert len(model.dropouts) == 4
    
    def test_init_custom_params(self):
        torch.manual_seed(42)
        model = DCCN_1D(embed_len=128, k_size=5, dilation=3, dropout=0.2)
        assert model.c == 128
        assert model.k == 5
        assert model.dilations == (1, 3, 9, 27)
        assert all(conv.kernel_size == (5,) for conv in model.convs)
        assert all(dropout.p == 0.2 for dropout in model.dropouts)
    
    def test_causal_pad(self):
        torch.manual_seed(42)
        model = DCCN_1D(embed_len=64, k_size=3)
        x = torch.randn(2, 64, 10)  # B, C, L
        padded = model._causal_pad(x, dilation=2)
        expected_pad = (3-1) * 2  # (k-1) * dilation
        assert padded.shape == (2, 64, 10 + expected_pad)
        assert torch.equal(padded[:, :, expected_pad:], x)
        assert torch.equal(padded[:, :, :expected_pad], torch.zeros(2, 64, expected_pad))
    
    def test_forward_shape(self):
        torch.manual_seed(42)
        batch_size, seq_len, embed_len = 4, 50, 256
        model = DCCN_1D(embed_len=embed_len)
        x = torch.randn(batch_size, seq_len, embed_len)
        
        output = model(x)
        assert output.shape == (batch_size, seq_len, embed_len)
    
    def test_forward_different_seq_lengths(self):
        torch.manual_seed(42)
        embed_len = 128
        model = DCCN_1D(embed_len=embed_len)
        
        for seq_len in [10, 25, 100]:
            x = torch.randn(2, seq_len, embed_len)
            output = model(x)
            assert output.shape == (2, seq_len, embed_len)
    
    def test_forward_single_sample(self):
        torch.manual_seed(42)
        model = DCCN_1D(embed_len=64)
        x = torch.randn(1, 20, 64)
        output = model(x)
        assert output.shape == (1, 20, 64)
    
    def test_gating_mechanism(self):
        torch.manual_seed(42)
        embed_len = 32
        model = DCCN_1D(embed_len=embed_len)
        x = torch.randn(2, 15, embed_len)
        
        # Check gating layer exists and has correct dimensions
        assert model.gating.in_channels == 4 * embed_len
        assert model.gating.out_channels == 4 * embed_len
        assert model.gating.kernel_size == (1,)
        
        output = model(x)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_residual_connections(self):
        torch.manual_seed(42)
        model = DCCN_1D(embed_len=64)
        x = torch.randn(2, 10, 64)
        
        # Test that model preserves some input signal through residuals
        with torch.no_grad():
            output = model(x)
            # Output should not be identical to input (due to processing)
            assert not torch.equal(output, x)
            # But should maintain reasonable magnitude due to residuals
            assert output.abs().mean() > 0.01
    
    def test_deterministic_output(self):
        embed_len = 128
        x = torch.randn(3, 25, embed_len)
        
        torch.manual_seed(123)
        model1 = DCCN_1D(embed_len=embed_len)
        output1 = model1(x)
        
        torch.manual_seed(123)
        model2 = DCCN_1D(embed_len=embed_len)
        output2 = model2(x)
        
        assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_gradient_flow(self):
        torch.manual_seed(42)
        model = DCCN_1D(embed_len=64)
        x = torch.randn(2, 20, 64, requires_grad=True)
        
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Check that model parameters have gradients
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
    
    def test_eval_mode(self):
        torch.manual_seed(42)
        model = DCCN_1D(embed_len=64, dropout=0.5)
        x = torch.randn(2, 15, 64)
        
        model.train()
        train_output = model(x)
        
        model.eval()
        eval_output = model(x)
        
        # Outputs should be different due to dropout
        assert not torch.equal(train_output, eval_output)
    
    def test_invalid_input_dimensions(self):
        torch.manual_seed(42)
        model = DCCN_1D(embed_len=64)
        
        # Wrong number of dimensions
        with pytest.raises((RuntimeError, ValueError)):
            model(torch.randn(64, 10))  # Missing batch dimension
        
        # Wrong embedding dimension
        with pytest.raises((RuntimeError, ValueError)):
            model(torch.randn(2, 10, 32))  # embed_len=64 expected
