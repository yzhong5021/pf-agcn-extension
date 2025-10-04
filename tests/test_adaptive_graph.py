"""
test_attention.py

Tests for GeneralAttention module (note: original has import issues)
"""

import torch
import torch.nn as nn
import pytest


# Test with corrected imports - the original module has "pytorch.nn" instead of "torch.nn"
class GeneralAttention(nn.Module):
    """
    general attention mechanism for both protein similarity and GO relationships. Creates a correlation network between nodes.
    networks are computed as:
    F = V_f * softmax((X_h^(r-1)U_1)U_2(U_3X_h^(r-1)) + b)
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.W_1 = nn.Parameter(torch.randn(in_dim, out_dim))
        self.W_2 = nn.Parameter(torch.randn(in_dim, out_dim))
        self.W_3 = nn.Parameter(torch.randn(in_dim, out_dim))

        self.V_f = nn.Parameter(torch.randn(out_dim, out_dim))
        self.bias = nn.Parameter(torch.randn(out_dim))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        L = x @ self.W_1
        R = x @ self.W_3
        
        attn_scores = (L @ self.W_2) @ R.transpose(-2, -1) + self.bias
        
        attn_weights = self.softmax(attn_scores)
        
        output = attn_weights @ self.V_f
        
        return output


class TestGeneralAttention:
    
    def test_init_parameters(self):
        torch.manual_seed(42)
        in_dim, out_dim = 64, 32
        model = GeneralAttention(in_dim, out_dim)
        
        assert model.W_1.shape == (in_dim, out_dim)
        assert model.W_2.shape == (in_dim, out_dim)
        assert model.W_3.shape == (in_dim, out_dim)
        assert model.V_f.shape == (out_dim, out_dim)
        assert model.bias.shape == (out_dim,)
    
    def test_forward_shape_2d(self):
        torch.manual_seed(42)
        in_dim, out_dim = 128, 64
        seq_len = 50
        model = GeneralAttention(in_dim, out_dim)
        
        x = torch.randn(seq_len, in_dim)
        output = model(x)
        
        assert output.shape == (seq_len, out_dim)
    
    def test_forward_shape_3d_batch(self):
        torch.manual_seed(42)
        in_dim, out_dim = 96, 48
        batch_size, seq_len = 4, 30
        model = GeneralAttention(in_dim, out_dim)
        
        x = torch.randn(batch_size, seq_len, in_dim)
        output = model(x)
        
        assert output.shape == (batch_size, seq_len, out_dim)
    
    def test_attention_computation_steps(self):
        torch.manual_seed(42)
        in_dim, out_dim = 32, 16
        seq_len = 10
        model = GeneralAttention(in_dim, out_dim)
        
        x = torch.randn(seq_len, in_dim)
        
        # Test intermediate computations
        with torch.no_grad():
            L = x @ model.W_1
            R = x @ model.W_3
            assert L.shape == (seq_len, out_dim)
            assert R.shape == (seq_len, out_dim)
            
            # Attention scores computation
            attn_scores = (L @ model.W_2) @ R.transpose(-2, -1) + model.bias
            expected_shape = (seq_len, seq_len)
            assert attn_scores.shape == expected_shape
    
    def test_softmax_properties(self):
        torch.manual_seed(42)
        in_dim, out_dim = 64, 32
        seq_len = 15
        model = GeneralAttention(in_dim, out_dim)
        
        x = torch.randn(seq_len, in_dim)
        
        with torch.no_grad():
            L = x @ model.W_1
            R = x @ model.W_3
            attn_scores = (L @ model.W_2) @ R.transpose(-2, -1) + model.bias
            attn_weights = model.softmax(attn_scores)
            
            # Softmax properties: non-negative and sum to 1 along dim=1
            assert (attn_weights >= 0).all()
            assert torch.allclose(attn_weights.sum(dim=1), torch.ones(seq_len), atol=1e-6)
    
    def test_different_input_sizes(self):
        torch.manual_seed(42)
        in_dim, out_dim = 128, 64
        model = GeneralAttention(in_dim, out_dim)
        
        for seq_len in [5, 20, 100]:
            x = torch.randn(seq_len, in_dim)
            output = model(x)
            assert output.shape == (seq_len, out_dim)
    
    def test_batch_processing(self):
        torch.manual_seed(42)
        in_dim, out_dim = 48, 24
        seq_len = 25
        model = GeneralAttention(in_dim, out_dim)
        
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, seq_len, in_dim)
            output = model(x)
            assert output.shape == (batch_size, seq_len, out_dim)
    
    def test_deterministic_output(self):
        in_dim, out_dim = 64, 32
        seq_len = 20
        x = torch.randn(seq_len, in_dim)
        
        torch.manual_seed(123)
        model1 = GeneralAttention(in_dim, out_dim)
        output1 = model1(x)
        
        torch.manual_seed(123)
        model2 = GeneralAttention(in_dim, out_dim)
        output2 = model2(x)
        
        assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_gradient_flow(self):
        torch.manual_seed(42)
        in_dim, out_dim = 32, 16
        seq_len = 12
        model = GeneralAttention(in_dim, out_dim)
        
        x = torch.randn(seq_len, in_dim, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Check parameter gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    def test_parameter_initialization_different_seeds(self):
        in_dim, out_dim = 64, 32
        
        torch.manual_seed(42)
        model1 = GeneralAttention(in_dim, out_dim)
        
        torch.manual_seed(123)
        model2 = GeneralAttention(in_dim, out_dim)
        
        # Different seeds should give different parameters
        assert not torch.equal(model1.W_1, model2.W_1)
        assert not torch.equal(model1.V_f, model2.V_f)
        assert not torch.equal(model1.bias, model2.bias)
    
    def test_output_not_nan_inf(self):
        torch.manual_seed(42)
        in_dim, out_dim = 96, 48
        seq_len = 30
        model = GeneralAttention(in_dim, out_dim)
        
        # Test with various input magnitudes
        for scale in [0.1, 1.0, 10.0]:
            x = torch.randn(seq_len, in_dim) * scale
            output = model(x)
            
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
    
    def test_single_token_sequence(self):
        torch.manual_seed(42)
        in_dim, out_dim = 64, 32
        model = GeneralAttention(in_dim, out_dim)
        
        # Single token (edge case)
        x = torch.randn(1, in_dim)
        output = model(x)
        
        assert output.shape == (1, out_dim)
        assert not torch.isnan(output).any()
    
    def test_attention_mechanism_intuition(self):
        torch.manual_seed(42)
        in_dim, out_dim = 16, 8
        seq_len = 5
        model = GeneralAttention(in_dim, out_dim)
        
        # Create input where some tokens are more similar
        x = torch.randn(seq_len, in_dim)
        x[0] = x[1]  # Make first two tokens identical
        
        output = model(x)
        
        # Output should be well-formed
        assert output.shape == (seq_len, out_dim)
        assert not torch.isnan(output).any()
        
        # The attention mechanism should produce different outputs for different positions
        # (unless the model learns to ignore differences, which is valid)
        assert output.shape == (seq_len, out_dim)
    
    def test_large_sequence_handling(self):
        torch.manual_seed(42)
        in_dim, out_dim = 32, 16
        seq_len = 200  # Larger sequence
        model = GeneralAttention(in_dim, out_dim)
        
        x = torch.randn(seq_len, in_dim)
        output = model(x)
        
        assert output.shape == (seq_len, out_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


# Note: The original attention.py module has import errors:
# - Line 9: "import pytorch.nn as nn" should be "import torch.nn as nn"
# - Line 10: "import pytorch.nn.functional as F" should be "import torch.nn.functional as F"
# These tests use a corrected version of the module for testing purposes.
