"""
test_seq_gating.py

Tests for SeqFuseGatePool sequence gating mechanism
"""

import torch
import torch.nn as nn
import pytest
from modules.seq_gating import SeqFuseGatePool


class TestSeqFuseGatePool:
    
    def test_init_default_params(self):
        torch.manual_seed(42)
        K = 128
        model = SeqFuseGatePool(K)
        
        assert model.proj_esm.in_features == 1280  # d_esm default
        assert model.proj_esm.out_features == K
        assert model.proj_dcc.in_features == 256   # c_dcc default
        assert model.proj_dcc.out_features == K
        assert model.gate_tok.in_features == 2 * K
        assert model.gate_tok.out_features == K
        assert model.dropout.p == 0.1
    
    def test_init_custom_params(self):
        torch.manual_seed(42)
        K, d_esm, c_dcc = 64, 512, 128
        model = SeqFuseGatePool(K, d_esm=d_esm, c_dcc=c_dcc, attn_hidden=256, dropout=0.2)
        
        assert model.proj_esm.in_features == d_esm
        assert model.proj_esm.out_features == K
        assert model.proj_dcc.in_features == c_dcc
        assert model.proj_dcc.out_features == K
        assert model.dropout.p == 0.2
    
    def test_forward_shape(self):
        torch.manual_seed(42)
        B, L, K = 4, 50, 128
        d_esm, c_dcc = 1280, 256
        
        model = SeqFuseGatePool(K, d_esm=d_esm, c_dcc=c_dcc)
        
        H_esm = torch.randn(B, L, d_esm)
        H_dcc = torch.randn(B, L, c_dcc)
        lengths = torch.randint(10, L+1, (B,))
        
        output = model(H_esm, H_dcc, lengths)
        assert output.shape == (B, K)
    
    def test_forward_different_batch_sizes(self):
        torch.manual_seed(42)
        K = 64
        L, d_esm, c_dcc = 30, 512, 128
        model = SeqFuseGatePool(K, d_esm=d_esm, c_dcc=c_dcc)
        
        for B in [1, 8, 16]:
            H_esm = torch.randn(B, L, d_esm)
            H_dcc = torch.randn(B, L, c_dcc)
            lengths = torch.randint(5, L+1, (B,))
            
            output = model(H_esm, H_dcc, lengths)
            assert output.shape == (B, K)
    
    def test_forward_different_seq_lengths(self):
        torch.manual_seed(42)
        K = 96
        B, d_esm, c_dcc = 3, 768, 192
        model = SeqFuseGatePool(K, d_esm=d_esm, c_dcc=c_dcc)
        
        for L in [20, 100, 200]:
            H_esm = torch.randn(B, L, d_esm)
            H_dcc = torch.randn(B, L, c_dcc)
            lengths = torch.randint(L//2, L+1, (B,))
            
            output = model(H_esm, H_dcc, lengths)
            assert output.shape == (B, K)
    
    def test_masking_mechanism(self):
        torch.manual_seed(42)
        K = 32
        B, L = 2, 10
        model = SeqFuseGatePool(K)
        
        H_esm = torch.randn(B, L, 1280)
        H_dcc = torch.randn(B, L, 256)
        lengths = torch.tensor([5, 8])  # Different actual lengths
        
        output = model(H_esm, H_dcc, lengths)
        assert output.shape == (B, K)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_gating_behavior(self):
        torch.manual_seed(42)
        K = 64
        B, L = 2, 20
        model = SeqFuseGatePool(K)
        
        H_esm = torch.randn(B, L, 1280)
        H_dcc = torch.randn(B, L, 256)
        lengths = torch.randint(10, L+1, (B,))
        
        # Forward pass to test gating
        with torch.no_grad():
            E = model.proj_esm(H_esm)
            D = model.proj_dcc(H_dcc)
            
            # Gate should be between 0 and 1
            gate_input = torch.cat([E, D], dim=-1)
            G = torch.sigmoid(model.gate_tok(gate_input))
            assert (G >= 0).all() and (G <= 1).all()
    
    def test_attention_pooling(self):
        torch.manual_seed(42)
        K = 48
        B, L = 3, 25
        model = SeqFuseGatePool(K)
        
        H_esm = torch.randn(B, L, 1280)
        H_dcc = torch.randn(B, L, 256)
        lengths = torch.tensor([15, 20, 25])
        
        output = model(H_esm, H_dcc, lengths)
        
        # Output should be normalized (layer norm)
        assert output.shape == (B, K)
        # Check that different samples can have different outputs
        assert not torch.equal(output[0], output[1])
    
    def test_zero_length_handling(self):
        torch.manual_seed(42)
        K = 32
        B, L = 2, 10
        model = SeqFuseGatePool(K)
        
        H_esm = torch.randn(B, L, 1280)
        H_dcc = torch.randn(B, L, 256)
        lengths = torch.tensor([0, 5])  # One sequence has zero length
        
        output = model(H_esm, H_dcc, lengths)
        assert output.shape == (B, K)
        # Should handle zero length gracefully (nan_to_num should help)
        assert not torch.isnan(output).any()
    
    def test_full_length_sequences(self):
        torch.manual_seed(42)
        K = 64
        B, L = 2, 15
        model = SeqFuseGatePool(K)
        
        H_esm = torch.randn(B, L, 1280)
        H_dcc = torch.randn(B, L, 256)
        lengths = torch.tensor([L, L])  # Full length sequences
        
        output = model(H_esm, H_dcc, lengths)
        assert output.shape == (B, K)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_deterministic_output(self):
        K = 96
        B, L = 3, 30
        H_esm = torch.randn(B, L, 1280)
        H_dcc = torch.randn(B, L, 256)
        lengths = torch.randint(10, L+1, (B,))
        
        torch.manual_seed(123)
        model1 = SeqFuseGatePool(K)
        output1 = model1(H_esm, H_dcc, lengths)
        
        torch.manual_seed(123)
        model2 = SeqFuseGatePool(K)
        output2 = model2(H_esm, H_dcc, lengths)
        
        assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_gradient_flow(self):
        torch.manual_seed(42)
        K = 64
        B, L = 2, 20
        model = SeqFuseGatePool(K)
        
        H_esm = torch.randn(B, L, 1280, requires_grad=True)
        H_dcc = torch.randn(B, L, 256, requires_grad=True)
        lengths = torch.randint(10, L+1, (B,))
        
        output = model(H_esm, H_dcc, lengths)
        loss = output.sum()
        loss.backward()
        
        assert H_esm.grad is not None
        assert H_dcc.grad is not None
        assert not torch.isnan(H_esm.grad).any()
        assert not torch.isnan(H_dcc.grad).any()
        
        # Check model parameter gradients
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
    
    def test_dropout_behavior(self):
        torch.manual_seed(42)
        K = 32
        B, L = 2, 15
        model = SeqFuseGatePool(K, dropout=0.5)
        
        H_esm = torch.randn(B, L, 1280)
        H_dcc = torch.randn(B, L, 256)
        lengths = torch.randint(5, L+1, (B,))
        
        model.train()
        train_output = model(H_esm, H_dcc, lengths)
        
        model.eval()
        eval_output = model(H_esm, H_dcc, lengths)
        
        # Outputs should be different due to dropout
        assert not torch.equal(train_output, eval_output)
    
    def test_layer_norm_properties(self):
        torch.manual_seed(42)
        K = 64
        B, L = 4, 25
        model = SeqFuseGatePool(K)
        
        H_esm = torch.randn(B, L, 1280)
        H_dcc = torch.randn(B, L, 256)
        lengths = torch.randint(10, L+1, (B,))
        
        output = model(H_esm, H_dcc, lengths)
        
        # Layer norm should center and scale the output
        assert output.shape == (B, K)
        # Check approximate normalization (mean close to 0, std close to 1)
        for i in range(B):
            sample_mean = output[i].mean()
            sample_std = output[i].std()
            assert abs(sample_mean) < 1e-5  # Should be close to 0
            assert abs(sample_std - 1.0) < 1e-5  # Should be close to 1
    
    def test_scorer_network(self):
        torch.manual_seed(42)
        K = 48
        attn_hidden = 96
        model = SeqFuseGatePool(K, attn_hidden=attn_hidden)
        
        # Test scorer network structure
        assert len(model.scorer) == 3  # Linear -> ReLU -> Linear
        assert isinstance(model.scorer[0], nn.Linear)
        assert isinstance(model.scorer[1], nn.ReLU)
        assert isinstance(model.scorer[2], nn.Linear)
        
        assert model.scorer[0].in_features == K
        assert model.scorer[0].out_features == attn_hidden
        assert model.scorer[2].in_features == attn_hidden
        assert model.scorer[2].out_features == 1
    
    def test_invalid_input_shapes(self):
        torch.manual_seed(42)
        K = 64
        model = SeqFuseGatePool(K)
        
        # Mismatched sequence lengths
        H_esm = torch.randn(2, 20, 1280)
        H_dcc = torch.randn(2, 15, 256)  # Different L
        lengths = torch.tensor([10, 8])
        
        with pytest.raises((RuntimeError, ValueError)):
            model(H_esm, H_dcc, lengths)
        
        # Wrong embedding dimensions
        H_esm_wrong = torch.randn(2, 20, 512)  # Wrong d_esm
        H_dcc_correct = torch.randn(2, 20, 256)
        
        with pytest.raises((RuntimeError, ValueError)):
            model(H_esm_wrong, H_dcc_correct, lengths)
