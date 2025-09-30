"""
test_mock_esm.py

Tests for MockESM embedding module
"""

import torch
import torch.nn as nn
import pytest
from modules.mock_esm import MockESM, N_AA


class TestMockESM:
    
    def test_init_parameters(self):
        torch.manual_seed(42)
        seq_len, hidden_len, embed_len, proj_len = 100, 512, 256, 128
        model = MockESM(seq_len, hidden_len, embed_len, proj_len)
        
        assert model.embed.num_embeddings == N_AA
        assert model.embed.embedding_dim == embed_len
        assert model.layer1.in_features == embed_len
        assert model.layer1.out_features == hidden_len
        assert model.layer2.in_features == hidden_len
        assert model.layer2.out_features == proj_len
    
    def test_n_aa_constant(self):
        assert N_AA == 21
    
    def test_forward_shape(self):
        torch.manual_seed(42)
        seq_len, hidden_len, embed_len, proj_len = 50, 256, 128, 64
        batch_size = 4
        model = MockESM(seq_len, hidden_len, embed_len, proj_len)
        
        # Valid amino acid indices (0 to N_AA-1)
        seqs = torch.randint(0, N_AA, (batch_size, seq_len))
        output = model(seqs)
        
        assert output.shape == (batch_size, seq_len, proj_len)
    
    def test_forward_different_batch_sizes(self):
        torch.manual_seed(42)
        seq_len, hidden_len, embed_len, proj_len = 30, 128, 64, 32
        model = MockESM(seq_len, hidden_len, embed_len, proj_len)
        
        for batch_size in [1, 8, 16]:
            seqs = torch.randint(0, N_AA, (batch_size, seq_len))
            output = model(seqs)
            assert output.shape == (batch_size, seq_len, proj_len)
    
    def test_forward_different_seq_lengths(self):
        torch.manual_seed(42)
        seq_len, hidden_len, embed_len, proj_len = 100, 256, 128, 64
        model = MockESM(seq_len, hidden_len, embed_len, proj_len)
        
        for actual_len in [10, 25, 75]:
            seqs = torch.randint(0, N_AA, (2, actual_len))
            output = model(seqs)
            assert output.shape == (2, actual_len, proj_len)
    
    def test_embedding_layer(self):
        torch.manual_seed(42)
        model = MockESM(50, 128, 64, 32)
        
        # Test embedding lookup
        seqs = torch.tensor([[0, 1, 20], [10, 15, 5]])  # Valid AA indices
        embeddings = model.embed(seqs)
        assert embeddings.shape == (2, 3, 64)
        
        # Different indices should give different embeddings
        assert not torch.equal(embeddings[0, 0], embeddings[0, 1])
    
    def test_mlp_layers(self):
        torch.manual_seed(42)
        embed_len, hidden_len, proj_len = 64, 128, 32
        model = MockESM(50, hidden_len, embed_len, proj_len)
        
        # Test MLP portion directly
        x = torch.randn(2, 10, embed_len)
        h1 = model.layer1(x)
        assert h1.shape == (2, 10, hidden_len)
        
        h1_relu = model.relu(h1)
        assert h1_relu.shape == (2, 10, hidden_len)
        assert (h1_relu >= 0).all()  # ReLU non-negativity
        
        output = model.layer2(h1_relu)
        assert output.shape == (2, 10, proj_len)
    
    def test_relu_activation(self):
        torch.manual_seed(42)
        model = MockESM(20, 64, 32, 16)
        seqs = torch.randint(0, N_AA, (3, 20))
        
        # Hook to capture intermediate activations
        activations = {}
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output
            return hook
        
        model.relu.register_forward_hook(hook_fn('relu'))
        output = model(seqs)
        
        # Check ReLU output is non-negative
        assert (activations['relu'] >= 0).all()
    
    def test_deterministic_output(self):
        seq_len, hidden_len, embed_len, proj_len = 25, 128, 64, 32
        seqs = torch.randint(0, N_AA, (2, seq_len))
        
        torch.manual_seed(123)
        model1 = MockESM(seq_len, hidden_len, embed_len, proj_len)
        output1 = model1(seqs)
        
        torch.manual_seed(123)
        model2 = MockESM(seq_len, hidden_len, embed_len, proj_len)
        output2 = model2(seqs)
        
        assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_gradient_flow(self):
        torch.manual_seed(42)
        model = MockESM(30, 128, 64, 32)
        seqs = torch.randint(0, N_AA, (2, 30))
        
        output = model(seqs)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist for all parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    def test_invalid_amino_acid_indices(self):
        torch.manual_seed(42)
        model = MockESM(20, 64, 32, 16)
        
        # Test with out-of-range indices
        invalid_seqs = torch.tensor([[N_AA, 0, 1]])  # N_AA is out of range
        with pytest.raises(IndexError):
            model(invalid_seqs)
        
        negative_seqs = torch.tensor([[-1, 0, 1]])  # Negative index
        with pytest.raises(IndexError):
            model(negative_seqs)
    
    def test_empty_sequence(self):
        torch.manual_seed(42)
        model = MockESM(0, 64, 32, 16)
        
        # Empty sequence
        seqs = torch.empty(2, 0, dtype=torch.long)
        output = model(seqs)
        assert output.shape == (2, 0, 16)
    
    def test_parameter_initialization(self):
        torch.manual_seed(42)
        model1 = MockESM(20, 64, 32, 16)
        torch.manual_seed(42)
        model2 = MockESM(20, 64, 32, 16)
        
        # Same seed should give same initialization
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            assert name1 == name2
            assert torch.equal(param1, param2)
    
    def test_output_range_reasonable(self):
        torch.manual_seed(42)
        model = MockESM(50, 128, 64, 32)
        seqs = torch.randint(0, N_AA, (4, 50))
        
        output = model(seqs)
        
        # Output should not be extreme values
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert output.abs().max() < 100  # Reasonable range
        assert output.abs().mean() > 0.01  # Not all zeros
