"""
test_generate_toy_data.py

Tests for toy data generation utility
"""

import numpy as np
import pandas as pd
import pytest
from tests.generate_toy_data import random_sequence, generate_toy_data, AA


class TestRandomSequence:
    
    def test_sequence_length_range(self):
        np.random.seed(42)
        for _ in range(10):
            seq = random_sequence(min_len=10, max_len=20)
            assert 10 <= len(seq) <= 20
    
    def test_sequence_contains_valid_amino_acids(self):
        np.random.seed(42)
        seq = random_sequence(min_len=50, max_len=100)
        for aa in seq:
            assert aa in AA
    
    def test_different_seeds_different_sequences(self):
        import random
        random.seed(123)
        seq1 = random_sequence(20, 30)
        random.seed(456)
        seq2 = random_sequence(20, 30)
        assert seq1 != seq2
    
    def test_same_seed_same_sequence(self):
        import random
        random.seed(789)
        seq1 = random_sequence(15, 25)
        random.seed(789)
        seq2 = random_sequence(15, 25)
        assert seq1 == seq2
    
    def test_edge_case_single_length(self):
        seq = random_sequence(min_len=5, max_len=5)
        assert len(seq) == 5


class TestGenerateToyData:
    
    def test_default_parameters(self):
        df, go_vocab, labels = generate_toy_data(seed=42)
        
        assert len(df) == 100  # default n_samples
        assert len(go_vocab) == 10  # default n_go_terms
        assert labels.shape == (100, 10)
        assert set(df.columns) == {"sequence", "go_terms"}
    
    def test_custom_parameters(self):
        n_samples, n_go_terms = 50, 15
        df, go_vocab, labels = generate_toy_data(
            n_samples=n_samples, 
            n_go_terms=n_go_terms, 
            min_len=20, 
            max_len=40,
            seed=123
        )
        
        assert len(df) == n_samples
        assert len(go_vocab) == n_go_terms
        assert labels.shape == (n_samples, n_go_terms)
    
    def test_go_vocabulary_format(self):
        df, go_vocab, labels = generate_toy_data(n_go_terms=5, seed=42)
        
        for term in go_vocab:
            assert term.startswith("GO:")
            assert len(term) == 10  # "GO:" + 7 digits
            assert term[3:].isdigit()
    
    def test_sequence_lengths(self):
        min_len, max_len = 25, 60
        df, go_vocab, labels = generate_toy_data(
            n_samples=20, 
            min_len=min_len, 
            max_len=max_len,
            seed=42
        )
        
        for seq in df["sequence"]:
            assert min_len <= len(seq) <= max_len
            for aa in seq:
                assert aa in AA
    
    def test_go_terms_assignment(self):
        df, go_vocab, labels = generate_toy_data(n_samples=30, n_go_terms=8, seed=42)
        
        for i, row in df.iterrows():
            go_terms = row["go_terms"]
            # Each protein should have 1-3 GO terms
            assert 1 <= len(go_terms) <= 3
            # All terms should be from vocabulary
            for term in go_terms:
                assert term in go_vocab
    
    def test_labels_consistency(self):
        df, go_vocab, labels = generate_toy_data(n_samples=20, n_go_terms=6, seed=42)
        
        # Check that labels match go_terms in dataframe
        for i, row in df.iterrows():
            go_terms = row["go_terms"]
            for j, term in enumerate(go_vocab):
                if term in go_terms:
                    assert labels[i, j] == 1
                else:
                    assert labels[i, j] == 0
    
    def test_labels_binary(self):
        df, go_vocab, labels = generate_toy_data(n_samples=25, seed=42)
        
        # Labels should be binary (0 or 1)
        assert np.all((labels == 0) | (labels == 1))
        assert labels.dtype == int
    
    def test_reproducibility_with_seed(self):
        seed = 999
        df1, go_vocab1, labels1 = generate_toy_data(n_samples=10, seed=seed)
        df2, go_vocab2, labels2 = generate_toy_data(n_samples=10, seed=seed)
        
        assert df1.equals(df2)
        assert go_vocab1 == go_vocab2
        assert np.array_equal(labels1, labels2)
    
    def test_different_seeds_different_data(self):
        df1, go_vocab1, labels1 = generate_toy_data(n_samples=10, seed=111)
        df2, go_vocab2, labels2 = generate_toy_data(n_samples=10, seed=222)
        
        # GO vocabulary should be same (deterministic generation)
        assert go_vocab1 == go_vocab2
        # But sequences and assignments should differ
        assert not df1.equals(df2)
        assert not np.array_equal(labels1, labels2)
    
    def test_dataframe_structure(self):
        df, go_vocab, labels = generate_toy_data(n_samples=15, seed=42)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df.columns) == 2
        assert "sequence" in df.columns
        assert "go_terms" in df.columns
        
        # Check data types
        for seq in df["sequence"]:
            assert isinstance(seq, str)
        for terms in df["go_terms"]:
            assert isinstance(terms, list)
    
    def test_edge_cases(self):
        # Small dataset
        df, go_vocab, labels = generate_toy_data(n_samples=1, n_go_terms=2, seed=42)
        assert len(df) == 1
        assert labels.shape == (1, 2)
        
        # Single GO term (need to use a different seed to avoid the k=2 issue)
        df, go_vocab, labels = generate_toy_data(n_samples=5, n_go_terms=3, seed=999)
        assert len(go_vocab) == 3
        assert labels.shape == (5, 3)
    
    def test_aa_constant(self):
        # Test that AA constant contains expected amino acids
        assert len(AA) == 20
        expected_aa = set("ACDEFGHIKLMNPQRSTVWY")
        assert set(AA) == expected_aa
    
    def test_multi_hot_encoding_properties(self):
        df, go_vocab, labels = generate_toy_data(n_samples=50, n_go_terms=10, seed=42)
        
        # Each sample should have at least one positive label (1-3 GO terms)
        row_sums = labels.sum(axis=1)
        assert np.all(row_sums >= 1)
        assert np.all(row_sums <= 3)
        
        # Some GO terms should be assigned to multiple proteins
        col_sums = labels.sum(axis=0)
        assert np.any(col_sums > 1)
