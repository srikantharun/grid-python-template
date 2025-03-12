import pytest
import numpy as np
import tensorflow as tf
from src.models.transformer_model import PharmaTransformerModel
from src.models.cnn_model import MolecularCNNModel
from transformers import AutoTokenizer

class TestTransformerModel:
    """Test cases for PharmaTransformerModel"""
    
    def test_model_initialization(self):
        """Test that model initializes correctly"""
        model = PharmaTransformerModel(
            pretrained_model="distilbert-base-uncased",
            num_classes=2
        )
        
        assert model is not None
        assert model.model is not None
        assert model.tokenizer is not None
        
    def test_model_compilation(self):
        """Test model compilation"""
        model = PharmaTransformerModel(
            pretrained_model="distilbert-base-uncased",
            num_classes=2
        )
        
        model.compile_model(learning_rate=1e-4)
        
        assert model.model.optimizer is not None
        assert model.model.loss is not None
        
    def test_binary_classification(self):
        """Test binary classification prediction shape"""
        model = PharmaTransformerModel(
            pretrained_model="distilbert-base-uncased",
            num_classes=2,
            max_length=8  # Small for testing
        )
        
        # Create dummy input
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        inputs = tokenizer(
            ["This is a test", "Another test"],
            padding="max_length",
            truncation=True,
            max_length=8,
            return_tensors="tf"
        )
        
        # Compile model
        model.compile_model()
        
        # Get predictions
        preds = model.predict(inputs)
        
        # Check shape for binary classification (should be (2,))
        assert preds.shape[0] == 2
        assert len(preds.shape) <= 2
        
    def test_multi_classification(self):
        """Test multi-class classification prediction shape"""
        model = PharmaTransformerModel(
            pretrained_model="distilbert-base-uncased",
            num_classes=3,
            max_length=8  # Small for testing
        )
        
        # Create dummy input
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        inputs = tokenizer(
            ["This is a test", "Another test"],
            padding="max_length",
            truncation=True,
            max_length=8,
            return_tensors="tf"
        )
        
        # Compile model
        model.compile_model()
        
        # Get predictions
        preds = model.predict(inputs)
        
        # Check shape for multi-class classification (should be (2, 3))
        assert preds.shape == (2, 3)
        
class TestCNNModel:
    """Test cases for MolecularCNNModel"""
    
    def test_model_initialization(self):
        """Test that model initializes correctly"""
        model = MolecularCNNModel(
            input_shape=(2048,),
            num_classes=2
        )
        
        assert model is not None
        assert model.model is not None
        
    def test_model_compilation(self):
        """Test model compilation"""
        model = MolecularCNNModel(
            input_shape=(2048,),
            num_classes=2
        )
        
        model.compile_model(learning_rate=1e-4)
        
        assert model.model.optimizer is not None
        assert model.model.loss is not None
        
    def test_prediction_shape(self):
        """Test prediction shape"""
        # Create model
        model = MolecularCNNModel(
            input_shape=(2048,),
            num_classes=2
        )
        
        # Compile model
        model.compile_model()
        
        # Create dummy input
        X = np.random.random((10, 2048))
        
        # Get predictions
        preds = model.predict(X)
        
        # Check shape for binary classification
        assert preds.shape[0] == 10
        
    def test_fingerprint_conversion(self):
        """Test SMILES to fingerprint conversion"""
        model = MolecularCNNModel(input_shape=(2048,))
        
        # Test with valid SMILES
        smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O", "CCC1=CC=CC=C1"]
        fps = model.convert_smiles_to_fingerprints(smiles)
        
        # Check shape
        assert fps.shape == (2, 2048)
        
        # Test with invalid SMILES
        smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O", "Invalid SMILES"]
        fps = model.convert_smiles_to_fingerprints(smiles)
        
        # Check shape
        assert fps.shape == (2, 2048)
        # Check that second fingerprint is all zeros
        assert np.sum(fps[1]) == 0
