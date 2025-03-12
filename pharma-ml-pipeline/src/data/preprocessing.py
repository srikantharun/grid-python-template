import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import AllChem

class MolecularFeatureExtractor:
    """
    Extract features from molecular data for machine learning
    """
    def __init__(self):
        self.feature_names = []
        
    def calculate_fingerprints(self, smiles_list, fp_type='morgan', radius=2, n_bits=2048):
        """
        Calculate molecular fingerprints from SMILES strings
        
        Args:
            smiles_list: List of SMILES strings
            fp_type: Type of fingerprint to calculate (morgan, rdkit, maccs, etc.)
            radius: Radius for Morgan fingerprints
            n_bits: Number of bits in the fingerprint
            
        Returns:
            Numpy array of fingerprints
        """
        fingerprints = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                # Handle invalid SMILES
                fingerprints.append(np.zeros(n_bits))
                continue
                
            if fp_type == 'morgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            elif fp_type == 'rdkit':
                fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
            elif fp_type == 'maccs':
                fp = AllChem.GetMACCSKeysFingerprint(mol)
                
            fingerprints.append(np.array(fp))
            
        return np.array(fingerprints)
    
    def calculate_descriptors(self, smiles_list):
        """
        Calculate molecular descriptors from SMILES strings
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Pandas DataFrame of descriptors
        """
        from rdkit.ML.Descriptors import MolecularDescriptorCalculator
        from rdkit.Chem import Descriptors
        
        # Get all descriptor names
        descriptor_names = [x[0] for x in Descriptors._descList]
        self.feature_names = descriptor_names
        
        # Initialize calculator
        calc = MolecularDescriptorCalculator(descriptor_names)
        
        # Calculate descriptors for each molecule
        descriptors = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                # Handle invalid SMILES
                descriptors.append([np.nan] * len(descriptor_names))
                continue
                
            # Calculate descriptors and add to list
            descriptors.append(calc.CalcDescriptors(mol))
            
        return pd.DataFrame(descriptors, columns=descriptor_names)
    
    def prepare_text_for_transformers(self, text_list, tokenizer, max_length=512):
        """
        Prepare text data for transformer models
        
        Args:
            text_list: List of text strings
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            
        Returns:
            Dictionary of tokenized inputs
        """
        return tokenizer(
            text_list,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='tf'
        )

class PharmaDataPreprocessor:
    """
    Preprocessing pipeline for pharmaceutical data
    """
    def __init__(self):
        self.scaler = None
        self.feature_extractor = MolecularFeatureExtractor()
        
    def prepare_data(self, df, smiles_col=None, target_col=None, 
                    text_col=None, tokenizer=None, test_size=0.2):
        """
        Prepare pharmaceutical data for model training
        
        Args:
            df: Input DataFrame
            smiles_col: Column containing SMILES strings (optional)
            target_col: Column containing target values
            text_col: Column containing text data (optional)
            tokenizer: Tokenizer for text data (optional)
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary containing train and test data
        """
        X = df.drop(columns=[target_col] if target_col in df.columns else [])
        
        # Process target data if available
        if target_col and target_col in df.columns:
            y = df[target_col].values
        else:
            y = None
            
        # Process SMILES data if available
        if smiles_col and smiles_col in df.columns:
            # Calculate fingerprints
            smiles_list = df[smiles_col].values
            fingerprints = self.feature_extractor.calculate_fingerprints(smiles_list)
            
            # Calculate descriptors
            descriptors_df = self.feature_extractor.calculate_descriptors(smiles_list)
            
            # Handle NaN values
            descriptors_df = descriptors_df.fillna(0)
            
            # Scale descriptors
            self.scaler = StandardScaler()
            scaled_descriptors = self.scaler.fit_transform(descriptors_df)
            
            # Combine fingerprints and descriptors
            X = np.hstack([fingerprints, scaled_descriptors])
            
        # Process text data if available
        if text_col and text_col in df.columns and tokenizer:
            text_list = df[text_col].values
            text_features = self.feature_extractor.prepare_text_for_transformers(
                text_list, tokenizer
            )
            
        # Split data
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) < 10 else None
            )
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
        else:
            X_train, X_test = train_test_split(X, test_size=test_size, random_state=42)
            
            return {
                'X_train': X_train,
                'X_test': X_test
            }

