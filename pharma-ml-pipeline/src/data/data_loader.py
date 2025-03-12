import os
import pandas as pd
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KaggleDataLoader:
    """
    Utility to download and prepare datasets from Kaggle
    """
    def __init__(self, dataset_path="./data"):
        self.dataset_path = dataset_path
        self.api = KaggleApi()
        self.api.authenticate()  # Uses environment variables KAGGLE_USERNAME and KAGGLE_KEY
        
        # Create data directory if it doesn't exist
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
            
    def download_dataset(self, dataset_name, unzip=True):
        """
        Download a dataset from Kaggle
        
        Args:
            dataset_name: Full Kaggle dataset name (username/dataset-slug)
            unzip: Whether to extract the downloaded zip file
            
        Returns:
            Path to the downloaded/extracted dataset
        """
        logger.info(f"Downloading dataset: {dataset_name}")
        
        try:
            self.api.dataset_download_files(
                dataset_name, 
                path=self.dataset_path,
                unzip=unzip
            )
            
            if unzip:
                dataset_dir = os.path.join(self.dataset_path, dataset_name.split("/")[-1])
                return dataset_dir
            else:
                return os.path.join(self.dataset_path, f"{dataset_name.split('/')[-1]}.zip")
                
        except Exception as e:
            logger.error(f"Error downloading dataset {dataset_name}: {str(e)}")
            raise
    
    def load_csv_data(self, file_path, **kwargs):
        """
        Load a CSV file into a pandas DataFrame
        
        Args:
            file_path: Path to the CSV file
            **kwargs: Additional arguments to pass to pd.read_csv
            
        Returns:
            pandas DataFrame with the loaded data
        """
        try:
            df = pd.read_csv(file_path, **kwargs)
            logger.info(f"Loaded CSV data from {file_path}: {df.shape} shape")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV data from {file_path}: {str(e)}")
            raise
            
    def load_molecular_data(self, file_path, format='csv'):
        """
        Load molecular data from various file formats 
        (specialized for pharmaceutical datasets)
        
        Args:
            file_path: Path to the molecular data file
            format: File format (csv, sdf, mol, etc.)
            
        Returns:
            Appropriate data structure for the molecular data
        """
        # This is a placeholder - would use RDKit or similar libraries
        # to properly load molecular data based on format
        
        if format == 'csv':
            return self.load_csv_data(file_path)
        else:
            logger.error(f"Format {format} not yet implemented for molecular data")
            raise NotImplementedError(f"Format {format} not yet implemented")
