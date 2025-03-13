# Pharmaceutical Machine Learning Pipeline

A comprehensive machine learning pipeline for pharmaceutical research and drug discovery, built with TensorFlow, Streamlit, and MLflow, deployed on Azure Kubernetes Service (AKS) with GPU support.

## Overview

This pipeline enables pharmaceutical researchers to:

1. Train deep learning models for molecular property prediction
2. Track experiments and model versions using MLflow
3. Visualize and explore molecular data through a Streamlit interface
4. Deploy models to production in a scalable Kubernetes environment
5. Leverage GPU acceleration for faster training and inference

## Architecture

![Architecture Diagram](https://raw.githubusercontent.com/username/pharma-ml-pipeline/main/docs/images/architecture.png)

The pipeline consists of the following components:

- **Data Processing**: Tools for loading, preprocessing, and feature extraction from molecular data
- **Model Training**: Support for transformer-based NLP models and CNN models for molecular fingerprints
- **Experiment Tracking**: MLflow integration for tracking metrics, parameters, and artifacts
- **Web Interface**: Streamlit app for data exploration, model training, and predictions
- **Deployment**: Kubernetes manifests for deploying on AKS with GPU support

## Features

- **Multiple Model Types**:
  - Transformer-based models for text/sequence data
  - CNN models for molecular fingerprints and descriptors
  
- **Molecular Data Processing**:
  - SMILES to molecular fingerprints conversion
  - Molecular descriptor calculation
  - Data visualization and exploration
  
- **Experiment Management**:
  - Parameter tracking
  - Metric logging
  - Model versioning
  - Artifact storage
  
- **Interactive Interface**:
  - Dataset exploration
  - Model training
  - Performance visualization
  - Prediction interface

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose for local development
- Azure CLI for cloud deployment
- Kubernetes CLI (kubectl) for managing the AKS cluster
- NVIDIA GPU (for GPU acceleration)

### Local Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pharma-ml-pipeline.git
   cd pharma-ml-pipeline
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   export MLFLOW_TRACKING_URI=http://localhost:5000
   export AZURE_STORAGE_CONNECTION_STRING="your-connection-string"
   ```

4. Start the services with Docker Compose:
   ```bash
   docker-compose up -d
   ```

5. Access the Streamlit app at http://localhost:8501

### Cloud Deployment on AKS

1. Create an AKS cluster with GPU-enabled node pool:
   ```bash
   # Follow instructions in README.md in the root directory
   ```

2. Build and push the Docker image:
   ```bash
   az acr build --registry youracrname --image pharma-ml:latest .
   ```

3. Deploy to AKS:
   ```bash
   # Update $ACR_NAME in kubemanifest/deployment.yaml
   sed -i 's|$ACR_NAME|youracrname|g' kubemanifest/deployment.yaml
   
   # Apply manifests
   kubectl apply -f kubemanifest/pvc.yaml
   kubectl apply -f kubemanifest/secret.yaml
   kubectl apply -f kubemanifest/deployment.yaml
   kubectl apply -f kubemanifest/service.yaml
   ```

## Using the Pipeline

### Data Sources

The pipeline supports several data sources:

1. **Local CSV files**: Upload your molecular data directly
2. **Kaggle datasets**: Connect to Kaggle API to download datasets
3. **Sample data**: Built-in sample pharmaceutical data for testing

### Training Models

#### Transformer Models for Text/Sequence Data

For training transformer-based models on textual data:

```bash
./scripts/run_pipeline.sh -d your-dataset -m transformer -g 1
```

This command:
- Downloads the specified dataset from Kaggle
- Trains a transformer model using GPU acceleration
- Logs the experiment to MLflow
- Registers the model in the model registry

#### CNN Models for Molecular Fingerprints

For training CNN models on molecular fingerprints:

```bash
./scripts/run_pipeline.sh -d your-dataset -m cnn -g 1
```

### Model Registry

Models are automatically registered in MLflow's model registry and can be:
- Promoted to staging or production
- Downloaded for offline use
- Deployed to production endpoints

### Streamlit Interface

The Streamlit interface provides:

1. **Dashboard**: Overview of experiments and models
2. **Data Explorer**: Tools for visualizing and analyzing molecular data
3. **Model Training**: Interface for training new models
4. **Model Evaluation**: Visual performance metrics
5. **Prediction**: Interface for making predictions on new molecules

## Project Structure

```
pharma-ml-pipeline/
├── app/                       # Streamlit application
├── config/                    # Configuration files
├── data/                      # Data directory
├── docs/                      # Documentation
├── kubemanifest/              # Kubernetes manifests
├── MLFlow/                    # MLflow integration
├── monitoring/                # Monitoring configuration
├── notebooks/                 # Jupyter notebooks
├── scripts/                   # Utility scripts
├── src/                       # Source code
│   ├── data/                  # Data processing code
│   ├── evaluation/            # Model evaluation
│   ├── models/                # Model definitions
│   ├── training/              # Training code
│   └── utils/                 # Utilities
├── tests/                     # Unit tests
├── Dockerfile                 # Container definition
├── docker-compose.yml         # Local development setup
└── requirements.txt           # Python dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
