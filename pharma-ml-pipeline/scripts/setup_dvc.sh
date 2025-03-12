#!/bin/bash

# Install DVC
pip install dvc[azure]

# Initialize DVC in the repository
dvc init

# Add Azure remote
dvc remote add -d azure azure://mlflow/dvc

# Add data directory to DVC
dvc add data/raw
dvc add data/processed

# Configure .gitignore
echo "# DVC tracked directories" >> .gitignore
echo "data/raw/**" >> .gitignore
echo "data/processed/**" >> .gitignore
echo "!data/raw/.gitkeep" >> .gitignore
echo "!data/processed/.gitkeep" >> .gitignore

# Create folders
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/interim
mkdir -p data/external

# Add placeholder files
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/interim/.gitkeep
touch data/external/.gitkeep

# Push to remote
dvc push

echo "DVC setup complete!"
