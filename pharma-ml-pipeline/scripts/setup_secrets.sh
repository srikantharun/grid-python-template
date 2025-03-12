#!/bin/bash
# Script to securely set up Kubernetes secrets

# Check for Azure CLI
if ! command -v az &> /dev/null; then
    echo "Error: Azure CLI is required but not found"
    exit 1
fi

# Check for kubectl
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is required but not found"
    exit 1
fi

# Get Azure KeyVault name from environment or input
KEYVAULT_NAME=${KEYVAULT_NAME:-""}
if [ -z "$KEYVAULT_NAME" ]; then
    read -p "Enter Azure KeyVault name: " KEYVAULT_NAME
fi

# Get Kubernetes namespace
NAMESPACE=${NAMESPACE:-"default"}

echo "Fetching secrets from Azure KeyVault $KEYVAULT_NAME..."

# Retrieve secrets from KeyVault
AZURE_STORAGE_CONNECTION_STRING=$(az keyvault secret show --name AZURE-STORAGE-CONNECTION-STRING --vault-name $KEYVAULT_NAME --query value -o tsv)
KAGGLE_USERNAME=$(az keyvault secret show --name KAGGLE-USERNAME --vault-name $KEYVAULT_NAME --query value -o tsv)
KAGGLE_KEY=$(az keyvault secret show --name KAGGLE-KEY --vault-name $KEYVAULT_NAME --query value -o tsv)
APPLICATIONINSIGHTS_CONNECTION_STRING=$(az keyvault secret show --name APPLICATIONINSIGHTS-CONNECTION-STRING --vault-name $KEYVAULT_NAME --query value -o tsv)

# Verify secrets were retrieved
if [ -z "$AZURE_STORAGE_CONNECTION_STRING" ] || [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
    echo "Error: Failed to retrieve required secrets from KeyVault"
    exit 1
fi

echo "Creating Kubernetes secrets..."

# Delete existing secret if it exists
kubectl delete secret azure-credentials --namespace $NAMESPACE --ignore-not-found

# Create Kubernetes secret
kubectl create secret generic azure-credentials \
    --namespace $NAMESPACE \
    --from-literal=AZURE_STORAGE_CONNECTION_STRING="$AZURE_STORAGE_CONNECTION_STRING" \
    --from-literal=KAGGLE_USERNAME="$KAGGLE_USERNAME" \
    --from-literal=KAGGLE_KEY="$KAGGLE_KEY" \
    --from-literal=APPLICATIONINSIGHTS_CONNECTION_STRING="$APPLICATIONINSIGHTS_CONNECTION_STRING"

echo "Secrets created successfully!"

# Create config map for logging configuration
kubectl create configmap pharma-ml-config \
    --namespace $NAMESPACE \
    --from-file=logging.yaml=config/logging.yaml \
    --from-file=model-config.json=config/model-config.json

echo "ConfigMap created successfully!"
