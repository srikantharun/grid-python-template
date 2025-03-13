#!/bin/bash
# This script cleans up Azure resources while preserving the ACR

set -e  # Exit immediately if a command exits with a non-zero status

echo "Starting cleanup of Azure resources (preserving ACR)..."

# Variables - update these to match your environment
RESOURCE_GROUP="aks-demo-rg"
CLUSTER_NAME="private-aks-cluster"
ACR_NAME="pharmamlacr1308"
KEYVAULT_NAME="pharma-keyvault"
VNET_NAME="aks-vnet"
IDENTITY_NAME="aks-identity"

# 1. Delete the AKS cluster
echo "Deleting AKS cluster $CLUSTER_NAME..."
az aks delete --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME --yes

# 2. Delete the KeyVault
echo "Deleting KeyVault $KEYVAULT_NAME..."
az keyvault delete --resource-group $RESOURCE_GROUP --name $KEYVAULT_NAME

# 3. Delete the Managed Identity
echo "Deleting Managed Identity $IDENTITY_NAME..."
az identity delete --resource-group $RESOURCE_GROUP --name $IDENTITY_NAME

# 4. Delete the Virtual Network
echo "Deleting Virtual Network $VNET_NAME..."
az network vnet delete --resource-group $RESOURCE_GROUP --name $VNET_NAME

# 5. Verify ACR still exists
echo "Verifying ACR $ACR_NAME is preserved..."
az acr show --resource-group $RESOURCE_GROUP --name $ACR_NAME

echo "Cleanup completed. The following resources have been deleted:"
echo "- AKS Cluster: $CLUSTER_NAME"
echo "- Key Vault: $KEYVAULT_NAME" 
echo "- Virtual Network: $VNET_NAME"
echo "- Managed Identity: $IDENTITY_NAME"
echo ""
echo "The ACR $ACR_NAME has been preserved."
echo ""
echo "To verify remaining resources:"
echo "az resource list --resource-group $RESOURCE_GROUP -o table"
