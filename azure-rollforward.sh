#!/bin/bash
# This script rolls forward the Azure resources for the Pharma ML Pipeline
# with all the corrections we've made during our troubleshooting

set -e  # Exit immediately if a command exits with a non-zero status

echo "Starting roll-forward of Azure resources for Pharma ML Pipeline..."

# Variables - update these to match your environment
RESOURCE_GROUP="aks-demo-rg"
CLUSTER_NAME="private-aks-cluster"
ACR_NAME="pharmamlacr1308"
KEYVAULT_NAME="pharma-keyvault"
GPU_VM_SIZE="Standard_NV4as_v4"  # Update this to your preferred GPU VM size
VNET_SUBNET_ID="/subscriptions/f008079f-ca49-4f7b-b884-39035ba657e0/resourceGroups/aks-demo-rg/providers/Microsoft.Network/virtualNetworks/aks-vnet/subnets/aks-subnet"

# 1. Connect to the AKS cluster
echo "Getting AKS credentials..."
az aks get-credentials --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME --overwrite-existing

# 2. Create or scale GPU node pool if needed
NODE_POOL_EXISTS=$(az aks nodepool list --resource-group $RESOURCE_GROUP --cluster-name $CLUSTER_NAME --query "[?name=='gpunodepool']" -o tsv)

if [ -z "$NODE_POOL_EXISTS" ]; then
    echo "Creating GPU node pool..."
    az aks nodepool add \
        --resource-group $RESOURCE_GROUP \
        --cluster-name $CLUSTER_NAME \
        --name gpunodepool \
        --node-count 1 \
        --node-vm-size $GPU_VM_SIZE \
        --vnet-subnet-id "$VNET_SUBNET_ID" \
        --no-wait
    
    echo "GPU node pool creation started. This may take several minutes..."
    echo "You can check status with: az aks nodepool show --resource-group $RESOURCE_GROUP --cluster-name $CLUSTER_NAME --name gpunodepool"
else
    echo "GPU node pool exists. Scaling to 1 node..."
    az aks nodepool update \
        --resource-group $RESOURCE_GROUP \
        --cluster-name $CLUSTER_NAME \
        --name gpunodepool \
        --node-count 1 \
        --no-wait
fi

echo "Waiting for GPU node pool to be ready (this may take a few minutes)..."
# Keep checking status until node pool is ready or 10 minutes elapsed
timeout=600
start_time=$(date +%s)
while true; do
    status=$(az aks nodepool show \
        --resource-group $RESOURCE_GROUP \
        --cluster-name $CLUSTER_NAME \
        --name gpunodepool \
        --query provisioningState -o tsv 2>/dev/null || echo "NotFound")
    
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    
    if [ "$status" == "Succeeded" ]; then
        echo "GPU node pool is ready."
        break
    elif [ $elapsed -ge $timeout ]; then
        echo "Timeout waiting for GPU node pool to be ready."
        echo "Please check status manually with: az aks nodepool show --resource-group $RESOURCE_GROUP --cluster-name $CLUSTER_NAME --name gpunodepool"
        break
    else
        echo "Current status: $status. Waiting... ($elapsed seconds elapsed)"
        sleep 30
    fi
done

# 3. Install NVIDIA Device Plugin
echo "Installing NVIDIA Device Plugin..."
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update
helm upgrade --install \
    --namespace kube-system \
    nvidia-device-plugin \
    nvdp/nvidia-device-plugin

# 4. Create Kubernetes namespace for MLflow if it doesn't exist
kubectl create namespace mlflow --dry-run=client -o yaml | kubectl apply -f -

# 5. Create Kubernetes secret for credentials
echo "Creating Kubernetes secret for credentials..."
# Replace these placeholder values with your actual credentials
AZURE_STORAGE_CONNECTION_STRING="your-connection-string"
KAGGLE_USERNAME="your-kaggle-username"
KAGGLE_KEY="your-kaggle-api-key"
APPINSIGHTS_CONNECTION_STRING="your-app-insights-connection-string"

kubectl create secret generic azure-credentials \
    --from-literal=AZURE_STORAGE_CONNECTION_STRING="$AZURE_STORAGE_CONNECTION_STRING" \
    --from-literal=KAGGLE_USERNAME="$KAGGLE_USERNAME" \
    --from-literal=KAGGLE_KEY="$KAGGLE_KEY" \
    --from-literal=APPLICATIONINSIGHTS_CONNECTION_STRING="$APPINSIGHTS_CONNECTION_STRING" \
    --dry-run=client -o yaml | kubectl apply -f -

# 6. Create ConfigMap for logging configuration
echo "Creating ConfigMap for logging configuration..."
kubectl create configmap pharma-ml-config \
    --from-file=logging.yaml=pharma-ml-pipeline/config/logging.yaml \
    --dry-run=client -o yaml | kubectl apply -f -

# 7. Create Persistent Volume Claims
echo "Creating Persistent Volume Claims..."
kubectl apply -f pharma-ml-pipeline/kubemanifest/pvc.yaml

# 8. Build and push Docker image to ACR
echo "Building and pushing Docker image to ACR..."
cd pharma-ml-pipeline

# Fix requirements.txt and other issues we encountered
echo "Fixing requirements.txt..."
sed -i 's/tensorflow-gpu==2.15.0/tensorflow-gpu==2.12.0/g' requirements.txt
sed -i 's/streamlit-ketcher==0.1.0/streamlit-ketcher==0.0.1/g' requirements.txt
sed -i 's/azure-storage-blob==12.18.3/azure-storage-blob<=12.13.0,>=12.5.0/g' requirements.txt

# Fix the missing scripts directory and entrypoint.sh
echo "Setting up entrypoint script..."
mkdir -p scripts
cp entrypoint/entrypoint.sh scripts/ || echo "Creating entrypoint script from scratch..."

if [ ! -f "scripts/entrypoint.sh" ]; then
    cat > scripts/entrypoint.sh << 'EOF'
#!/bin/bash
set -e

# This script serves as the container entrypoint and handles different execution modes

# Check if we're running as root
if [ "$(id -u)" = "0" ]; then
  echo "Warning: Running as root is not recommended. Consider using a non-root user."
fi

# Create log directory if it doesn't exist
mkdir -p /app/logs

# Set up environment based on execution mode
MODE=${1:-"streamlit"}

case $MODE in
  "streamlit")
    echo "Starting Streamlit server..."
    exec streamlit run /app/app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
    ;;
    
  "train")
    MODEL_TYPE=${2:-"transformer"}
    echo "Starting model training for type: $MODEL_TYPE"
    
    if [ "$MODEL_TYPE" = "transformer" ]; then
      DATASET=${3:-"sample"}
      TEXT_COL=${4:-"Description"}
      TARGET_COL=${5:-"Activity"}
      
      exec python -m src.training.run_training --model_type=transformer \
        --dataset=$DATASET --text_col=$TEXT_COL --target_col=$TARGET_COL
    elif [ "$MODEL_TYPE" = "cnn" ]; then
      DATASET=${3:-"sample"}
      SMILES_COL=${4:-"SMILES"}
      TARGET_COL=${5:-"Activity"}
      
      exec python -m src.training.run_training --model_type=cnn \
        --dataset=$DATASET --smiles_col=$SMILES_COL --target_col=$TARGET_COL
    else
      echo "Unknown model type: $MODEL_TYPE"
      exit 1
    fi
    ;;
    
  "evaluate")
    MODEL_PATH=${2:-"/app/models/latest"}
    echo "Evaluating model at: $MODEL_PATH"
    
    exec python -m src.evaluation.run_evaluation --model_path=$MODEL_PATH
    ;;
    
  "predict")
    MODEL_PATH=${2:-"/app/models/latest"}
    INPUT_PATH=${3:-"/app/data/input.csv"}
    OUTPUT_PATH=${4:-"/app/data/predictions.csv"}
    
    echo "Running prediction with model: $MODEL_PATH"
    
    exec python -m src.prediction.run_prediction \
      --model_path=$MODEL_PATH --input_path=$INPUT_PATH --output_path=$OUTPUT_PATH
    ;;
    
  "shell")
    echo "Starting interactive shell..."
    exec /bin/bash
    ;;
    
  *)
    echo "Unknown mode: $MODE"
    echo "Available modes: streamlit, train, evaluate, predict, shell"
    exit 1
    ;;
esac
EOF
    chmod +x scripts/entrypoint.sh
fi

# Fix the Dockerfile to use pip for RDKit
echo "Fixing Dockerfile..."
sed -i 's/RUN conda install -c conda-forge rdkit -y/RUN pip install rdkit/g' Dockerfile

# Build the image using Azure ACR Tasks
echo "Building image with ACR Tasks..."
cd ..  # Go back to parent directory
az acr build --registry $ACR_NAME --image pharma-ml:latest --timeout 7200 ./pharma-ml-pipeline

# 9. Update deployment YAML and deploy
echo "Updating deployment YAML..."
sed -i "s|\$ACR_NAME|$ACR_NAME|g" pharma-ml-pipeline/kubemanifest/deployment.yaml

# 10. Deploy application
echo "Deploying application..."
kubectl apply -f pharma-ml-pipeline/kubemanifest/service-account.yaml
kubectl apply -f pharma-ml-pipeline/kubemanifest/deployment.yaml
kubectl apply -f pharma-ml-pipeline/kubemanifest/service.yaml

echo "Roll-forward process completed!"
echo "To access the application, check the service external IP:"
echo "kubectl get service pharma-ml-service"

# Provide command to monitor GPU nodes
echo "To check if GPU is available in your nodes:"
echo "kubectl get nodes -o=custom-columns=NAME:.metadata.name,GPU:.status.capacity.'nvidia\.com/gpu'"

# Provide command to check pods
echo "To check the status of your pods:"
echo "kubectl get pods"
