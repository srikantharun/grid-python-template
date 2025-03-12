# Setting Up GPU Nodes in Azure Kubernetes Service (AKS)

This guide explains how to create and configure GPU-enabled node pools in Azure Kubernetes Service (AKS) for running machine learning workloads.

## Prerequisites

- Azure CLI installed and configured
- Kubernetes CLI (`kubectl`) installed
- An existing Azure Kubernetes Service (AKS) cluster or permissions to create one

## Step 1: Register for GPU Resources on Azure

First, register your subscription to use GPU resources:

```bash
# Register the Microsoft.ContainerService provider if not already registered
az provider register --namespace Microsoft.ContainerService

# Register for GPU resources
az feature register --name GPUDedicatedVHDPreview --namespace Microsoft.ContainerService
```

Check registration status (it may take a few minutes):

```bash
az feature show --name GPUDedicatedVHDPreview --namespace Microsoft.ContainerService
```

Once the registration state is "Registered", refresh the provider:

```bash
az provider register --namespace Microsoft.ContainerService
```

## Step 2: Create an AKS Cluster (Skip if you already have one)

If you don't already have an AKS cluster, create one:

```bash
# Create a resource group
az group create --name pharma-ml-rg --location eastus

# Create AKS cluster (without GPU nodes initially)
az aks create \
    --resource-group pharma-ml-rg \
    --name pharma-ml-cluster \
    --node-count 2 \
    --generate-ssh-keys \
    --enable-managed-identity \
    --node-vm-size Standard_D4s_v3
```

## Step 3: Add a GPU Node Pool to Your Cluster

Now, add a GPU-enabled node pool to your cluster:

```bash
# List available GPU VM sizes in your region
az vm list-sizes --location eastus --query "[?contains(name, 'Standard_NC') || contains(name, 'Standard_ND')].name" -o table

# Add a GPU node pool
az aks nodepool add \
    --resource-group pharma-ml-rg \
    --cluster-name pharma-ml-cluster \
    --name gpunodepool \
    --node-count 1 \
    --node-vm-size Standard_NC6s_v3 \
    --no-wait
```

**Note:** Common GPU VM sizes:
- `Standard_NC6s_v3`: 1 NVIDIA Tesla V100 GPU
- `Standard_NC12s_v3`: 2 NVIDIA Tesla V100 GPUs
- `Standard_NC24s_v3`: 4 NVIDIA Tesla V100 GPUs
- `Standard_ND40rs_v2`: 8 NVIDIA Tesla V100 GPUs

Monitor the node pool creation status:

```bash
az aks nodepool show \
    --resource-group pharma-ml-rg \
    --cluster-name pharma-ml-cluster \
    --name gpunodepool \
    --query provisioningState -o tsv
```

## Step 4: Install NVIDIA Device Plugin

Connect to your AKS cluster:

```bash
az aks get-credentials --resource-group pharma-ml-rg --name pharma-ml-cluster
```

Install the NVIDIA device plugin using Helm:

```bash
# Add NVIDIA Helm repository
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update

# Install the plugin
helm install \
    --namespace kube-system \
    nvidia-device-plugin \
    nvdp/nvidia-device-plugin
```

## Step 5: Verify GPU Nodes and Resources

Verify that your GPU nodes are available:

```bash
# Check nodes with GPU resources
kubectl get nodes -o json | jq '.items[] | {name:.metadata.name, gpu:.status.allocatable["nvidia.com/gpu"]}'

# Check NVIDIA device plugin pods
kubectl get pods -n kube-system | grep nvidia-device-plugin

# Verify GPU resources are available
kubectl describe nodes | grep -A10 "Capacity:" | grep "nvidia.com/gpu"
```

## Step 6: Create a PVC for ML Data and Models

Create Persistent Volume Claims for your ML data and models:

```bash
# Create PVCs
kubectl apply -f kubemanifest/pvc.yaml
```

## Step 7: Deploy MLflow for Experiment Tracking

Deploy MLflow for tracking your experiments:

```bash
# Create MLflow namespace
kubectl create namespace mlflow

# Create Azure Storage Secret for MLflow (replace with your actual values)
kubectl create secret generic azure-storage-secret \
    --namespace mlflow \
    --from-literal=AZURE_STORAGE_ACCESS_KEY=your-access-key \
    --from-literal=AZURE_STORAGE_CONNECTION_STRING='your-connection-string'

# Deploy MLflow using Helm
helm repo add larribas https://larribas.github.io/helm-charts
helm repo update

helm install mlflow larribas/mlflow \
    --namespace mlflow \
    --set database.engine=postgresql \
    --set database.user=mlflow \
    --set database.password=mlflowpassword \
    --set database.host=mlflow-postgresql \
    --set database.port=5432 \
    --set database.name=mlflow \
    --set artifacts.backend=azure \
    --set artifacts.azure.storageAccount=yourstorageaccount \
    --set artifacts.azure.container=mlflow \
    --set artifacts.azure.accessKey=your-access-key
```

## Step 8: Deploy Training Job with GPU Resources

Deploy a training job that uses GPU resources:

```bash
# Create container registry if needed
az acr create --resource-group pharma-ml-rg --name pharmamlacrXXXXX --sku Basic

# Build and push your Docker image (replace with your values)
az acr build --registry pharmamlacrXXXXX --image pharma-ml:latest .

# Update deployment.yaml with your ACR name
# Replace $ACR_NAME with your actual ACR name in kubemanifest/deployment.yaml

# Deploy GPU-based training job
kubectl apply -f kubemanifest/deployment.yaml
```

## Step 9: Monitor GPU Usage

Monitor GPU usage in your pods:

```bash
# Check GPU allocation
kubectl get pods --all-namespaces | grep -i gpu

# To see GPU metrics, install the NVIDIA Data Center GPU Manager (DCGM) exporter
helm repo add gpu-helm-charts https://nvidia.github.io/dcgm-exporter/helm-charts
helm repo update
helm install --namespace default --name dcgm-exporter gpu-helm-charts/dcgm-exporter

# If you have Prometheus installed, you can now monitor GPU metrics
```

## Step 10: Auto-Scaling GPU Nodes (Optional)

Configure autoscaling for your GPU node pool:

```bash
# Enable autoscaling for GPU node pool
az aks nodepool update \
    --resource-group pharma-ml-rg \
    --cluster-name pharma-ml-cluster \
    --name gpunodepool \
    --enable-cluster-autoscaler \
    --min-count 0 \
    --max-count 3
```

## Troubleshooting

### Common Issues and Solutions

1. **GPU not detected in pods:**
   - Ensure NVIDIA device plugin is running
   - Check if GPU drivers are properly installed
   - Verify pod has proper resource requests/limits

   ```yaml
   resources:
     limits:
       nvidia.com/gpu: 1
   ```

2. **Out of GPU resources:**
   - Add more GPU nodes to the node pool
   - Scale down other GPU workloads
   - Check if GPUs are being properly released after use

3. **Pod stuck in "Pending" state:**
   - Check for GPU resource availability
   - Check node taints
   - Inspect pod events: `kubectl describe pod <pod-name>`

4. **Performance issues:**
   - Check GPU utilization: `kubectl exec -it <pod-name> -- nvidia-smi`
   - Monitor memory usage
   - Use the NVIDIA profiler for detailed performance analysis

## Cost Optimization Tips

GPU resources on Azure can be expensive. Here are some tips to optimize costs:

1. **Use spot/preemptible instances for non-critical workloads:**

   ```bash
   az aks nodepool add \
       --resource-group pharma-ml-rg \
       --cluster-name pharma-ml-cluster \
       --name gpuspotpool \
       --node-count 1 \
       --node-vm-size Standard_NC6s_v3 \
       --priority Spot \
       --eviction-policy Delete \
       --spot-max-price -1
   ```

2. **Implement autoscaling to scale down when not in use**

3. **Scale to zero when no workloads are running:**

   ```bash
   az aks nodepool update \
       --resource-group pharma-ml-rg \
       --cluster-name pharma-ml-cluster \
       --name gpunodepool \
       --min-count 0
   ```

4. **Use reserved instances for long-running workloads**

5. **Consider Azure Machine Learning service for fully managed ML workflows**

## Additional Resources

- [AKS Documentation](https://docs.microsoft.com/en-us/azure/aks/)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/overview.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Kubeflow Documentation](https://www.kubeflow.org/docs/)
