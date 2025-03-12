#!/bin/bash
set -e

# This script starts the full ML pipeline

# Parse command line arguments
DATASET=""
MODEL_TYPE="transformer"
GPU_NODE_COUNT=1

print_usage() {
  echo "Usage: $0 -d DATASET -m MODEL_TYPE -g GPU_NODE_COUNT"
  echo "  -d DATASET       Dataset name (Kaggle dataset or 'sample')"
  echo "  -m MODEL_TYPE    Model type (transformer or cnn)"
  echo "  -g GPU_COUNT     Number of GPU nodes to provision (default: 1)"
  exit 1
}

while getopts ":d:m:g:" opt; do
  case ${opt} in
    d )
      DATASET=$OPTARG
      ;;
    m )
      MODEL_TYPE=$OPTARG
      ;;
    g )
      GPU_NODE_COUNT=$OPTARG
      ;;
    \? )
      echo "Invalid option: $OPTARG" 1>&2
      print_usage
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      print_usage
      ;;
  esac
done

if [ -z "$DATASET" ]; then
  echo "Error: Dataset is required"
  print_usage
fi

echo "Starting pipeline with dataset: $DATASET, model: $MODEL_TYPE, GPU nodes: $GPU_NODE_COUNT"

# Scale up GPU node pool if needed
if [ $GPU_NODE_COUNT -gt 0 ]; then
  echo "Scaling GPU node pool to $GPU_NODE_COUNT nodes..."
  az aks nodepool update \
    --resource-group pharma-ml-rg \
    --cluster-name pharma-ml-cluster \
    --name gpunodepool \
    --node-count $GPU_NODE_COUNT
  
  # Wait for nodes to be ready
  echo "Waiting for GPU nodes to be ready..."
  kubectl wait --for=condition=Ready nodes -l accelerator=nvidia-tesla --timeout=300s
fi

# Create Kubernetes job for training
cat << EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: pharma-ml-training-job
spec:
  template:
    spec:
      containers:
      - name: pharma-ml-training
        image: $(kubectl get deployment pharma-ml-pipeline -o jsonpath='{.spec.template.spec.containers[0].image}')
        command: ["./scripts/entrypoint.sh", "train", "$MODEL_TYPE", "$DATASET"]
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "4"
          requests:
            nvidia.com/gpu: 1
            memory: "4Gi"
            cpu: "2"
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow-service:5000"
        - name: AZURE_STORAGE_CONNECTION_STRING
          valueFrom:
            secretKeyRef:
              name: azure-credentials
              key: AZURE_STORAGE_CONNECTION_STRING
        - name: KAGGLE_USERNAME
          valueFrom:
            secretKeyRef:
              name: azure-credentials
              key: KAGGLE_USERNAME
        - name: KAGGLE_KEY
          valueFrom:
            secretKeyRef:
              name: azure-credentials
              key: KAGGLE_KEY
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: model-volume
          mountPath: /app/models
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: data-pvc
      - name: model-volume
        persistentVolumeClaim:
          claimName: model-pvc
      nodeSelector:
        accelerator: nvidia-tesla
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      restartPolicy: Never
  backoffLimit: 2
EOF

echo "Training job created. Monitoring logs..."
kubectl wait --for=condition=complete job/pharma-ml-training-job --timeout=3600s
kubectl logs -f job/pharma-ml-training-job

# Scale down GPU node pool if needed
if [ $GPU_NODE_COUNT -gt 0 ]; then
  echo "Scaling down GPU node pool to save costs..."
  az aks nodepool update \
    --resource-group pharma-ml-rg \
    --cluster-name pharma-ml-cluster \
    --name gpunodepool \
    --node-count 0
fi

echo "Pipeline completed successfully!"
