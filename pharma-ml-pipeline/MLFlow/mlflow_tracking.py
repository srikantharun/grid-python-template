import os
import mlflow
from mlflow.tracking import MlflowClient

def setup_mlflow_tracking(experiment_name="pharma-model", azure_storage=True):
    """
    Set up MLflow tracking with Azure backend storage
    
    Args:
        experiment_name: Name of the MLflow experiment
        azure_storage: Whether to use Azure Storage as artifact store
    
    Returns:
        experiment_id: ID of the created or existing experiment
    """
    # Set up tracking URI (can be AzureML or self-hosted MLflow)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Set up Azure Storage for artifacts if enabled
    if azure_storage:
        artifact_location = os.environ.get(
            "MLFLOW_ARTIFACT_ROOT", 
            "wasbs://mlflow@<your-storage-account>.blob.core.windows.net/artifacts"
        )
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name, 
                artifact_location=artifact_location
            )
        else:
            experiment_id = experiment.experiment_id
    else:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
    
    return experiment_id
