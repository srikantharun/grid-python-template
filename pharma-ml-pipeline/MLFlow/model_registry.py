import mlflow
from mlflow.models import infer_signature
import os

def register_model(model, model_name, X_test, y_pred, version_description=None):
    """
    Register a trained model to MLflow model registry
    
    Args:
        model: Trained model object (TensorFlow, PyTorch, etc.)
        model_name: Name to register the model under
        X_test: Sample input data for signature inference
        y_pred: Sample output data for signature inference
        version_description: Optional description for this model version
    
    Returns:
        model_version: Version of the registered model
    """
    signature = infer_signature(X_test, y_pred)
    
    # Log model to current run
    if isinstance(model, tf.keras.Model):
        mlflow.tensorflow.log_model(
            model, 
            "model",
            signature=signature,
        )
    else:
        mlflow.sklearn.log_model(
            model, 
            "model",
            signature=signature,
        )
    
    # Register model to registry
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    model_details = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )
    
    # Add description if provided
    if version_description:
        client = MlflowClient()
        client.update_model_version(
            name=model_name,
            version=model_details.version,
            description=version_description
        )
    
    return model_details.version
