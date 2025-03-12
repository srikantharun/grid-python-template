import mlflow
import tensorflow as tf
import time
import json

class MLflowTrackingCallback(tf.keras.callbacks.Callback):
    """
    Callback for tracking TensorFlow/Keras training with MLflow
    """
    def __init__(self, log_batch=False):
        super().__init__()
        self.log_batch = log_batch
        self.batch_time = None
        self.epoch_time = None
        
    def on_train_begin(self, logs=None):
        mlflow.log_param("optimizer", self.model.optimizer._name)
        mlflow.log_param("loss_function", self.model.loss)
        
        # Log model architecture
        model_summary = []
        self.model.summary(print_fn=lambda x: model_summary.append(x))
        mlflow.log_text("\n".join(model_summary), "model_summary.txt")
        
        # Log model config
        model_config = self.model.get_config()
        mlflow.log_text(json.dumps(model_config, indent=2), "model_config.json")
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        # Log all metrics
        for metric_name, metric_value in logs.items():
            mlflow.log_metric(f"epoch_{metric_name}", metric_value, step=epoch)
            
        # Log epoch duration
        if self.epoch_time:
            epoch_duration = time.time() - self.epoch_time
            mlflow.log_metric("epoch_duration", epoch_duration, step=epoch)
    
    def on_batch_begin(self, batch, logs=None):
        if self.log_batch:
            self.batch_time = time.time()
    
    def on_batch_end(self, batch, logs=None):
        if self.log_batch:
            # Log batch metrics with higher resolution
            for metric_name, metric_value in logs.items():
                mlflow.log_metric(f"batch_{metric_name}", metric_value)
            
            # Log batch duration
            if self.batch_time:
                batch_duration = time.time() - self.batch_time
                mlflow.log_metric("batch_duration", batch_duration)

def track_experiment(experiment_name, run_name, params=None, tags=None):
    """
    Context manager for MLflow experiment tracking
    
    Args:
        experiment_name: Name of the MLflow experiment
        run_name: Name for this specific run
        params: Dictionary of parameters to log
        tags: Dictionary of tags to apply to the run
    
    Returns:
        Context manager for the MLflow run
    """
    # Set up experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    # Start run
    return mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags=tags
    )
