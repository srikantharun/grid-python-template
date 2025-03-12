import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, mean_absolute_error, mean_squared_error,
    r2_score, explained_variance_score
)
import mlflow
import io
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Evaluator for machine learning models in pharmaceutical applications
    """
    def __init__(self):
        """Initialize the evaluator"""
        pass
        
    def evaluate_model(self, model, X_test, y_test, task_type=None, metrics=None):
        """
        Evaluate a model on test data
        
        Args:
            model: Trained model with predict method
            X_test: Test features
            y_test: Test targets
            task_type: Type of task ('classification', 'regression', or None for auto-detect)
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Predict on test data
        y_pred = model.predict(X_test)
        
        # Reshape predictions if needed
        if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()
        
        # Auto-detect task type if not specified
        if task_type is None:
            if np.array_equal(np.unique(y_test), [0, 1]) or len(np.unique(y_test)) < 10:
                task_type = 'classification'
            else:
                task_type = 'regression'
                
        # Determine metrics to compute
        if metrics is None:
            if task_type == 'classification':
                metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            else:
                metrics = ['mse', 'rmse', 'mae', 'r2', 'explained_variance']
        
        # Compute metrics
        results = {}
        
        if task_type == 'classification':
            # Determine if binary or multi-class
            is_binary = len(np.unique(y_test)) == 2
            
            # For binary classification, threshold predictions if needed
            if is_binary and len(y_pred.shape) == 1:
                y_pred_class = (y_pred > 0.5).astype(int)
            elif is_binary and y_pred.shape[1] == 2:
                y_pred_class = np.argmax(y_pred, axis=1)
            else:
                y_pred_class = np.argmax(y_pred, axis=1)
            
            # Compute classification metrics
            for metric in metrics:
                if metric == 'accuracy':
                    results['accuracy'] = accuracy_score(y_test, y_pred_class)
                elif metric == 'precision':
                    if is_binary:
                        results['precision'] = precision_score(y_test, y_pred_class)
                    else:
                        results['precision'] = precision_score(y_test, y_pred_class, average='weighted')
                elif metric == 'recall':
                    if is_binary:
                        results['recall'] = recall_score(y_test, y_pred_class)
                    else:
                        results['recall'] = recall_score(y_test, y_pred_class, average='weighted')
                elif metric == 'f1':
                    if is_binary:
                        results['f1'] = f1_score(y_test, y_pred_class)
                    else:
                        results['f1'] = f1_score(y_test, y_pred_class, average='weighted')
                elif metric == 'roc_auc':
                    if is_binary:
                        if len(y_pred.shape) == 1:
                            results['roc_auc'] = roc_auc_score(y_test, y_pred)
                        else:
                            results['roc_auc'] = roc_auc_score(y_test, y_pred[:, 1])
                    else:
                        # One-hot encode targets for multi-class ROC AUC
                        from sklearn.preprocessing import label_binarize
                        classes = np.unique(y_test)
                        y_test_bin = label_binarize(y_test, classes=classes)
                        results['roc_auc'] = roc_auc_score(y_test_bin, y_pred, average='macro', multi_class='ovr')
        else:
            # Compute regression metrics
            for metric in metrics:
                if metric == 'mse':
                    results['mse'] = mean_squared_error(y_test, y_pred)
                elif metric == 'rmse':
                    results['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
                elif metric == 'mae':
                    results['mae'] = mean_absolute_error(y_test, y_pred)
                elif metric == 'r2':
                    results['r2'] = r2_score(y_test, y_pred)
                elif metric == 'explained_variance':
                    results['explained_variance'] = explained_variance_score(y_test, y_pred)
        
        return results
    
    def plot_confusion_matrix(self, model, X_test, y_test, log_to_mlflow=False):
        """
        Generate and plot confusion matrix
        
        Args:
            model: Trained model with predict method
            X_test: Test features
            y_test: Test targets
            log_to_mlflow: Whether to log the plot to MLflow
            
        Returns:
            Matplotlib figure
        """
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Convert to class labels
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred = (y_pred > 0.5).astype(int)
            
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Log to MLflow if requested
        if log_to_mlflow:
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
            
        return plt.gcf()
    
    def plot_roc_curve(self, model, X_test, y_test, log_to_mlflow=False):
        """
        Generate and plot ROC curve
        
        Args:
            model: Trained model with predict method
            X_test: Test features
            y_test: Test targets
            log_to_mlflow: Whether to log the plot to MLflow
            
        Returns:
            Matplotlib figure
        """
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Ensure we have probability scores
        if len(y_pred.shape) > 1:
            if y_pred.shape[1] == 2:
                y_pred = y_pred[:, 1]  # Use probability of positive class
            else:
                logger.warning("Multi-class ROC curve not supported in this method")
                return None
                
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        
        # Log to MLflow if requested
        if log_to_mlflow:
            mlflow.log_figure(plt.gcf(), "roc_curve.png")
            
        return plt.gcf()
    
    def plot_precision_recall_curve(self, model, X_test, y_test, log_to_mlflow=False):
        """
        Generate and plot precision-recall curve
        
        Args:
            model: Trained model with predict method
            X_test: Test features
            y_test: Test targets
            log_to_mlflow: Whether to log the plot to MLflow
            
        Returns:
            Matplotlib figure
        """
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Ensure we have probability scores
        if len(y_pred.shape) > 1:
            if y_pred.shape[1] == 2:
                y_pred = y_pred[:, 1]  # Use probability of positive class
            else:
                logger.warning("Multi-class precision-recall curve not supported in this method")
                return None
                
        # Compute precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        
        # Log to MLflow if requested
        if log_to_mlflow:
            mlflow.log_figure(plt.gcf(), "precision_recall_curve.png")
            
        return plt.gcf()
    
    def plot_regression_results(self, model, X_test, y_test, log_to_mlflow=False):
        """
        Generate and plot actual vs predicted values for regression
        
        Args:
            model: Trained model with predict method
            X_test: Test features
            y_test: Test targets
            log_to_mlflow: Whether to log the plot to MLflow
            
        Returns:
            Matplotlib figure
        """
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Flatten if needed
        if len(y_pred.shape) > 1:
            y_pred = y_pred.flatten()
            
        # Compute metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Actual vs Predicted (R² = {r2:.3f}, MAE = {mae:.3f})')
        
        # Log to MLflow if requested
        if log_to_mlflow:
            mlflow.log_figure(plt.gcf(), "regression_results.png")
            
        return plt.gcf()
    
    def plot_feature_importance(self, model, feature_names, log_to_mlflow=False):
        """
        Plot feature importance for tree-based models
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            log_to_mlflow: Whether to log the plot to MLflow
            
        Returns:
            Matplotlib figure or None if not supported
        """
        # Check if model has feature importances
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return None
            
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort importances
        indices = np.argsort(importances)[::-1]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance')
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        
        # Log to MLflow if requested
        if log_to_mlflow:
            mlflow.log_figure(plt.gcf(), "feature_importance.png")
            
        return plt.gcf()
