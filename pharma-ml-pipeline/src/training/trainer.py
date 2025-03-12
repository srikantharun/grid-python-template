import tensorflow as tf
import mlflow
import os
import sys
import logging
from datetime import datetime

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import KaggleDataLoader
from data.preprocessing import PharmaDataPreprocessor
from models.transformer_model import PharmaTransformerModel
from MLFlow.mlflow_tracking import setup_mlflow_tracking
from MLFlow.experiment_tracking import MLflowTrackingCallback, track_experiment
from MLFlow.model_registry import register_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PharmaModelTrainer:
    """
    Trainer for pharmaceutical model pipeline
    """
    def __init__(self, experiment_name="pharma-model"):
        self.experiment_name = experiment_name
        
        # Set up MLflow
        self.experiment_id = setup_mlflow_tracking(experiment_name)
        
        # Initialize components
        self.data_loader = KaggleDataLoader()
        self.preprocessor = PharmaDataPreprocessor()
        
    def train_transformer_model(self, dataset_name, text_col, target_col,
                              pretrained_model="distilbert-base-uncased",
                              num_classes=None, epochs=10, batch_size=16,
                              learning_rate=5e-5, max_length=512):
        """
        Train a transformer model on text data
        
        Args:
            dataset_name: Kaggle dataset name
            text_col: Column name for text data
            target_col: Column name for target data
            pretrained_model: Name of pre-trained model to use
            num_classes: Number of classes (None for regression)
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            max_length: Maximum sequence length
            
        Returns:
            Trained model
        """
        run_name = f"transformer-{pretrained_model.split('/')[-1]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Set up MLflow run
        with track_experiment(
            self.experiment_name, 
            run_name,
            params={
                "model_type": "transformer",
                "pretrained_model": pretrained_model,
                "num_classes": num_classes,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "max_length": max_length
            }
        ) as run:
            logger.info(f"Started MLflow run: {run.info.run_id}")
            
            # Download and load the dataset
            dataset_path = self.data_loader.download_dataset(dataset_name)
            
            # Find CSV files in the dataset
            import glob
            csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
            
            if not csv_files:
                raise ValueError(f"No CSV files found in dataset {dataset_name}")
                
            # Load the first CSV file (or implement logic to select the right one)
            df = self.data_loader.load_csv_data(csv_files[0])
            
            # Initialize tokenizer from the model
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
            
            # Log dataset info
            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("data_shape", str(df.shape))
            mlflow.log_param("text_column", text_col)
            mlflow.log_param("target_column", target_col)
            
            # Preprocess data
            text_data = df[text_col].tolist()
            
            # Tokenize text data
            tokenized_data = tokenizer(
                text_data,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='tf'
            )
            
            # Prepare target data
            if num_classes is not None and num_classes > 2:
                # One-hot encode for multi-class
                targets = tf.keras.utils.to_categorical(df[target_col].values)
            else:
                targets = df[target_col].values
            
            # Split data
            from sklearn.model_selection import train_test_split
            indices = range(len(text_data))
            train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
            
            # Extract train/test sets
            X_train = {
                'input_ids': tokenized_data['input_ids'][train_idx],
                'attention_mask': tokenized_data['attention_mask'][train_idx]
            }
            X_test = {
                'input_ids': tokenized_data['input_ids'][test_idx],
                'attention_mask': tokenized_data['attention_mask'][test_idx]
            }
            
            y_train = targets[train_idx]
            y_test = targets[test_idx]
            
            # Initialize model
            model = PharmaTransformerModel(
                pretrained_model=pretrained_model,
                num_classes=num_classes,
                max_length=max_length
            )
            
            # Compile model
            model.compile_model(learning_rate=learning_rate)
            
            # Set up callbacks
            callbacks = [
                MLflowTrackingCallback(),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=2,
                    min_lr=1e-6
                )
            ]
            
            # Train model
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks
            )
            
            # Evaluate model
            eval_results = model.model.evaluate(X_test, y_test)
            
            # Log evaluation metrics
            metrics = dict(zip(model.model.metrics_names, eval_results))
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"eval_{metric_name}", metric_value)
            
            # Generate predictions for signature
            y_pred = model.predict(X_test)
            
            # Register model
            model_version = register_model(
                model.model,
                f"pharma-transformer-{pretrained_model.split('/')[-1]}",
                X_test,
                y_pred,
                version_description=f"Trained on {dataset_name} for {epochs} epochs"
            )
            
            logger.info(f"Registered model version: {model_version}")
            
            # Save model locally
            model_path = f"models/transformer_{run.info.run_id}"
            model.save_model(model_path)
            mlflow.log_artifacts(model_path, "saved_model")
            
            return model
            
    def train_cnn_model(self, dataset_name, smiles_col, target_col, 
                     num_classes=None, epochs=50, batch_size=32):
        """
        Train a CNN model on molecular fingerprints
        
        Args:
            dataset_name: Kaggle dataset name
            smiles_col: Column name for SMILES data
            target_col: Column name for target data
            num_classes: Number of classes (None for regression)
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Trained CNN model
        """
        run_name = f"cnn-model-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Set up MLflow run
        with track_experiment(
            self.experiment_name, 
            run_name,
            params={
                "model_type": "cnn",
                "num_classes": num_classes,
                "epochs": epochs,
                "batch_size": batch_size
            }
        ) as run:
            logger.info(f"Started MLflow run: {run.info.run_id}")
            
            # Download and load the dataset
            dataset_path = self.data_loader.download_dataset(dataset_name)
            
            # Find CSV files in the dataset
            import glob
            csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
            
            if not csv_files:
                raise ValueError(f"No CSV files found in dataset {dataset_name}")
                
            # Load the first CSV file (or implement logic to select the right one)
            df = self.data_loader.load_csv_data(csv_files[0])
            
            # Log dataset info
            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("data_shape", str(df.shape))
            mlflow.log_param("smiles_column", smiles_col)
            mlflow.log_param("target_column", target_col)
            
            # Preprocess data
            data = self.preprocessor.prepare_data(
                df, 
                smiles_col=smiles_col, 
                target_col=target_col
            )
            
            X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
            
            # Build model
            input_shape = X_train.shape[1:]
            
            # Create CNN model for fingerprints
            inputs = tf.keras.layers.Input(shape=input_shape)
            
            # Reshape for CNN if needed
            if len(input_shape) == 1:
                x = tf.keras.layers.Reshape((input_shape[0], 1))(inputs)
            else:
                x = inputs
                
            # Add CNN layers
            x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x)
            x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
            x = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu')(x)
            x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
            x = tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation='relu')(x)
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            
            # Add fully connected layers
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            
            # Output layer
            if num_classes is None or num_classes == 1:
                outputs = tf.keras.layers.Dense(1)(x)
                loss = 'mse'
                metrics = ['mae']
            elif num_classes == 2:
                outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
                loss = 'binary_crossentropy'
                metrics = ['accuracy', tf.keras.metrics.AUC()]
            else:
                outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
                loss = 'categorical_crossentropy'
                metrics = ['accuracy']
                
            # Create model
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-4),
                loss=loss,
                metrics=metrics
            )
            
            # Set up callbacks
            callbacks = [
                MLflowTrackingCallback(),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
            
            # Train model
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks
            )
            
            # Evaluate model
            eval_results = model.evaluate(X_test, y_test)
            
            # Log evaluation metrics
            metrics_dict = dict(zip(model.metrics_names, eval_results))
            for metric_name, metric_value in metrics_dict.items():
                mlflow.log_metric(f"eval_{metric_name}", metric_value)
            
            # Generate predictions for signature
            y_pred = model.predict(X_test)
            
            # Register model
            model_version = register_model(
                model,
                "pharma-cnn-fingerprint",
                X_test,
                y_pred,
                version_description=f"Trained on {dataset_name} for {epochs} epochs"
            )
            
            logger.info(f"Registered model version: {model_version}")
            
            # Save model locally
            model_path = f"models/cnn_{run.info.run_id}"
            model.save(model_path)
            mlflow.log_artifacts(model_path, "saved_model")
            
            return model
