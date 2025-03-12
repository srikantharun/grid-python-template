import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

class MolecularCNNModel:
    """
    CNN model for molecular fingerprint data
    """
    def __init__(self, input_shape=(2048,), num_classes=None, dropout_rate=0.5, 
                conv_layers=[(64, 3), (128, 3), (256, 3)], fc_layers=[256, 128]):
        """
        Initialize the CNN model

        Args:
            input_shape: Shape of input fingerprint vectors
            num_classes: Number of output classes (None for regression)
            dropout_rate: Dropout rate for regularization
            conv_layers: List of tuples (filters, kernel_size) for conv layers
            fc_layers: List of units for fully connected layers
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.conv_layers = conv_layers
        self.fc_layers = fc_layers
        
        # Build model
        self.model = self._build_model()
        
    def _build_model(self):
        """
        Build the CNN model architecture
        
        Returns:
            TensorFlow model
        """
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Reshape for 1D convolution if input is 1D
        if len(self.input_shape) == 1:
            x = layers.Reshape((self.input_shape[0], 1))(inputs)
        else:
            x = inputs
        
        # Add convolutional layers
        for filters, kernel_size in self.conv_layers:
            x = layers.Conv1D(
                filters=filters, 
                kernel_size=kernel_size, 
                activation='relu',
                padding='same'
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling1D(pool_size=2)(x)
        
        # Flatten and add fully connected layers
        x = layers.GlobalAveragePooling1D()(x)
        
        for units in self.fc_layers:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer
        if self.num_classes is None or self.num_classes == 1:
            # Regression
            outputs = layers.Dense(1)(x)
        elif self.num_classes == 2:
            # Binary classification
            outputs = layers.Dense(1, activation='sigmoid')(x)
        else:
            # Multi-class classification
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def compile_model(self, learning_rate=1e-4, loss=None, metrics=None, 
                     optimizer=None, mixed_precision=False):
        """
        Compile the model
        
        Args:
            learning_rate: Learning rate for the optimizer
            loss: Loss function (auto-selected if None)
            metrics: List of metrics (auto-selected if None)
            optimizer: Optimizer to use (Adam by default)
            mixed_precision: Whether to use mixed precision training
        """
        # Set up mixed precision if requested
        if mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
        
        # Determine appropriate loss function if not specified
        if loss is None:
            if self.num_classes is None or self.num_classes == 1:
                loss = tf.keras.losses.MeanSquaredError()
            elif self.num_classes == 2:
                loss = tf.keras.losses.BinaryCrossentropy()
            else:
                loss = tf.keras.losses.CategoricalCrossentropy()
                
        # Determine appropriate metrics if not specified
        if metrics is None:
            if self.num_classes is None or self.num_classes == 1:
                metrics = ['mae', 'mse']
            elif self.num_classes == 2:
                metrics = ['accuracy', tf.keras.metrics.AUC()]
            else:
                metrics = ['accuracy']
        
        # Set up optimizer
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
    def fit(self, X, y, validation_data=None, epochs=50, batch_size=32, 
           callbacks=None, class_weights=None):
        """
        Train the model
        
        Args:
            X: Input features
            y: Target values
            validation_data: Tuple of (X_val, y_val)
            epochs: Number of training epochs
            batch_size: Training batch size
            callbacks: List of Keras callbacks
            class_weights: Weights for imbalanced classes
            
        Returns:
            Training history
        """
        return self.model.fit(
            X, y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights
        )
        
    def predict(self, X):
        """
        Generate predictions
        
        Args:
            X: Input features
            
        Returns:
            Model predictions
        """
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """
        Evaluate the model
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Evaluation metrics
        """
        return self.model.evaluate(X, y)
    
    def save_model(self, path):
        """
        Save the model
        
        Args:
            path: Path to save the model
        """
        self.model.save(path)
        
        # Save model configuration
        import json
        config = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'conv_layers': self.conv_layers,
            'fc_layers': self.fc_layers
        }
        
        with open(f"{path}/model_config.json", "w") as f:
            json.dump(config, f)
        
    @classmethod
    def load_model(cls, path):
        """
        Load a saved model
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        # Create instance
        instance = cls.__new__(cls)
        
        # Load model configuration
        import json
        with open(f"{path}/model_config.json", "r") as f:
            config = json.load(f)
        
        # Set attributes
        for key, value in config.items():
            setattr(instance, key, value)
        
        # Load model
        instance.model = tf.keras.models.load_model(path)
        
        return instance

    def convert_smiles_to_fingerprints(self, smiles_list, fp_type='morgan', 
                                      radius=2, n_bits=2048):
        """
        Convert SMILES strings to molecular fingerprints
        
        Args:
            smiles_list: List of SMILES strings
            fp_type: Type of fingerprint ('morgan', 'rdkit', 'maccs')
            radius: Radius for Morgan fingerprints
            n_bits: Number of bits in fingerprint
            
        Returns:
            Numpy array of fingerprints
        """
        fingerprints = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                # Handle invalid SMILES
                fingerprints.append(np.zeros(n_bits))
                continue
                
            if fp_type == 'morgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            elif fp_type == 'rdkit':
                fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
            elif fp_type == 'maccs':
                fp = AllChem.GetMACCSKeysFingerprint(mol)
                
            fingerprints.append(np.array(fp))
            
        return np.array(fingerprints)
