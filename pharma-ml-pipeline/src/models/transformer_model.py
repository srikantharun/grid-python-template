import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
import os

class PharmaTransformerModel:
    """
    Transformer-based model for pharmaceutical tasks
    """
    def __init__(self, pretrained_model="distilbert-base-uncased", 
                num_classes=None, max_length=512):
        self.pretrained_model = pretrained_model
        self.max_length = max_length
        self.num_classes = num_classes
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        
        # Build model
        self.model = self._build_model()
        
    def _build_model(self):
        """
        Build the transformer model architecture
        
        Returns:
            TensorFlow model
        """
        # Input layers
        input_ids = tf.keras.layers.Input(
            shape=(self.max_length,), dtype=tf.int32, name="input_ids"
        )
        attention_mask = tf.keras.layers.Input(
            shape=(self.max_length,), dtype=tf.int32, name="attention_mask"
        )
        
        # Load pre-trained transformer
        transformer = TFAutoModel.from_pretrained(self.pretrained_model)
        
        # Get transformer outputs
        outputs = transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use the [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Add dropout
        x = tf.keras.layers.Dropout(0.1)(pooled_output)
        
        # Add classification or regression head
        if self.num_classes is not None:
            if self.num_classes == 1:
                # Regression task
                outputs = tf.keras.layers.Dense(1)(x)
            elif self.num_classes == 2:
                # Binary classification
                outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            else:
                # Multi-class classification
                outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        else:
            # Feature extraction only
            outputs = x
            
        # Create model
        model = tf.keras.Model(
            inputs=[input_ids, attention_mask],
            outputs=outputs
        )
        
        return model
    
    def compile_model(self, learning_rate=5e-5, loss=None, metrics=None):
        """
        Compile the model
        
        Args:
            learning_rate: Learning rate for the optimizer
            loss: Loss function (auto-selected if None)
            metrics: List of metrics (auto-selected if None)
        """
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
                
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )
        
    def fit(self, X, y, validation_data=None, epochs=10, batch_size=16, callbacks=None):
        """
        Train the model
        
        Args:
            X: Input features or tokenized text
            y: Target values
            validation_data: Tuple of (X_val, y_val)
            epochs: Number of training epochs
            batch_size: Training batch size
            callbacks: List of Keras callbacks
            
        Returns:
            Training history
        """
        return self.model.fit(
            X, y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
    def predict(self, X):
        """
        Generate predictions
        
        Args:
            X: Input features or tokenized text
            
        Returns:
            Model predictions
        """
        return self.model.predict(X)
    
    def save_model(self, path):
        """
        Save the model
        
        Args:
            path: Path to save the model
        """
        self.model.save(path)
        
        # Save tokenizer
        tokenizer_path = os.path.join(path, 'tokenizer')
        if not os.path.exists(tokenizer_path):
            os.makedirs(tokenizer_path)
        self.tokenizer.save_pretrained(tokenizer_path)
        
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
        
        # Load tokenizer
        tokenizer_path = os.path.join(path, 'tokenizer')
        instance.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Load model
        instance.model = tf.keras.models.load_model(path)
        
        return instance
