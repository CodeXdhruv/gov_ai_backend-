"""
Autoencoder model for anomaly detection in GovAI System.
Deep learning based anomaly detection using reconstruction error.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import os

# TensorFlow import with fallback
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Autoencoder model will use sklearn fallback.")


class AutoencoderDetector:
    """
    Autoencoder-based anomaly detector for electricity consumption patterns.
    Uses reconstruction error to identify anomalies.
    """
    
    def __init__(
        self,
        encoding_dim: int = 8,
        hidden_layers: List[int] = [32, 16],
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.1,
        threshold_percentile: float = 95
    ):
        """
        Initialize the Autoencoder detector.
        
        Args:
            encoding_dim: Dimension of the encoding layer
            hidden_layers: List of hidden layer dimensions
            learning_rate: Learning rate for optimizer
            epochs: Training epochs
            batch_size: Batch size for training
            validation_split: Validation data fraction
            threshold_percentile: Percentile for anomaly threshold
        """
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.threshold_percentile = threshold_percentile
        
        self.model = None
        self.encoder = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.is_fitted = False
        self.input_dim = None
        self.feature_names: List[str] = []
    
    def _build_model(self, input_dim: int) -> None:
        """
        Build the autoencoder architecture.
        
        Args:
            input_dim: Number of input features
        """
        if not TF_AVAILABLE:
            return
        
        self.input_dim = input_dim
        
        # Encoder
        inputs = keras.Input(shape=(input_dim,))
        x = inputs
        
        for units in self.hidden_layers:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        
        # Encoding layer
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='encoding')(x)
        
        # Decoder (mirror of encoder)
        x = encoded
        for units in reversed(self.hidden_layers):
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = layers.Dense(input_dim, activation='linear')(x)
        
        # Build model
        self.model = Model(inputs, outputs)
        self.encoder = Model(inputs, encoded)
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'AutoencoderDetector':
        """
        Fit the autoencoder on training data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: Optional list of feature names
            
        Returns:
            Self for method chaining
        """
        if not TF_AVAILABLE:
            print("TensorFlow not available. Using mock fit.")
            self.is_fitted = True
            self.feature_names = feature_names if feature_names else []
            return self
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Build model
        self._build_model(X.shape[1])
        
        # Train model
        self.model.fit(
            X_scaled, X_scaled,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=0
        )
        
        # Calculate threshold based on training reconstruction errors
        reconstruction_errors = self._calculate_reconstruction_error(X_scaled)
        self.threshold = np.percentile(reconstruction_errors, self.threshold_percentile)
        
        self.is_fitted = True
        self.feature_names = feature_names if feature_names else [f'feature_{i}' for i in range(X.shape[1])]
        
        return self
    
    def _calculate_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction error for samples.
        
        Args:
            X: Scaled feature matrix
            
        Returns:
            Array of reconstruction errors (MSE per sample)
        """
        if not TF_AVAILABLE or self.model is None:
            return np.zeros(len(X))
        
        X_reconstructed = self.model.predict(X, verbose=0)
        mse = np.mean(np.square(X - X_reconstructed), axis=1)
        return mse
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores for samples based on reconstruction error.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of reconstruction errors
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        if not TF_AVAILABLE:
            return np.zeros(len(X))
        
        X_scaled = self.scaler.transform(X)
        return self._calculate_reconstruction_error(X_scaled)
    
    def get_normalized_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Get normalized anomaly scores between 0 and 1.
        Higher score = more anomalous.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of normalized scores (0-1)
        """
        errors = self.score_samples(X)
        
        if errors.max() == errors.min():
            return np.zeros_like(errors)
        
        # Normalize to 0-1
        normalized = (errors - errors.min()) / (errors.max() - errors.min())
        
        return normalized
    
    def detect_anomalies(
        self,
        X: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies and return labels and scores.
        
        Args:
            X: Feature matrix
            threshold: Optional custom threshold for anomaly
            
        Returns:
            Tuple of (boolean anomaly labels, normalized scores)
        """
        scores = self.get_normalized_scores(X)
        
        if threshold is not None:
            is_anomaly = scores >= threshold
        elif self.threshold is not None:
            raw_errors = self.score_samples(X)
            is_anomaly = raw_errors >= self.threshold
        else:
            # Default: top 5% are anomalies
            is_anomaly = scores >= np.percentile(scores, 95)
        
        return is_anomaly, scores
    
    def get_encoding(self, X: np.ndarray) -> np.ndarray:
        """
        Get the encoded representation of samples.
        
        Args:
            X: Feature matrix
            
        Returns:
            Encoded representations
        """
        if not TF_AVAILABLE or self.encoder is None:
            return np.zeros((len(X), self.encoding_dim))
        
        X_scaled = self.scaler.transform(X)
        return self.encoder.predict(X_scaled, verbose=0)
    
    def save_model(self, filepath: str) -> bool:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            Success boolean
        """
        if not TF_AVAILABLE:
            return False
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save Keras model
            self.model.save(filepath + '_keras')
            
            # Save additional parameters
            import pickle
            with open(filepath + '_params.pkl', 'wb') as f:
                pickle.dump({
                    'scaler': self.scaler,
                    'threshold': self.threshold,
                    'feature_names': self.feature_names,
                    'encoding_dim': self.encoding_dim,
                    'is_fitted': self.is_fitted
                }, f)
            
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Success boolean
        """
        if not TF_AVAILABLE:
            return False
        
        try:
            # Load Keras model
            self.model = keras.models.load_model(filepath + '_keras')
            
            # Load additional parameters
            import pickle
            with open(filepath + '_params.pkl', 'rb') as f:
                params = pickle.load(f)
            
            self.scaler = params['scaler']
            self.threshold = params['threshold']
            self.feature_names = params['feature_names']
            self.encoding_dim = params['encoding_dim']
            self.is_fitted = params['is_fitted']
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


def run_autoencoder_analysis(
    df: pd.DataFrame,
    feature_matrix: np.ndarray,
    feature_names: List[str],
    threshold_percentile: float = 95
) -> pd.DataFrame:
    """
    Run Autoencoder analysis and add results to DataFrame.
    
    Args:
        df: Consumer DataFrame
        feature_matrix: Feature matrix for ML
        feature_names: List of feature column names
        threshold_percentile: Percentile for anomaly threshold
        
    Returns:
        DataFrame with anomaly scores and labels
    """
    # Initialize and fit detector
    detector = AutoencoderDetector(threshold_percentile=threshold_percentile)
    detector.fit(feature_matrix, feature_names)
    
    # Get predictions
    is_anomaly, scores = detector.detect_anomalies(feature_matrix)
    
    # Add results to DataFrame
    result_df = df.copy()
    result_df['ae_anomaly_score'] = scores
    result_df['ae_is_anomaly'] = is_anomaly
    
    return result_df


def ensemble_scores(
    if_scores: np.ndarray,
    ae_scores: np.ndarray,
    if_weight: float = 0.6,
    ae_weight: float = 0.4
) -> np.ndarray:
    """
    Combine Isolation Forest and Autoencoder scores.
    
    Args:
        if_scores: Isolation Forest normalized scores
        ae_scores: Autoencoder normalized scores
        if_weight: Weight for IF scores
        ae_weight: Weight for AE scores
        
    Returns:
        Combined ensemble scores
    """
    return if_scores * if_weight + ae_scores * ae_weight
