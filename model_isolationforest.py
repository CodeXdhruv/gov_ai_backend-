"""
Isolation Forest model for anomaly detection in GovAI System.
Primary unsupervised learning model for electricity theft detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
import os


class IsolationForestDetector:
    """
    Isolation Forest based anomaly detector for electricity consumption patterns.
    """
    
    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        max_samples: str = 'auto',
        random_state: int = 42
    ):
        """
        Initialize the Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of outliers (0.0 to 0.5)
            n_estimators: Number of trees in the forest
            max_samples: Number of samples to draw for each tree
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names: List[str] = []
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'IsolationForestDetector':
        """
        Fit the model on training data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: Optional list of feature names
            
        Returns:
            Self for method chaining
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit Isolation Forest
        self.model.fit(X_scaled)
        
        self.is_fitted = True
        self.feature_names = feature_names if feature_names else [f'feature_{i}' for i in range(X.shape[1])]
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of labels (1 = normal, -1 = anomaly)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores for samples.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of anomaly scores (lower = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        X_scaled = self.scaler.transform(X)
        return self.model.score_samples(X_scaled)
    
    def get_normalized_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Get normalized anomaly scores between 0 and 1.
        Higher score = more anomalous.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of normalized scores (0-1)
        """
        raw_scores = self.score_samples(X)
        
        # Isolation Forest scores are negative, more negative = more anomalous
        # We invert and normalize to 0-1 scale
        min_score = raw_scores.min()
        max_score = raw_scores.max()
        
        if max_score == min_score:
            return np.zeros_like(raw_scores)
        
        # Invert so higher = more anomalous
        normalized = (max_score - raw_scores) / (max_score - min_score)
        
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
            threshold: Optional custom threshold for anomaly (0-1)
            
        Returns:
            Tuple of (boolean anomaly labels, normalized scores)
        """
        scores = self.get_normalized_scores(X)
        
        if threshold is not None:
            is_anomaly = scores >= threshold
        else:
            # Use the model's built-in decision
            predictions = self.predict(X)
            is_anomaly = predictions == -1
        
        return is_anomaly, scores
    
    def save_model(self, filepath: str) -> bool:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            Success boolean
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'feature_names': self.feature_names,
                    'contamination': self.contamination,
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
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.contamination = data['contamination']
            self.is_fitted = data['is_fitted']
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


def run_isolation_forest_analysis(
    df: pd.DataFrame,
    feature_matrix: np.ndarray,
    feature_names: List[str],
    contamination: float = 0.05
) -> pd.DataFrame:
    """
    Run Isolation Forest analysis and add results to DataFrame.
    
    Args:
        df: Consumer DataFrame
        feature_matrix: Feature matrix for ML
        feature_names: List of feature column names
        contamination: Expected proportion of anomalies
        
    Returns:
        DataFrame with anomaly scores and labels
    """
    # Initialize and fit detector
    detector = IsolationForestDetector(contamination=contamination)
    detector.fit(feature_matrix, feature_names)
    
    # Get predictions
    is_anomaly, scores = detector.detect_anomalies(feature_matrix)
    
    # Add results to DataFrame
    result_df = df.copy()
    result_df['anomaly_score'] = scores
    result_df['is_anomaly'] = is_anomaly
    
    # Add status labels
    def get_status(row):
        if row['anomaly_score'] >= 0.9:
            return 'High Risk'
        elif row['anomaly_score'] >= 0.75:
            return 'Suspicious'
        elif row['anomaly_score'] >= 0.5:
            return 'Review Needed'
        else:
            return 'Normal'
    
    result_df['status'] = result_df.apply(get_status, axis=1)
    
    return result_df


def get_analysis_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate analysis summary from results DataFrame.
    
    Args:
        df: Results DataFrame with anomaly scores
        
    Returns:
        Summary dictionary
    """
    summary = {
        'total_consumers': len(df),
        'anomalies_detected': int(df['is_anomaly'].sum()) if 'is_anomaly' in df.columns else 0,
        'high_risk_count': len(df[df['status'] == 'High Risk']) if 'status' in df.columns else 0,
        'suspicious_count': len(df[df['status'] == 'Suspicious']) if 'status' in df.columns else 0,
        'review_needed_count': len(df[df['status'] == 'Review Needed']) if 'status' in df.columns else 0,
        'avg_anomaly_score': float(df['anomaly_score'].mean()) if 'anomaly_score' in df.columns else 0,
        'max_anomaly_score': float(df['anomaly_score'].max()) if 'anomaly_score' in df.columns else 0,
    }
    
    # High risk zones (regions with most anomalies)
    if 'region' in df.columns and 'is_anomaly' in df.columns:
        zone_anomalies = df.groupby('region')['is_anomaly'].sum()
        high_risk_zones = zone_anomalies[zone_anomalies > 0].sort_values(ascending=False)
        summary['high_risk_zones'] = high_risk_zones.index.tolist()[:5]
        summary['zone_anomaly_counts'] = high_risk_zones.to_dict()
    else:
        summary['high_risk_zones'] = []
        summary['zone_anomaly_counts'] = {}
    
    return summary
