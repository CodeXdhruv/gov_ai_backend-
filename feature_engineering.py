"""
Feature engineering module for GovAI Electricity Theft Detection System.
Extracts meaningful features from aggregated consumer data for anomaly detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def calculate_variability_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate variability and pattern features.
    
    Args:
        df: Aggregated consumer DataFrame
        
    Returns:
        DataFrame with additional variability features
    """
    df = df.copy()
    
    # Coefficient of variation (CV)
    if 'avg_consumption' in df.columns and 'std_consumption' in df.columns:
        df['cv_consumption'] = df['std_consumption'] / (df['avg_consumption'] + 1e-6)
    
    # Range ratio
    if 'max_consumption' in df.columns and 'min_consumption' in df.columns:
        df['range_consumption'] = df['max_consumption'] - df['min_consumption']
        df['range_ratio'] = df['range_consumption'] / (df['avg_consumption'] + 1e-6)
    
    return df


def calculate_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate behavioral pattern features.
    
    Args:
        df: Aggregated consumer DataFrame
        
    Returns:
        DataFrame with behavioral features
    """
    df = df.copy()
    
    # Night/Day consumption ratio
    if 'night_usage_ratio' in df.columns:
        # Higher ratio means more night usage
        df['night_day_ratio'] = df['night_usage_ratio'] / (1 - df['night_usage_ratio'] + 1e-6)
    
    # Weekend/Weekday ratio
    if 'weekend_readings_ratio' in df.columns:
        df['weekend_weekday_ratio'] = df['weekend_readings_ratio'] / (1 - df['weekend_readings_ratio'] + 1e-6)
    
    # Peak hour usage indicator
    if 'peak_usage_ratio' in df.columns:
        df['off_peak_ratio'] = 1 - df['peak_usage_ratio']
    
    return df


def calculate_anomaly_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate potential anomaly indicator features.
    
    Args:
        df: DataFrame with consumption features
        
    Returns:
        DataFrame with anomaly indicator features
    """
    df = df.copy()
    
    # Low consumption flag (potential meter bypass)
    if 'avg_consumption' in df.columns:
        avg_mean = df['avg_consumption'].mean()
        df['low_consumption_flag'] = (df['avg_consumption'] < avg_mean * 0.3).astype(int)
    
    # High variability flag (irregular patterns)
    if 'cv_consumption' in df.columns:
        cv_threshold = df['cv_consumption'].quantile(0.9)
        df['high_variability_flag'] = (df['cv_consumption'] > cv_threshold).astype(int)
    
    # Unusual night usage flag
    if 'night_day_ratio' in df.columns:
        night_threshold = df['night_day_ratio'].quantile(0.95)
        df['unusual_night_flag'] = (df['night_day_ratio'] > night_threshold).astype(int)
    
    return df


def create_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Create final feature matrix for ML models.
    
    Args:
        df: Aggregated consumer DataFrame
        
    Returns:
        Tuple of (feature matrix, feature names, full DataFrame with all features)
    """
    # Apply all feature engineering
    df_features = df.copy()
    df_features = calculate_variability_features(df_features)
    df_features = calculate_behavioral_features(df_features)
    df_features = calculate_anomaly_indicators(df_features)
    
    # Define ML features
    ml_feature_columns = [
        'avg_consumption',
        'std_consumption',
        'cv_consumption',
        'range_ratio',
        'night_day_ratio',
        'weekend_weekday_ratio',
        'off_peak_ratio',
        'low_consumption_flag',
        'high_variability_flag',
        'unusual_night_flag'
    ]
    
    # Filter to available columns
    available_features = [col for col in ml_feature_columns if col in df_features.columns]
    
    # Create feature matrix
    X = df_features[available_features].values
    
    # Handle NaN and infinite values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    return X, available_features, df_features


def scale_features(X: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, Any]:
    """
    Scale features for ML model input.
    
    Args:
        X: Feature matrix
        method: Scaling method ('standard' or 'minmax')
        
    Returns:
        Tuple of (scaled features, scaler object)
    """
    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler


def get_feature_importance_from_isolation_forest(model, feature_names: List[str]) -> Dict[str, float]:
    """
    Extract feature importance from Isolation Forest model.
    
    Note: Isolation Forest doesn't have direct feature importance,
    so we use a proxy based on tree splits.
    
    Args:
        model: Trained Isolation Forest model
        feature_names: List of feature names
        
    Returns:
        Dictionary of feature importances
    """
    # Use average path length as a proxy for importance
    # Features that cause more splits are more important
    importances = {}
    
    n_features = len(feature_names)
    for i, name in enumerate(feature_names):
        # Simple uniform importance as IsolationForest doesn't provide this directly
        importances[name] = 1.0 / n_features
    
    return importances


def summarize_features(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create summary statistics of engineered features.
    
    Args:
        df: DataFrame with all features
        
    Returns:
        Summary dictionary
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    summary = {}
    for col in numeric_cols:
        summary[col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'median': float(df[col].median())
        }
    
    return summary


def identify_feature_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify consumption profiles based on feature patterns.
    
    Args:
        df: DataFrame with all features
        
    Returns:
        DataFrame with profile labels
    """
    df = df.copy()
    
    # Define profiles based on consumption patterns
    conditions = [
        # Night shifter: High night usage
        (df.get('night_day_ratio', 0) > 1.5),
        # Weekend heavy: High weekend usage
        (df.get('weekend_weekday_ratio', 0) > 1.5),
        # Low consumer: Very low average usage
        (df.get('avg_consumption', 999) < df.get('avg_consumption', 0).quantile(0.1)),
        # High consumer: Very high average usage
        (df.get('avg_consumption', 0) > df.get('avg_consumption', 0).quantile(0.9)),
        # Irregular: High variability
        (df.get('cv_consumption', 0) > df.get('cv_consumption', 0).quantile(0.9))
    ]
    
    profiles = ['Night Shifter', 'Weekend Heavy', 'Low Consumer', 'High Consumer', 'Irregular']
    
    df['consumption_profile'] = 'Normal'
    for condition, profile in zip(conditions, profiles):
        try:
            df.loc[condition, 'consumption_profile'] = profile
        except:
            pass
    
    return df
