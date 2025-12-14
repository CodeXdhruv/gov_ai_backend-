"""
Data preprocessing module for GovAI Electricity Theft Detection System.
Handles data cleaning, transformation, and preparation for ML models.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from utils.data_utils import validate_csv_columns, classify_region


# Expected column mappings for different data formats
COLUMN_MAPPINGS = {
    'london_smart_meters': {
        'consumer_id': ['LCLid', 'consumer_id', 'id', 'meter_id'],
        'timestamp': ['DateTime', 'timestamp', 'datetime', 'date_time'],
        'consumption': ['KWH/hh', 'consumption_kWh', 'kwh', 'energy', 'consumption'],
        'acorn_group': ['Acorn_grouped', 'acorn', 'acorn_group', 'region']
    }
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to expected format.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with standardized column names
    """
    df_copy = df.copy()
    column_map = {}
    
    for standard_name, possible_names in COLUMN_MAPPINGS['london_smart_meters'].items():
        for col in df_copy.columns:
            if col.lower().strip() in [p.lower() for p in possible_names]:
                column_map[col] = standard_name
                break
    
    df_copy = df_copy.rename(columns=column_map)
    return df_copy


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean and preprocess raw smart meter data.
    
    Args:
        df: Raw input DataFrame
        
    Returns:
        Tuple of (cleaned DataFrame, cleaning report)
    """
    report = {
        'original_rows': len(df),
        'original_columns': len(df.columns),
        'removed_duplicates': 0,
        'removed_nulls': 0,
        'removed_negative': 0,
        'final_rows': 0
    }
    
    df_clean = df.copy()
    
    # Standardize column names
    df_clean = standardize_columns(df_clean)
    
    # Remove duplicates
    initial_len = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    report['removed_duplicates'] = initial_len - len(df_clean)
    
    # Handle missing values in consumption
    if 'consumption' in df_clean.columns:
        initial_len = len(df_clean)
        # Replace 'Null' strings with NaN
        df_clean['consumption'] = df_clean['consumption'].replace('Null', np.nan)
        df_clean['consumption'] = pd.to_numeric(df_clean['consumption'], errors='coerce')
        df_clean = df_clean.dropna(subset=['consumption'])
        report['removed_nulls'] = initial_len - len(df_clean)
        
        # Remove negative values
        initial_len = len(df_clean)
        df_clean = df_clean[df_clean['consumption'] >= 0]
        report['removed_negative'] = initial_len - len(df_clean)
    
    # Parse timestamps
    if 'timestamp' in df_clean.columns:
        df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'], errors='coerce')
        df_clean = df_clean.dropna(subset=['timestamp'])
    
    # Add region classification
    if 'acorn_group' in df_clean.columns:
        df_clean['region'] = df_clean['acorn_group'].apply(classify_region)
    else:
        df_clean['region'] = 'Zone F'
    
    report['final_rows'] = len(df_clean)
    report['cleaning_ratio'] = report['final_rows'] / report['original_rows'] if report['original_rows'] > 0 else 0
    
    return df_clean, report


def aggregate_by_consumer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate data to consumer level with time-based features.
    
    Args:
        df: Cleaned DataFrame with timestamp column
        
    Returns:
        Aggregated DataFrame with one row per consumer
    """
    # Extract time features
    if 'timestamp' in df.columns:
        df = df.copy()
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
        df['is_peak'] = df['hour'].apply(lambda x: 1 if (7 <= x <= 9) or (18 <= x <= 21) else 0)
    
    # Aggregate by consumer
    agg_funcs = {
        'consumption': ['mean', 'std', 'min', 'max', 'sum', 'count'],
        'region': 'first'
    }
    
    # Add time-based aggregations if available
    if 'is_night' in df.columns:
        agg_funcs['is_night'] = 'mean'
    if 'is_weekend' in df.columns:
        agg_funcs['is_weekend'] = 'mean'
    if 'is_peak' in df.columns:
        agg_funcs['is_peak'] = 'mean'
    
    aggregated = df.groupby('consumer_id').agg(agg_funcs)
    
    # Flatten column names
    aggregated.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                          for col in aggregated.columns]
    
    aggregated = aggregated.reset_index()
    
    # Rename columns for clarity
    column_renames = {
        'consumption_mean': 'avg_consumption',
        'consumption_std': 'std_consumption',
        'consumption_min': 'min_consumption',
        'consumption_max': 'max_consumption',
        'consumption_sum': 'total_consumption',
        'consumption_count': 'reading_count',
        'region_first': 'region',
        'is_night_mean': 'night_usage_ratio',
        'is_weekend_mean': 'weekend_readings_ratio',
        'is_peak_mean': 'peak_usage_ratio'
    }
    aggregated = aggregated.rename(columns=column_renames)
    
    return aggregated


def get_consumer_timeseries(df: pd.DataFrame, consumer_id: str) -> pd.DataFrame:
    """
    Get time series data for a specific consumer.
    
    Args:
        df: Full cleaned DataFrame
        consumer_id: Consumer identifier
        
    Returns:
        DataFrame with consumer's time series data
    """
    consumer_data = df[df['consumer_id'] == consumer_id].copy()
    
    if 'timestamp' in consumer_data.columns:
        consumer_data = consumer_data.sort_values('timestamp')
    
    return consumer_data


def calculate_consumption_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate overall statistics for the dataset.
    
    Args:
        df: Aggregated consumer DataFrame
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        'total_consumers': len(df),
        'avg_consumption_mean': float(df['avg_consumption'].mean()) if 'avg_consumption' in df.columns else 0,
        'avg_consumption_std': float(df['avg_consumption'].std()) if 'avg_consumption' in df.columns else 0,
        'regions': df['region'].value_counts().to_dict() if 'region' in df.columns else {},
        'total_readings': int(df['reading_count'].sum()) if 'reading_count' in df.columns else 0
    }
    
    return stats


def prepare_for_ml(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Prepare aggregated data for ML model input.
    
    Args:
        df: Aggregated consumer DataFrame
        
    Returns:
        Tuple of (feature DataFrame, list of feature names)
    """
    feature_columns = [
        'avg_consumption',
        'std_consumption',
        'min_consumption',
        'max_consumption',
        'night_usage_ratio',
        'weekend_readings_ratio',
        'peak_usage_ratio'
    ]
    
    # Filter to available columns
    available_features = [col for col in feature_columns if col in df.columns]
    
    # Create feature matrix
    features_df = df[available_features].copy()
    
    # Fill NaN with column means
    features_df = features_df.fillna(features_df.mean())
    
    # Replace infinite values
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.fillna(0)
    
    return features_df, available_features
