"""
Data utility functions for GovAI Electricity Theft Detection System.
Provides helpers for data validation, file I/O, and region classification.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


# Define regions/zones for classification (London ACORN-based)
ZONES = {
    'Zone A': ['ACORN-A', 'ACORN-B', 'ACORN-C'],
    'Zone B': ['ACORN-D', 'ACORN-E', 'ACORN-F'],
    'Zone C': ['ACORN-G', 'ACORN-H', 'ACORN-I'],
    'Zone D': ['ACORN-J', 'ACORN-K', 'ACORN-L'],
    'Zone E': ['ACORN-M', 'ACORN-N', 'ACORN-O', 'ACORN-P', 'ACORN-Q'],
    'Zone F': ['ACORN-U', 'Affluent', 'Comfortable', 'Adversity']
}

# India state-based zones
INDIA_ZONES = {
    'North India': ['UP', 'DL', 'HR', 'PB', 'RJ', 'UK', 'HP', 'JK', 'CH', 'LD'],
    'West India': ['MH', 'GJ', 'GO', 'DD', 'DN'],
    'South India': ['KA', 'TN', 'KL', 'AP', 'TS', 'PY'],
    'East India': ['WB', 'BR', 'JH', 'OR', 'AS', 'SK', 'TR', 'MN', 'NL', 'ML', 'MZ', 'AR'],
    'Central India': ['MP', 'CG']
}

# Indian state full names mapping
INDIA_STATE_NAMES = {
    'UP': 'Uttar Pradesh', 'MH': 'Maharashtra', 'DL': 'Delhi', 'KA': 'Karnataka',
    'GJ': 'Gujarat', 'TN': 'Tamil Nadu', 'WB': 'West Bengal', 'RJ': 'Rajasthan',
    'MP': 'Madhya Pradesh', 'AP': 'Andhra Pradesh', 'TS': 'Telangana', 'KL': 'Kerala',
    'BR': 'Bihar', 'HR': 'Haryana', 'PB': 'Punjab', 'OR': 'Odisha', 'AS': 'Assam',
    'JH': 'Jharkhand', 'UK': 'Uttarakhand', 'HP': 'Himachal Pradesh', 'CG': 'Chhattisgarh',
    'JK': 'Jammu & Kashmir', 'GO': 'Goa', 'TR': 'Tripura', 'MN': 'Manipur', 'ML': 'Meghalaya',
    'NL': 'Nagaland', 'MZ': 'Mizoram', 'AR': 'Arunachal Pradesh', 'SK': 'Sikkim',
    'CH': 'Chandigarh', 'PY': 'Puducherry', 'DD': 'Daman & Diu', 'DN': 'Dadra & Nagar Haveli',
    'LD': 'Ladakh'
}


def validate_csv_columns(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, Any]:
    """
    Validate that required columns exist in the DataFrame.
    
    Args:
        df: Input DataFrame
        required_columns: List of required column names
        
    Returns:
        Dict with 'valid' boolean and 'missing' list of missing columns
    """
    missing = [col for col in required_columns if col not in df.columns]
    return {
        'valid': len(missing) == 0,
        'missing': missing,
        'available': list(df.columns)
    }


def classify_region(acorn_group: str) -> str:
    """
    Classify ACORN group into a zone/region (London data).
    
    Args:
        acorn_group: ACORN classification string
        
    Returns:
        Zone identifier (e.g., 'Zone A')
    """
    if pd.isna(acorn_group) or acorn_group is None:
        return 'Zone F'  # Default zone for unknown
    
    acorn_str = str(acorn_group).strip()
    
    for zone, groups in ZONES.items():
        for group in groups:
            if group.lower() in acorn_str.lower():
                return zone
    
    return 'Zone F'  # Default zone


def classify_india_region(state_code: str) -> str:
    """
    Classify Indian state code into a regional zone.
    
    Args:
        state_code: Indian state code (e.g., 'UP', 'MH', 'DL')
        
    Returns:
        Regional zone (e.g., 'North India', 'South India')
    """
    if pd.isna(state_code) or state_code is None:
        return 'Unknown'
    
    state_str = str(state_code).strip().upper()
    
    for zone, states in INDIA_ZONES.items():
        if state_str in states:
            return zone
    
    # If exact match not found, try partial match
    for zone, states in INDIA_ZONES.items():
        for state in states:
            if state in state_str or state_str in state:
                return zone
    
    return 'Unknown'


def get_india_state_name(state_code: str) -> str:
    """
    Get full state name from state code.
    
    Args:
        state_code: Indian state code (e.g., 'UP')
        
    Returns:
        Full state name (e.g., 'Uttar Pradesh')
    """
    if pd.isna(state_code) or state_code is None:
        return 'Unknown'
    
    state_str = str(state_code).strip().upper()
    return INDIA_STATE_NAMES.get(state_str, state_str)


def get_time_of_day(hour: int) -> str:
    """
    Classify hour into time of day category.
    
    Args:
        hour: Hour of day (0-23)
        
    Returns:
        Time category string
    """
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    elif 18 <= hour < 22:
        return 'evening'
    else:
        return 'night'


def is_peak_hour(hour: int) -> bool:
    """
    Check if hour is a peak consumption hour.
    
    Peak hours: 7-9 AM and 6-9 PM
    
    Args:
        hour: Hour of day (0-23)
        
    Returns:
        Boolean indicating peak hour
    """
    return (7 <= hour <= 9) or (18 <= hour <= 21)


def get_anomaly_status(score: float) -> str:
    """
    Convert anomaly score to human-readable status.
    
    Args:
        score: Anomaly score (0-1, higher = more anomalous)
        
    Returns:
        Status string
    """
    if score >= 0.9:
        return 'High Risk'
    elif score >= 0.75:
        return 'Suspicious'
    elif score >= 0.5:
        return 'Review Needed'
    else:
        return 'Normal'


def get_status_color(status: str) -> str:
    """
    Get color code for status visualization.
    
    Args:
        status: Status string
        
    Returns:
        Hex color code
    """
    colors = {
        'High Risk': '#DC2626',      # Red
        'Suspicious': '#F59E0B',      # Amber
        'Review Needed': '#3B82F6',   # Blue
        'Normal': '#10B981'           # Green
    }
    return colors.get(status, '#6B7280')  # Gray default


def save_json(data: Any, filepath: str) -> bool:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Output file path
        
    Returns:
        Success boolean
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"Error saving JSON: {e}")
        return False


def load_json(filepath: str) -> Optional[Any]:
    """
    Load data from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data or None if error
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None


def format_timestamp(dt: datetime) -> str:
    """
    Format datetime for display.
    
    Args:
        dt: Datetime object
        
    Returns:
        Formatted string
    """
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def generate_ai_explanation(consumer_data: Dict[str, Any]) -> str:
    """
    Generate human-readable AI explanation for anomaly detection.
    
    Args:
        consumer_data: Consumer analysis data
        
    Returns:
        Explanation text
    """
    explanations = []
    
    score = consumer_data.get('anomaly_score', 0)
    avg_usage = consumer_data.get('avg_usage', 0)
    night_day_ratio = consumer_data.get('night_day_ratio', 1)
    weekend_ratio = consumer_data.get('weekend_ratio', 1)
    std_dev = consumer_data.get('std_deviation', 0)
    
    if score >= 0.9:
        explanations.append("Detected highly irregular consumption pattern.")
    elif score >= 0.75:
        explanations.append("Detected suspicious consumption behavior.")
    
    if night_day_ratio > 2:
        explanations.append("Unusually high nighttime consumption compared to daytime.")
    elif night_day_ratio < 0.2:
        explanations.append("Very low nighttime consumption which may indicate meter bypass during billing periods.")
    
    if weekend_ratio > 2:
        explanations.append("Significant spike in weekend usage compared to weekdays.")
    elif weekend_ratio < 0.3:
        explanations.append("Weekend usage abnormally low compared to weekday patterns.")
    
    if std_dev > avg_usage * 2:
        explanations.append("High variability in consumption suggests irregular usage patterns.")
    
    if avg_usage < 0.5:
        explanations.append("Consistently low readings may indicate meter tampering.")
    
    if not explanations:
        explanations.append("Consumption pattern deviates from normal behavior based on ML analysis.")
    
    return " ".join(explanations)


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Dict with mean, std, min, max, median
    """
    arr = np.array(values)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'median': float(np.median(arr))
    }
