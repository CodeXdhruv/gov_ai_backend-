"""
Sample data generator for GovAI Electricity Theft Detection System.
Generates realistic smart meter data for demo and testing purposes.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional
import os


def generate_consumer_id(prefix: str = 'MAC', index: int = 0) -> str:
    """Generate a realistic consumer ID."""
    return f"{prefix}{str(index).zfill(6)}"


def generate_normal_consumption(
    hours: int,
    base_load: float = 0.3,
    peak_factor: float = 2.5,
    noise_level: float = 0.15
) -> np.ndarray:
    """
    Generate normal consumption pattern with realistic daily cycles.
    
    Args:
        hours: Number of hours to generate
        base_load: Base consumption level (kWh per half hour)
        peak_factor: Multiplier for peak hours
        noise_level: Random noise factor
        
    Returns:
        Array of consumption values
    """
    # Two readings per hour (half-hourly)
    readings = hours * 2
    consumption = np.zeros(readings)
    
    for i in range(readings):
        hour = (i // 2) % 24
        
        # Base consumption
        value = base_load
        
        # Morning peak (7-9 AM)
        if 7 <= hour <= 9:
            value *= peak_factor * 0.8
        
        # Evening peak (6-9 PM)
        elif 18 <= hour <= 21:
            value *= peak_factor
        
        # Afternoon moderate usage
        elif 12 <= hour <= 17:
            value *= 1.3
        
        # Night low usage
        elif hour < 6 or hour >= 23:
            value *= 0.4
        
        # Weekend adjustment (every 7th day pattern)
        day = i // 48
        if day % 7 >= 5:  # Weekend
            value *= 1.2
        
        # Add random noise
        value *= (1 + np.random.normal(0, noise_level))
        
        consumption[i] = max(0, value)
    
    return consumption


def generate_anomalous_consumption(
    hours: int,
    anomaly_type: str = 'theft',
    base_load: float = 0.3
) -> np.ndarray:
    """
    Generate anomalous consumption patterns.
    
    Args:
        hours: Number of hours to generate
        anomaly_type: Type of anomaly ('theft', 'bypass', 'irregular', 'spike')
        base_load: Base consumption level
        
    Returns:
        Array of consumption values
    """
    readings = hours * 2
    
    if anomaly_type == 'theft':
        # Very low readings - meter tampering
        consumption = np.random.uniform(0.01, 0.1, readings)
        # Occasional normal readings to avoid detection
        normal_periods = np.random.choice(readings, size=readings // 10, replace=False)
        consumption[normal_periods] = np.random.uniform(0.2, 0.5, len(normal_periods))
    
    elif anomaly_type == 'bypass':
        # Meter bypass - zero readings during certain periods
        consumption = generate_normal_consumption(hours, base_load)
        # Zero out billing periods (simulate bypass during meter reading)
        for day in range(hours // 24):
            if day % 30 < 5:  # First 5 days of each month
                start_idx = day * 48
                end_idx = min(start_idx + 48, readings)
                consumption[start_idx:end_idx] *= 0.1
    
    elif anomaly_type == 'irregular':
        # Highly irregular pattern - possible illegal connection
        consumption = np.zeros(readings)
        for i in range(readings):
            hour = (i // 2) % 24
            # Random spikes at unusual times
            if np.random.random() < 0.3:
                consumption[i] = np.random.uniform(0.5, 3.0)
            else:
                consumption[i] = np.random.uniform(0, 0.1)
    
    elif anomaly_type == 'spike':
        # Normal pattern with suspicious spikes
        consumption = generate_normal_consumption(hours, base_load)
        # Add random spikes (illegal heavy usage)
        spike_periods = np.random.choice(readings, size=readings // 20, replace=False)
        consumption[spike_periods] = np.random.uniform(3.0, 8.0, len(spike_periods))
    
    else:
        consumption = generate_normal_consumption(hours, base_load)
    
    return consumption


def generate_sample_dataset(
    n_consumers: int = 1000,
    days: int = 90,
    anomaly_ratio: float = 0.05,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate a complete sample dataset.
    
    Args:
        n_consumers: Number of consumers to generate
        days: Number of days of data
        anomaly_ratio: Proportion of anomalous consumers
        output_path: Optional path to save CSV
        
    Returns:
        DataFrame with generated data
    """
    hours = days * 24
    n_anomalies = int(n_consumers * anomaly_ratio)
    n_normal = n_consumers - n_anomalies
    
    all_data = []
    
    # ACORN groups for region assignment
    acorn_groups = ['ACORN-A', 'ACORN-B', 'ACORN-C', 'ACORN-D', 'ACORN-E',
                    'ACORN-F', 'ACORN-G', 'ACORN-H', 'Affluent', 'Comfortable']
    
    # Generate start date
    start_date = datetime(2023, 1, 1)
    
    anomaly_types = ['theft', 'bypass', 'irregular', 'spike']
    
    print(f"Generating {n_normal} normal consumers...")
    
    # Generate normal consumers
    for i in range(n_normal):
        consumer_id = generate_consumer_id('MAC', i)
        base_load = np.random.uniform(0.2, 0.6)
        consumption = generate_normal_consumption(hours, base_load)
        acorn = np.random.choice(acorn_groups)
        
        for j, value in enumerate(consumption):
            timestamp = start_date + timedelta(minutes=30 * j)
            all_data.append({
                'LCLid': consumer_id,
                'DateTime': timestamp,
                'KWH/hh': round(value, 4),
                'Acorn_grouped': acorn,
                'stdorToU': 'Std'
            })
    
    print(f"Generating {n_anomalies} anomalous consumers...")
    
    # Generate anomalous consumers
    for i in range(n_anomalies):
        consumer_id = generate_consumer_id('MAC', n_normal + i)
        anomaly_type = np.random.choice(anomaly_types)
        consumption = generate_anomalous_consumption(hours, anomaly_type)
        acorn = np.random.choice(acorn_groups)
        
        for j, value in enumerate(consumption):
            timestamp = start_date + timedelta(minutes=30 * j)
            all_data.append({
                'LCLid': consumer_id,
                'DateTime': timestamp,
                'KWH/hh': round(value, 4),
                'Acorn_grouped': acorn,
                'stdorToU': 'Std'
            })
    
    print("Creating DataFrame...")
    df = pd.DataFrame(all_data)
    
    # Shuffle to mix normal and anomalous
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    if output_path:
        print(f"Saving to {output_path}...")
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} records")
    
    return df


def generate_quick_sample(
    n_consumers: int = 100,
    days: int = 7,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate a smaller sample for quick testing.
    
    Args:
        n_consumers: Number of consumers
        days: Number of days
        output_path: Optional path to save
        
    Returns:
        DataFrame
    """
    return generate_sample_dataset(n_consumers, days, anomaly_ratio=0.1, output_path=output_path)


if __name__ == '__main__':
    # Generate sample dataset for testing
    output_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    # Quick sample for development
    quick_path = os.path.join(output_dir, 'sample_quick.csv')
    generate_quick_sample(n_consumers=100, days=7, output_path=quick_path)
    
    # Larger sample for demo
    demo_path = os.path.join(output_dir, 'sample_demo.csv')
    generate_sample_dataset(n_consumers=500, days=30, output_path=demo_path)
    
    print("Sample data generation complete!")
