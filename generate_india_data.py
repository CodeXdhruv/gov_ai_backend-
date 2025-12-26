"""
India-Specific Smart Meter Data Generator for GovAI Electricity Theft Detection System.
Generates realistic smart meter data with Indian consumer IDs, DISCOM codes, and region patterns.

Real Data Sources for India:
1. Kaggle: "Smart Meter Data India" (Mathura & Bareilly, UP) - https://www.kaggle.com/datasets/
2. Kaggle: "Smart Energy Meters in Bangalore India"
3. NITI Aayog India Climate & Energy Dashboard (ICED) - https://iced.niti.gov.in/
4. Open Government Data Platform India - https://data.gov.in/
5. National Smart Grid Mission Dashboard - https://nsgm.gov.in/
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional, List, Dict
import os
import random

# Indian State-wise DISCOM Information
INDIAN_DISCOMS = {
    'UP': {  # Uttar Pradesh
        'discoms': ['UPPCL', 'DVVNL', 'MVVNL', 'PVVNL', 'PUVVNL', 'KESCO'],
        'cities': ['Lucknow', 'Kanpur', 'Varanasi', 'Agra', 'Meerut', 'Prayagraj', 'Mathura', 'Bareilly'],
        'prefix': 'UP'
    },
    'MH': {  # Maharashtra
        'discoms': ['MSEDCL', 'BEST', 'TPDDL', 'AEML'],
        'cities': ['Mumbai', 'Pune', 'Nagpur', 'Nashik', 'Thane', 'Aurangabad'],
        'prefix': 'MH'
    },
    'GJ': {  # Gujarat
        'discoms': ['UGVCL', 'DGVCL', 'MGVCL', 'PGVCL', 'TORRENT'],
        'cities': ['Ahmedabad', 'Surat', 'Vadodara', 'Rajkot', 'Gandhinagar'],
        'prefix': 'GJ'
    },
    'RJ': {  # Rajasthan
        'discoms': ['JVVNL', 'AVVNL', 'JDVVNL'],
        'cities': ['Jaipur', 'Jodhpur', 'Udaipur', 'Kota', 'Ajmer', 'Bikaner'],
        'prefix': 'RJ'
    },
    'MP': {  # Madhya Pradesh
        'discoms': ['MPPKVVCL', 'MPMKVVCL', 'MPWZ', 'MPCZ'],
        'cities': ['Bhopal', 'Indore', 'Gwalior', 'Jabalpur', 'Ujjain'],
        'prefix': 'MP'
    },
    'DL': {  # Delhi
        'discoms': ['TPDDL', 'BSES-R', 'BSES-Y', 'NDMC', 'NDPL'],
        'cities': ['New Delhi', 'North Delhi', 'South Delhi', 'East Delhi', 'West Delhi'],
        'prefix': 'DL'
    },
    'KA': {  # Karnataka
        'discoms': ['BESCOM', 'MESCOM', 'HESCOM', 'GESCOM', 'CESC'],
        'cities': ['Bengaluru', 'Mysuru', 'Mangaluru', 'Hubli', 'Belagavi'],
        'prefix': 'KA'
    },
    'TN': {  # Tamil Nadu
        'discoms': ['TANGEDCO', 'TNEB'],
        'cities': ['Chennai', 'Coimbatore', 'Madurai', 'Tiruchirappalli', 'Salem'],
        'prefix': 'TN'
    },
    'WB': {  # West Bengal
        'discoms': ['WBSEDCL', 'CESC'],
        'cities': ['Kolkata', 'Howrah', 'Durgapur', 'Siliguri', 'Asansol'],
        'prefix': 'WB'
    },
    'AP': {  # Andhra Pradesh
        'discoms': ['APSPDCL', 'APEPDCL'],
        'cities': ['Visakhapatnam', 'Vijayawada', 'Guntur', 'Tirupati', 'Nellore'],
        'prefix': 'AP'
    },
    'TS': {  # Telangana
        'discoms': ['TSSPDCL', 'TSNPDCL'],
        'cities': ['Hyderabad', 'Warangal', 'Nizamabad', 'Karimnagar', 'Khammam'],
        'prefix': 'TS'
    },
    'KL': {  # Kerala
        'discoms': ['KSEB'],
        'cities': ['Thiruvananthapuram', 'Kochi', 'Kozhikode', 'Thrissur', 'Kannur'],
        'prefix': 'KL'
    },
    'HR': {  # Haryana
        'discoms': ['UHBVN', 'DHBVN'],
        'cities': ['Gurugram', 'Faridabad', 'Panipat', 'Ambala', 'Hisar'],
        'prefix': 'HR'
    },
    'PB': {  # Punjab
        'discoms': ['PSPCL'],
        'cities': ['Ludhiana', 'Amritsar', 'Jalandhar', 'Patiala', 'Bathinda'],
        'prefix': 'PB'
    },
    'BR': {  # Bihar
        'discoms': ['NBPDCL', 'SBPDCL'],
        'cities': ['Patna', 'Gaya', 'Muzaffarpur', 'Bhagalpur', 'Darbhanga'],
        'prefix': 'BR'
    },
    'AS': {  # Assam
        'discoms': ['APDCL'],
        'cities': ['Guwahati', 'Silchar', 'Dibrugarh', 'Jorhat', 'Tezpur'],
        'prefix': 'AS'
    }
}

# Consumer Categories as per Indian electricity boards
CONSUMER_CATEGORIES = {
    'DOM': 'Domestic',
    'COM': 'Commercial', 
    'IND': 'Industrial',
    'AGR': 'Agricultural',
    'GOV': 'Government',
    'PUB': 'Public Lighting'
}


def generate_indian_consumer_id(state: str, discom: str, category: str, index: int) -> str:
    """
    Generate realistic Indian consumer ID.
    Format: STATE-DISCOM-CATEGORY-XXXXXXXX
    Example: UP-UPPCL-DOM-00001234
    """
    return f"{state}-{discom}-{category}-{str(index).zfill(8)}"


def generate_meter_number(state: str, index: int) -> str:
    """
    Generate Indian smart meter number.
    Format: SM-STATE-YYYYMM-XXXXXX (SM = Smart Meter, installation date, serial)
    """
    year = random.randint(2020, 2024)
    month = random.randint(1, 12)
    return f"SM-{state}-{year}{str(month).zfill(2)}-{str(index).zfill(6)}"


def generate_indian_consumption(
    hours: int,
    consumer_type: str = 'DOM',
    season: str = 'summer',
    base_load: float = 0.3
) -> np.ndarray:
    """
    Generate consumption pattern based on Indian usage patterns.
    
    Indian-specific patterns:
    - Peak hours: 6-9 PM (evening peak due to lighting/AC)
    - Agricultural pumps: Early morning/late evening
    - Summer: Higher AC usage (April-June)
    - Monsoon: Moderate usage
    - Winter: Lower usage (except North India heating)
    """
    readings = hours * 2  # Half-hourly readings
    consumption = np.zeros(readings)
    
    # Season multiplier (Indian climate)
    season_multiplier = {
        'summer': 1.8,      # High AC usage
        'monsoon': 1.2,     # Moderate
        'winter': 1.0,      # Low (heating in North)
        'spring': 1.1
    }.get(season, 1.0)
    
    # Consumer type base loads (typical Indian households)
    type_base = {
        'DOM': 0.25,        # Domestic: ~6 kWh/day avg
        'COM': 0.8,         # Commercial: Higher
        'IND': 2.5,         # Industrial: Much higher
        'AGR': 1.2,         # Agricultural: Pump loads
        'GOV': 0.6,         # Government offices
        'PUB': 0.15         # Street lights
    }.get(consumer_type, base_load)
    
    for i in range(readings):
        hour = (i // 2) % 24
        day = i // 48
        
        value = type_base * season_multiplier
        
        if consumer_type == 'DOM':
            # Indian domestic pattern
            if 6 <= hour <= 8:          # Morning rush
                value *= 2.0
            elif 18 <= hour <= 22:      # Evening peak (TV, lights, AC)
                value *= 3.0
            elif 12 <= hour <= 15:      # Afternoon (cooking, AC)
                value *= 1.8
            elif 23 <= hour or hour < 5:  # Night
                value *= 0.3
                
        elif consumer_type == 'AGR':
            # Agricultural pump pattern
            if 5 <= hour <= 7 or 17 <= hour <= 19:  # Irrigation times
                value *= 4.0
            else:
                value *= 0.1
                
        elif consumer_type == 'COM':
            # Commercial pattern (shops/offices)
            if 10 <= hour <= 21:        # Business hours (Indian shops)
                value *= 2.0
            else:
                value *= 0.2
                
        elif consumer_type == 'IND':
            # Industrial (shift-based)
            if 6 <= hour <= 22:         # Working hours
                value *= 1.5
            else:
                value *= 0.4
                
        elif consumer_type == 'PUB':
            # Street lighting
            if 18 <= hour or hour <= 6:
                value *= 1.0
            else:
                value *= 0.0
        
        # Weekend adjustment (lower commercial, stable domestic)
        if day % 7 >= 5:
            if consumer_type in ['COM', 'IND', 'GOV']:
                value *= 0.3
            else:
                value *= 1.1
        
        # Random variation
        value *= (1 + np.random.normal(0, 0.15))
        consumption[i] = max(0, round(value, 4))
    
    return consumption


def generate_anomalous_indian_consumption(
    hours: int,
    anomaly_type: str = 'theft',
    consumer_type: str = 'DOM'
) -> np.ndarray:
    """
    Generate anomalous consumption patterns specific to Indian context.
    
    Common theft methods in India:
    - Direct hooking (bypassing meter)
    - Meter tampering
    - Neutral wire manipulation
    - Magnet interference
    """
    readings = hours * 2
    
    if anomaly_type == 'direct_hook':
        # Direct hooking - very low readings despite usage
        consumption = np.random.uniform(0.01, 0.08, readings)
        # Occasional spikes when forgetting to disconnect
        spike_indices = np.random.choice(readings, size=readings // 15, replace=False)
        consumption[spike_indices] = np.random.uniform(0.5, 1.5, len(spike_indices))
    
    elif anomaly_type == 'meter_slow':
        # Meter slowing (magnet/tampering) - reduces by 30-60%
        normal = generate_indian_consumption(hours, consumer_type)
        consumption = normal * np.random.uniform(0.4, 0.7)
    
    elif anomaly_type == 'neutral_bypass':
        # Neutral wire bypass - zero during specific periods
        consumption = generate_indian_consumption(hours, consumer_type)
        # Zero out evening peak (when most usage)
        for i in range(readings):
            hour = (i // 2) % 24
            if 18 <= hour <= 22:
                consumption[i] *= np.random.uniform(0, 0.2)
    
    elif anomaly_type == 'agricultural_theft':
        # Common in rural India - unauthorized pump connections
        consumption = np.zeros(readings)
        for i in range(readings):
            hour = (i // 2) % 24
            # Irregular pump usage at odd hours
            if np.random.random() < 0.25:
                consumption[i] = np.random.uniform(1.5, 4.0)
            else:
                consumption[i] = np.random.uniform(0, 0.05)
    
    elif anomaly_type == 'commercial_bypass':
        # Commercial establishment underreporting
        normal = generate_indian_consumption(hours, 'COM')
        consumption = normal * np.random.uniform(0.2, 0.4)
    
    else:
        # Generic theft pattern
        consumption = np.random.uniform(0.01, 0.15, readings)
    
    return consumption


def generate_india_dataset(
    n_consumers: int = 500,
    days: int = 30,
    anomaly_ratio: float = 0.08,
    states: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    include_metadata: bool = True
) -> pd.DataFrame:
    """
    Generate complete India-specific smart meter dataset.
    
    Args:
        n_consumers: Number of consumers
        days: Number of days of data
        anomaly_ratio: Proportion of anomalous consumers (theft cases)
        states: List of states to include (None = all states)
        output_path: Optional CSV output path
        include_metadata: Include additional Indian-specific columns
    
    Returns:
        DataFrame with Indian smart meter data
    """
    hours = days * 24
    n_anomalies = int(n_consumers * anomaly_ratio)
    n_normal = n_consumers - n_anomalies
    
    if states is None:
        states = list(INDIAN_DISCOMS.keys())
    
    all_data = []
    consumer_metadata = []
    
    # Current date for realistic timestamps
    start_date = datetime(2024, 10, 1)  # Recent data
    
    # Determine season based on start date
    month = start_date.month
    if month in [4, 5, 6]:
        season = 'summer'
    elif month in [7, 8, 9]:
        season = 'monsoon'
    elif month in [10, 11, 12, 1, 2]:
        season = 'winter'
    else:
        season = 'spring'
    
    consumer_types = ['DOM'] * 60 + ['COM'] * 20 + ['IND'] * 10 + ['AGR'] * 8 + ['GOV'] * 2
    anomaly_types = ['direct_hook', 'meter_slow', 'neutral_bypass', 'agricultural_theft', 'commercial_bypass']
    
    print(f"Generating {n_normal} normal Indian consumers...")
    
    for i in range(n_normal):
        state = np.random.choice(states)
        state_info = INDIAN_DISCOMS[state]
        discom = np.random.choice(state_info['discoms'])
        city = np.random.choice(state_info['cities'])
        consumer_type = np.random.choice(consumer_types)
        
        consumer_id = generate_indian_consumer_id(state, discom, consumer_type, i)
        meter_no = generate_meter_number(state, i)
        
        consumption = generate_indian_consumption(hours, consumer_type, season)
        
        for j, value in enumerate(consumption):
            timestamp = start_date + timedelta(minutes=30 * j)
            record = {
                'Consumer_ID': consumer_id,
                'Meter_Number': meter_no,
                'DateTime': timestamp,
                'KWH': round(value, 4),
                'State': state,
                'DISCOM': discom,
                'City': city,
                'Consumer_Type': consumer_type,
                'Tariff_Category': CONSUMER_CATEGORIES.get(consumer_type, 'Domestic')
            }
            all_data.append(record)
        
        consumer_metadata.append({
            'Consumer_ID': consumer_id,
            'Is_Anomaly': False,
            'Anomaly_Type': None
        })
    
    print(f"Generating {n_anomalies} anomalous consumers (theft cases)...")
    
    for i in range(n_anomalies):
        state = np.random.choice(states)
        state_info = INDIAN_DISCOMS[state]
        discom = np.random.choice(state_info['discoms'])
        city = np.random.choice(state_info['cities'])
        consumer_type = np.random.choice(['DOM', 'COM', 'AGR'])  # Most theft in these categories
        anomaly_type = np.random.choice(anomaly_types)
        
        consumer_id = generate_indian_consumer_id(state, discom, consumer_type, n_normal + i)
        meter_no = generate_meter_number(state, n_normal + i)
        
        consumption = generate_anomalous_indian_consumption(hours, anomaly_type, consumer_type)
        
        for j, value in enumerate(consumption):
            timestamp = start_date + timedelta(minutes=30 * j)
            record = {
                'Consumer_ID': consumer_id,
                'Meter_Number': meter_no,
                'DateTime': timestamp,
                'KWH': round(value, 4),
                'State': state,
                'DISCOM': discom,
                'City': city,
                'Consumer_Type': consumer_type,
                'Tariff_Category': CONSUMER_CATEGORIES.get(consumer_type, 'Domestic')
            }
            all_data.append(record)
        
        consumer_metadata.append({
            'Consumer_ID': consumer_id,
            'Is_Anomaly': True,
            'Anomaly_Type': anomaly_type
        })
    
    print("Creating DataFrame...")
    df = pd.DataFrame(all_data)
    
    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    if output_path:
        print(f"Saving to {output_path}...")
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} records")
        
        # Also save ground truth for validation
        if include_metadata:
            metadata_df = pd.DataFrame(consumer_metadata)
            metadata_path = output_path.replace('.csv', '_ground_truth.csv')
            metadata_df.to_csv(metadata_path, index=False)
            print(f"Saved ground truth to {metadata_path}")
    
    return df


def generate_quick_india_sample(
    n_consumers: int = 100,
    days: int = 7,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """Generate smaller India-specific sample for quick testing."""
    return generate_india_dataset(
        n_consumers=n_consumers,
        days=days,
        anomaly_ratio=0.10,
        states=['UP', 'MH', 'DL', 'KA', 'GJ'],  # Major states
        output_path=output_path
    )


if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    # Quick sample for development
    quick_path = os.path.join(output_dir, 'india_sample_quick.csv')
    generate_quick_india_sample(n_consumers=100, days=7, output_path=quick_path)
    
    # Larger sample for demo
    demo_path = os.path.join(output_dir, 'india_sample_demo.csv')
    generate_india_dataset(n_consumers=500, days=30, output_path=demo_path)
    
    print("\n" + "="*60)
    print("India-specific sample data generation complete!")
    print("="*60)
    print("\nReal Data Sources for India:")
    print("1. Kaggle: Smart Meter Data India (Mathura & Bareilly)")
    print("   https://www.kaggle.com/datasets/")
    print("2. NITI Aayog ICED Dashboard:")
    print("   https://iced.niti.gov.in/")
    print("3. Open Government Data Platform:")
    print("   https://data.gov.in/")
    print("4. National Smart Grid Mission:")
    print("   https://nsgm.gov.in/")
    print("="*60)
