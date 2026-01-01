#!/usr/bin/env python3
"""
Download and prepare heart disease dataset
"""

import pandas as pd
from pathlib import Path
import logging
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Column names based on UCI documentation
COLUMN_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
    'ca', 'thal', 'target'
]


def load_cleveland_data(source_path):
    """Load Cleveland dataset"""
    logger.info(f"Loading Cleveland dataset from {source_path}")
    df = pd.read_csv(source_path, names=COLUMN_NAMES, na_values='?')
    logger.info(f"Loaded {len(df)} records")
    return df


def clean_data(df):
    """Clean and preprocess data"""
    logger.info("Cleaning data...")
    
    # Handle missing values
    for col in ['ca', 'thal']:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
            logger.info(f"Filled {col} missing values with mode")
    
    # Convert target to binary
    df['target'] = (df['target'] > 0).astype(int)
    
    logger.info(f"Cleaned data shape: {df.shape}")
    logger.info(f"Target distribution:\n{df['target'].value_counts()}")
    
    return df


def main():
    # Paths
    source_dir = Path("heart+disease")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Cleveland dataset (most commonly used)
    cleveland_file = source_dir / "processed.cleveland.data"
    
    if not cleveland_file.exists():
        logger.error(f"Dataset not found at {cleveland_file}")
        logger.info("Please ensure the heart disease dataset is in the 'heart+disease' folder")
        return
    
    # Load and clean data
    df = load_cleveland_data(cleveland_file)
    df_clean = clean_data(df)
    
    # Save processed data
    output_file = output_dir / "heart_disease_clean.csv"
    df_clean.to_csv(output_file, index=False)
    logger.info(f"Processed data saved to {output_file}")
    
    # Save a copy of raw data
    raw_output = output_dir / "heart_disease_raw.csv"
    shutil.copy(cleveland_file, raw_output)
    logger.info(f"Raw data copied to {raw_output}")
    
    # Print summary
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total Records: {len(df_clean)}")
    print(f"Features: {len(df_clean.columns) - 1}")
    print(f"\nFeature List:")
    for col in df_clean.columns:
        if col != 'target':
            print(f"  - {col}")
    print(f"\nTarget Distribution:")
    print(df_clean['target'].value_counts())
    print("="*60)


if __name__ == "__main__":
    main()
