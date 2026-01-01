"""
Data preprocessing utilities for heart disease prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(filepath):
    """
    Load heart disease dataset
    
    Args:
        filepath: Path to the data file
        
    Returns:
        DataFrame with loaded data
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded successfully from {filepath}")
        logger.info(f"Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def clean_data(df):
    """
    Clean and preprocess the dataset
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Handle missing values
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype in ['float64', 'int64']:
                # Use mode for categorical-like numerical features
                if col in ['ca', 'thal']:
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                else:
                    # Use median for continuous features
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
            logger.info(f"Filled missing values in {col}")
    
    # Convert target to binary if not already
    if 'target' in df_clean.columns:
        df_clean['target'] = (df_clean['target'] > 0).astype(int)
    
    logger.info("Data cleaning completed")
    logger.info(f"Cleaned shape: {df_clean.shape}")
    logger.info(f"Missing values: {df_clean.isnull().sum().sum()}")
    
    return df_clean


def prepare_features(df):
    """
    Separate features and target
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    X = df.drop('target', axis=1)
    y = df['target']
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    
    return X, y


def get_scaler(X_train):
    """
    Fit and return a StandardScaler
    
    Args:
        X_train: Training features
        
    Returns:
        Fitted StandardScaler
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    logger.info("Scaler fitted on training data")
    return scaler


def scale_features(X, scaler):
    """
    Scale features using provided scaler
    
    Args:
        X: Features to scale
        scaler: Fitted scaler
        
    Returns:
        Scaled features as DataFrame
    """
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    return X_scaled_df


if __name__ == "__main__":
    # Example usage
    data_path = Path("../data/processed/heart_disease_clean.csv")
    if data_path.exists():
        df = load_data(data_path)
        df_clean = clean_data(df)
        X, y = prepare_features(df_clean)
        print(f"Features: {list(X.columns)}")
        print(f"Target distribution:\n{y.value_counts()}")
