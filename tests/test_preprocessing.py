"""
Unit tests for data processing and preprocessing
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import load_data, clean_data, prepare_features


class TestDataProcessing:
    """Test data processing functions"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pd.DataFrame({
            'age': [63, 67, 67, 37, 41],
            'sex': [1, 1, 1, 1, 0],
            'cp': [1, 4, 4, 3, 2],
            'trestbps': [145, 160, 120, 130, 130],
            'chol': [233, 286, 229, 250, 204],
            'fbs': [1, 0, 0, 0, 0],
            'restecg': [2, 2, 2, 0, 2],
            'thalach': [150, 108, 129, 187, 172],
            'exang': [0, 1, 1, 0, 0],
            'oldpeak': [2.3, 1.5, 2.6, 3.5, 1.4],
            'slope': [3, 2, 2, 3, 1],
            'ca': [0.0, 3.0, 2.0, 0.0, 0.0],
            'thal': [6.0, 3.0, 7.0, 3.0, 3.0],
            'target': [0, 2, 1, 0, 0]
        })
    
    def test_load_data(self, sample_data, tmp_path):
        """Test data loading"""
        # Save sample data
        test_file = tmp_path / "test_data.csv"
        sample_data.to_csv(test_file, index=False)
        
        # Load data
        loaded_data = load_data(test_file)
        
        # Assertions
        assert loaded_data is not None
        assert isinstance(loaded_data, pd.DataFrame)
        assert loaded_data.shape[0] == 5
        assert 'target' in loaded_data.columns
    
    def test_clean_data(self, sample_data):
        """Test data cleaning"""
        # Add missing values
        data_with_missing = sample_data.copy()
        data_with_missing.loc[0, 'ca'] = np.nan
        data_with_missing.loc[1, 'thal'] = np.nan
        
        # Clean data
        cleaned = clean_data(data_with_missing)
        
        # Assertions
        assert cleaned.isnull().sum().sum() == 0
        assert cleaned.shape[0] == data_with_missing.shape[0]
    
    def test_prepare_features(self, sample_data):
        """Test feature preparation"""
        X, y = prepare_features(sample_data)
        
        # Assertions
        assert X.shape[0] == sample_data.shape[0]
        assert X.shape[1] == sample_data.shape[1] - 1
        assert len(y) == sample_data.shape[0]
        assert 'target' not in X.columns
    
    def test_target_binary_conversion(self, sample_data):
        """Test target variable binary conversion"""
        cleaned = clean_data(sample_data)
        
        # Check if target is binary
        assert cleaned['target'].nunique() <= 2
        assert cleaned['target'].min() == 0
        assert cleaned['target'].max() == 1


class TestDataValidation:
    """Test data validation"""
    
    def test_feature_ranges(self):
        """Test if features are within expected ranges"""
        data = {
            'age': [63],
            'sex': [1],
            'cp': [1],
            'trestbps': [145],
            'chol': [233],
            'fbs': [1],
            'restecg': [2],
            'thalach': [150],
            'exang': [0],
            'oldpeak': [2.3],
            'slope': [3],
            'ca': [0.0],
            'thal': [6.0]
        }
        df = pd.DataFrame(data)
        
        # Check ranges
        assert 0 <= df['age'].iloc[0] <= 120
        assert df['sex'].iloc[0] in [0, 1]
        assert 0 <= df['cp'].iloc[0] <= 4
        assert df['trestbps'].iloc[0] > 0
        assert df['chol'].iloc[0] > 0
        assert df['fbs'].iloc[0] in [0, 1]
    
    def test_no_null_values_after_cleaning(self):
        """Test that cleaning removes all null values"""
        data = pd.DataFrame({
            'age': [63, 67, None],
            'sex': [1, 1, 1],
            'cp': [1, 4, 4],
            'trestbps': [145, 160, 120],
            'chol': [233, 286, 229],
            'fbs': [1, 0, 0],
            'restecg': [2, 2, 2],
            'thalach': [150, 108, 129],
            'exang': [0, 1, 1],
            'oldpeak': [2.3, 1.5, 2.6],
            'slope': [3, 2, 2],
            'ca': [0.0, None, 2.0],
            'thal': [6.0, 3.0, None],
            'target': [0, 2, 1]
        })
        
        cleaned = clean_data(data)
        assert cleaned.isnull().sum().sum() == 0


class TestFeatureScaling:
    """Test feature scaling functions"""
    
    def test_get_scaler(self):
        """Test scaler fitting"""
        from src.preprocessing import get_scaler
        
        data = pd.DataFrame({
            'age': [63, 67, 45, 55, 72],
            'trestbps': [145, 160, 120, 130, 140],
            'chol': [233, 286, 229, 250, 204]
        })
        
        scaler = get_scaler(data)
        assert scaler is not None
        assert hasattr(scaler, 'mean_')
        assert len(scaler.mean_) == 3
    
    def test_scale_features(self):
        """Test feature scaling"""
        from src.preprocessing import get_scaler, scale_features
        
        data = pd.DataFrame({
            'age': [63, 67, 45, 55, 72],
            'trestbps': [145, 160, 120, 130, 140],
            'chol': [233, 286, 229, 250, 204]
        })
        
        scaler = get_scaler(data)
        scaled = scale_features(data, scaler)
        
        assert scaled is not None
        assert isinstance(scaled, pd.DataFrame)
        assert scaled.shape == data.shape
        assert list(scaled.columns) == list(data.columns)


