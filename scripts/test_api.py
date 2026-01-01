#!/usr/bin/env python3
"""
Test the deployed API with sample requests
"""

import requests
import json
import sys
from pathlib import Path

# Sample patient data
SAMPLE_DATA = {
    "age": 63.0,
    "sex": 1,
    "cp": 1,
    "trestbps": 145.0,
    "chol": 233.0,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150.0,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0.0,
    "thal": 6.0
}

SAMPLE_DATA_2 = {
    "age": 67.0,
    "sex": 1,
    "cp": 4,
    "trestbps": 160.0,
    "chol": 286.0,
    "fbs": 0,
    "restecg": 2,
    "thalach": 108.0,
    "exang": 1,
    "oldpeak": 1.5,
    "slope": 2,
    "ca": 3.0,
    "thal": 3.0
}


def test_health_endpoint(base_url):
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("Testing Health Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def test_root_endpoint(base_url):
    """Test root endpoint"""
    print("\n" + "="*60)
    print("Testing Root Endpoint")
    print("="*60)
    
    try:
        response = requests.get(base_url)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def test_predict_endpoint(base_url):
    """Test prediction endpoint"""
    print("\n" + "="*60)
    print("Testing Predict Endpoint")
    print("="*60)
    
    try:
        print(f"\nInput Data:")
        print(json.dumps(SAMPLE_DATA, indent=2))
        
        response = requests.post(
            f"{base_url}/predict",
            json=SAMPLE_DATA,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nPrediction Result:")
            print(f"  Prediction: {result['prediction']} ({'Disease' if result['prediction'] == 1 else 'No Disease'})")
            print(f"  Probability: {result['probability']:.4f}")
            print(f"  Risk Level: {result['risk_level']}")
            print(f"  Model Version: {result['model_version']}")
            print(f"  Timestamp: {result['timestamp']}")
        else:
            print(f"Error: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def test_batch_predict_endpoint(base_url):
    """Test batch prediction endpoint"""
    print("\n" + "="*60)
    print("Testing Batch Predict Endpoint")
    print("="*60)
    
    try:
        batch_data = [SAMPLE_DATA, SAMPLE_DATA_2]
        print(f"\nBatch Size: {len(batch_data)}")
        
        response = requests.post(
            f"{base_url}/batch_predict",
            json=batch_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nBatch Prediction Results:")
            for i, pred in enumerate(result['predictions'], 1):
                print(f"\n  Patient {i}:")
                print(f"    Prediction: {pred['prediction']} ({'Disease' if pred['prediction'] == 1 else 'No Disease'})")
                print(f"    Probability: {pred['probability']:.4f}")
                print(f"    Risk Level: {pred['risk_level']}")
        else:
            print(f"Error: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def main():
    # Get base URL from command line or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    print("\n" + "="*60)
    print(f"TESTING API AT: {base_url}")
    print("="*60)
    
    # Run tests
    results = {
        "Root Endpoint": test_root_endpoint(base_url),
        "Health Endpoint": test_health_endpoint(base_url),
        "Predict Endpoint": test_predict_endpoint(base_url),
        "Batch Predict Endpoint": test_batch_predict_endpoint(base_url)
    }
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    # Exit code
    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("="*60 + "\n")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
