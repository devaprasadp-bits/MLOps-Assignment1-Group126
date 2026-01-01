#!/usr/bin/env python3
"""
Training script for heart disease prediction model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import argparse
import logging
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report
)

import mlflow
import mlflow.sklearn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42


def load_and_prepare_data(data_path):
    """Load and prepare data for training"""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Target distribution:\n{y.value_counts()}")
    
    return X, y


def train_model(model_type, X_train, y_train, X_test, y_test):
    """Train a model and log to MLflow"""
    
    with mlflow.start_run(run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        if model_type == "logistic_regression":
            logger.info("Training Logistic Regression")
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))
            ])
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("max_iter", 1000)
            
        elif model_type == "random_forest":
            logger.info("Training Random Forest")
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                ))
            ])
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        logger.info("Training model...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1_score': f1_score(y_test, y_test_pred),
            'roc_auc': roc_auc_score(y_test, y_test_proba)
        }
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            logger.info(f"{metric_name}: {metric_value:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_roc_auc_std", cv_scores.std())
        logger.info(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        return model, metrics


def hyperparameter_tuning(X_train, y_train, X_test, y_test):
    """Perform hyperparameter tuning for Random Forest"""
    
    logger.info("Starting hyperparameter tuning...")
    
    with mlflow.start_run(run_name=f"random_forest_tuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Define parameter grid
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [5, 10, 15],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
        
        # Create base pipeline
        base_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
        ])
        
        # Grid search
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, 
            scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        
        # Log parameters
        mlflow.log_param("model_type", "RandomForest_Tuned")
        for param_name, param_value in grid_search.best_params_.items():
            mlflow.log_param(param_name, param_value)
            logger.info(f"Best {param_name}: {param_value}")
        
        # Evaluate
        y_test_pred = best_model.predict(X_test)
        y_test_proba = best_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1_score': f1_score(y_test, y_test_pred),
            'roc_auc': roc_auc_score(y_test, y_test_proba),
            'cv_best_score': grid_search.best_score_
        }
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            logger.info(f"{metric_name}: {metric_value:.4f}")
        
        # Log model
        mlflow.sklearn.log_model(best_model, "model")
        
        return best_model, metrics


def main():
    parser = argparse.ArgumentParser(description='Train heart disease prediction model')
    parser.add_argument('--data', type=str, default='data/processed/heart_disease_clean.csv',
                        help='Path to processed data')
    parser.add_argument('--model', type=str, default='random_forest',
                        choices=['logistic_regression', 'random_forest', 'all'],
                        help='Model type to train')
    parser.add_argument('--tune', action='store_true',
                        help='Perform hyperparameter tuning')
    parser.add_argument('--output', type=str, default='models',
                        help='Output directory for models')
    
    args = parser.parse_args()
    
    # Setup MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("heart_disease_prediction")
    
    # Load data
    X, y = load_and_prepare_data(args.data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")
    
    # Train models
    best_model = None
    best_score = 0
    
    if args.model == 'all':
        models = ['logistic_regression', 'random_forest']
    else:
        models = [args.model]
    
    for model_type in models:
        model, metrics = train_model(model_type, X_train, y_train, X_test, y_test)
        if metrics['roc_auc'] > best_score:
            best_score = metrics['roc_auc']
            best_model = model
    
    # Hyperparameter tuning
    if args.tune:
        tuned_model, tuned_metrics = hyperparameter_tuning(X_train, y_train, X_test, y_test)
        if tuned_metrics['roc_auc'] > best_score:
            best_model = tuned_model
            best_score = tuned_metrics['roc_auc']
    
    # Save best model
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / 'best_model.pkl'
    
    joblib.dump(best_model, model_path)
    logger.info(f"Best model saved to {model_path}")
    logger.info(f"Best ROC-AUC score: {best_score:.4f}")


if __name__ == "__main__":
    main()
