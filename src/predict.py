"""
Prediction functions for GAPLN-PGC56 models.

This module provides functions to load models and make predictions
for lymph node metastasis at stations 5 and 6.
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from .preprocessing import preprocess_data, validate_input_features


def load_model(model_name):
    """
    Load XGBoost model from file.
    
    Parameters:
    -----------
    model_name : str
        Model name, either 'ln5' or 'ln6'
    
    Returns:
    --------
    xgb.XGBClassifier
        Loaded XGBoost model
    """
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(os.path.dirname(current_dir), 'models')
    
    # Construct model path
    if model_name == 'ln5':
        model_path = os.path.join(models_dir, 'xgb_model_5.json')
    elif model_name == 'ln6':
        model_path = os.path.join(models_dir, 'xgb_model_6.json')
    else:
        raise ValueError(f"Unknown model name: {model_name}. Must be 'ln5' or 'ln6'.")
    
    # Load model
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    return model


def get_model_features(model_name):
    """
    Get the list of features required by each model.
    
    Parameters:
    -----------
    model_name : str
        Model name, either 'ln5' or 'ln6'
    
    Returns:
    --------
    list
        List of feature names required by the model
    """
    if model_name == 'ln5':
        # LN_5 model uses 9 features
        return [
            'Tumor_Location',
            'Distal_Resection_Margin',
            'Age',
            'Tumor_Wall_Invasion',
            'CEA_B',
            'AFP_B',
            'Tumor_Size',
            'CA19_9',
            'Cell_Differentiation'
        ]
    elif model_name == 'ln6':
        # LN_6 model uses 8 features
        return [
            'Proximal_Resection_Margin',
            'Tumor_Location',
            'Tumor_Wall_Invasion',
            'Tumor_Type_3A',
            'CEA_B',
            'Tumor_Size_Mucosal_1_3',
            'Blood_Type_ABO',
            'Sex'
        ]
    else:
        raise ValueError(f"Unknown model name: {model_name}. Must be 'ln5' or 'ln6'.")


def predict_ln5(data):
    """
    Predict lymph node metastasis at Station 5.
    
    Parameters:
    -----------
    data : pd.DataFrame, dict, or pd.Series
        Patient data with all required features
    
    Returns:
    --------
    dict or list of dict
        Prediction results containing:
        - prediction: Binary prediction (0 or 1)
        - risk_score: Probability of metastasis (0-1)
    
    Example:
    --------
    >>> patient = {'Age': 65, 'Sex': 1, 'CEA': 3.5, ...}
    >>> result = predict_ln5(patient)
    >>> print(f"Prediction: {result['prediction']}, Risk Score: {result['risk_score']:.3f}")
    """
    # Validate input
    is_valid, missing = validate_input_features(data)
    if not is_valid:
        raise ValueError(f"Missing required features: {missing}")
    
    # Preprocess data
    processed_data = preprocess_data(data, model_name='ln5')
    
    # Load model
    model = load_model('ln5')
    
    # Get required features
    model_features = get_model_features('ln5')
    X_input = processed_data[model_features]
    
    # Make prediction
    predictions = model.predict(X_input)
    probabilities = model.predict_proba(X_input)[:, 1]  # Probability of class 1
    
    # Format results
    results = []
    for i in range(len(X_input)):
        result = {
            'prediction': int(predictions[i]),
            'risk_score': float(probabilities[i])
        }
        results.append(result)
    
    # Return single result if input was single patient
    if isinstance(data, (dict, pd.Series)):
        return results[0]
    else:
        return results


def predict_ln6(data):
    """
    Predict lymph node metastasis at Station 6.
    
    Parameters:
    -----------
    data : pd.DataFrame, dict, or pd.Series
        Patient data with all required features
    
    Returns:
    --------
    dict or list of dict
        Prediction results containing:
        - prediction: Binary prediction (0 or 1)
        - risk_score: Probability of metastasis (0-1)
    
    Example:
    --------
    >>> patient = {'Age': 65, 'Sex': 1, 'CEA': 3.5, ...}
    >>> result = predict_ln6(patient)
    >>> print(f"Prediction: {result['prediction']}, Risk Score: {result['risk_score']:.3f}")
    """
    # Validate input
    is_valid, missing = validate_input_features(data)
    if not is_valid:
        raise ValueError(f"Missing required features: {missing}")
    
    # Preprocess data
    processed_data = preprocess_data(data, model_name='ln6')
    
    # Load model
    model = load_model('ln6')
    
    # Get required features
    model_features = get_model_features('ln6')
    X_input = processed_data[model_features]
    
    # Make prediction
    predictions = model.predict(X_input)
    probabilities = model.predict_proba(X_input)[:, 1]  # Probability of class 1
    
    # Format results
    results = []
    for i in range(len(X_input)):
        result = {
            'prediction': int(predictions[i]),
            'risk_score': float(probabilities[i])
        }
        results.append(result)
    
    # Return single result if input was single patient
    if isinstance(data, (dict, pd.Series)):
        return results[0]
    else:
        return results


def predict_both(data):
    """
    Predict lymph node metastasis at both Station 5 and Station 6.
    
    This is a convenience function that predicts both models at once.
    
    Parameters:
    -----------
    data : pd.DataFrame, dict, or pd.Series
        Patient data with all required features
    
    Returns:
    --------
    dict or list of dict
        Prediction results containing:
        - ln5: Station 5 prediction results (prediction, risk_score)
        - ln6: Station 6 prediction results (prediction, risk_score)
    
    Example:
    --------
    >>> patient = {'Age': 65, 'Sex': 1, 'CEA': 3.5, ...}
    >>> results = predict_both(patient)
    >>> print(f"LN5 - Prediction: {results['ln5']['prediction']}, Risk: {results['ln5']['risk_score']:.3f}")
    >>> print(f"LN6 - Prediction: {results['ln6']['prediction']}, Risk: {results['ln6']['risk_score']:.3f}")
    """
    # Validate input
    is_valid, missing = validate_input_features(data)
    if not is_valid:
        raise ValueError(f"Missing required features: {missing}")
    
    # Predict both models
    ln5_results = predict_ln5(data)
    ln6_results = predict_ln6(data)
    
    # Format combined results
    if isinstance(data, (dict, pd.Series)):
        # Single patient
        return {
            'ln5': ln5_results,
            'ln6': ln6_results
        }
    else:
        # Multiple patients
        combined_results = []
        for i in range(len(ln5_results)):
            combined_results.append({
                'ln5': ln5_results[i],
                'ln6': ln6_results[i]
            })
        return combined_results


if __name__ == "__main__":
    # Example usage
    print("Testing prediction functions...\n")
    
    # Example patient data
    example_patient = {
        'Age': 65,
        'Sex': 1,
        'Blood_Type_ABO': 1,
        'CEA': 3.5,
        'CA19_9': 25.0,
        'AFP': 2.1,
        'Tumor_Size': 4.5,
        'Tumor_Location': 4,
        'Circumferential_Extent': 5,
        'Borrmann_Type': 7,
        'Proximal_Resection_Margin': 30.0,
        'Distal_Resection_Margin': 25.0,
        'Cell_Differentiation': 1
    }
    
    print("Patient Information:")
    print(f"  Age: {example_patient['Age']} years")
    print(f"  Sex: {'Male' if example_patient['Sex'] == 1 else 'Female'}")
    print(f"  CEA: {example_patient['CEA']} ng/mL")
    print(f"  Tumor Size: {example_patient['Tumor_Size']} cm\n")
    
    # Predict LN5
    print("=" * 60)
    print("Station 5 (LN5) Prediction:")
    print("=" * 60)
    ln5_result = predict_ln5(example_patient)
    print(f"Prediction: {ln5_result['prediction']}")
    print(f"Risk Score: {ln5_result['risk_score']:.4f}\n")
    
    # Predict LN6
    print("=" * 60)
    print("Station 6 (LN6) Prediction:")
    print("=" * 60)
    ln6_result = predict_ln6(example_patient)
    print(f"Prediction: {ln6_result['prediction']}")
    print(f"Risk Score: {ln6_result['risk_score']:.4f}\n")
    
    # Predict both
    print("=" * 60)
    print("Combined Prediction (Both Stations):")
    print("=" * 60)
    both_results = predict_both(example_patient)
    print(f"LN5 - Prediction: {both_results['ln5']['prediction']}, Risk Score: {both_results['ln5']['risk_score']:.4f}")
    print(f"LN6 - Prediction: {both_results['ln6']['prediction']}, Risk Score: {both_results['ln6']['risk_score']:.4f}")