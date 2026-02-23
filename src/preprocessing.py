"""
Data preprocessing functions for GAPLN-PGC56 models.

This module provides functions to preprocess patient data before prediction,
including feature engineering, Box-Cox transformation, and standardization.

Note: This module does NOT include imputation. Users should provide complete
data or handle missing values before using these functions.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import special


def load_preprocessing_params(model_name):
    """
    Load preprocessing parameters for specified model.

    Parameters:
    -----------
    model_name : str
        Model name, either 'ln5' or 'ln6'

    Returns:
    --------
    tuple
        (boxcox_params, scaler_params)
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessing_dir = os.path.join(os.path.dirname(current_dir), 'preprocessing')

    boxcox_path = os.path.join(preprocessing_dir, f'boxcox_params_{model_name}.json')
    with open(boxcox_path, 'r') as f:
        boxcox_params = json.load(f)

    scaler_path = os.path.join(preprocessing_dir, f'scaler_params_{model_name}.json')
    with open(scaler_path, 'r') as f:
        scaler_params = json.load(f)

    return boxcox_params, scaler_params


def get_boxcox_features(model_name):
    """
    Get the list of features that need Box-Cox transformation for each model.

    Parameters:
    -----------
    model_name : str
        Model name, either 'ln5' or 'ln6'

    Returns:
    --------
    list
        List of feature names that need Box-Cox transformation
    """
    if model_name == 'ln5':
        return ['CEA', 'AFP']
    elif model_name == 'ln6':
        return ['CEA']
    else:
        raise ValueError(f"Unknown model name: {model_name}. Must be 'ln5' or 'ln6'.")


def engineer_features(data_df):
    """
    Engineer derived features from raw input features.

    Parameters:
    -----------
    data_df : pd.DataFrame
        DataFrame with raw features

    Returns:
    --------
    pd.DataFrame
        DataFrame with additional engineered features
    """
    df = data_df.copy()

    # 1. Tumor_Wall_Invasion = Circumferential_Extent (same encoding)
    if 'Circumferential_Extent' in df.columns:
        df['Tumor_Wall_Invasion'] = df['Circumferential_Extent']

    # 2. Tumor_Type_3A = Borrmann_Type (just an alias)
    if 'Borrmann_Type' in df.columns:
        df['Tumor_Type_3A'] = df['Borrmann_Type']

    # 3. Tumor_Size_Mucosal_1_3: Categorize tumor size
    # 1: >8cm, 2: 4-8cm, 3: <4cm
    if 'Tumor_Size' in df.columns:
        def categorize_tumor_size(size):
            if size > 8:
                return 1
            elif size >= 4:
                return 2
            else:
                return 3

        df['Tumor_Size_Mucosal_1_3'] = df['Tumor_Size'].apply(categorize_tumor_size)

    if 'Blood_Type_ABO' in df.columns:
        df['ABO_BT'] = df['Blood_Type_ABO']
        
    if 'Tumor_Size_Mucosal_1_3' in df.columns:
        df['Tumor_Size_Mucosal_1-3'] = df['Tumor_Size_Mucosal_1_3']

    # 4. 欄位名稱對應到模型訓練時的名稱
    if 'Distal_Resection_Margin' in df.columns:
        df['Distal_Margin'] = df['Distal_Resection_Margin']
    if 'CA19_9' in df.columns:
        df['CA199'] = df['CA19_9']
    if 'Cell_Differentiation' in df.columns:
        df['Cell_Type'] = df['Cell_Differentiation']

    return df


def apply_boxcox_transform(value, feature_name, boxcox_params):
    """
    Apply Box-Cox transformation to a single value.

    Parameters:
    -----------
    value : float
        Original feature value
    feature_name : str
        Name of the feature
    boxcox_params : dict
        Dictionary containing lambda and shift parameters

    Returns:
    --------
    float
        Transformed value
    """
    if feature_name not in boxcox_params:
        return value

    params = boxcox_params[feature_name]
    lambda_value = params['lambda']
    shift = params['shift']

    shifted_value = value + shift

    if lambda_value == 0:
        transformed = np.log(shifted_value)
    else:
        transformed = (shifted_value ** lambda_value - 1) / lambda_value

    return transformed


def apply_standardization(value, feature_name, scaler_params):
    """
    Apply Z-score standardization to a single value.

    Parameters:
    -----------
    value : float
        Original or transformed feature value
    feature_name : str
        Name of the feature
    scaler_params : dict
        Dictionary containing mean and std parameters

    Returns:
    --------
    float
        Standardized value
    """
    if feature_name not in scaler_params:
        return value

    params = scaler_params[feature_name]
    mean = params['mean']
    std = params['std']

    standardized = (value - mean) / std

    return standardized


def preprocess_single_patient(patient_data, model_name):
    """
    Preprocess a single patient's data for prediction.

    Parameters:
    -----------
    patient_data : dict or pd.Series
        Patient data with feature names as keys
    model_name : str
        Model name, either 'ln5' or 'ln6'

    Returns:
    --------
    pd.DataFrame
        Preprocessed data ready for model prediction
    """
    boxcox_params, scaler_params = load_preprocessing_params(model_name)

    if isinstance(patient_data, dict):
        data_df = pd.DataFrame([patient_data])
    elif isinstance(patient_data, pd.Series):
        data_df = pd.DataFrame([patient_data.to_dict()])
    else:
        data_df = patient_data.copy()

    # Step 1: Engineer derived features
    data_df = engineer_features(data_df)

    # Step 2: Get features that need Box-Cox transformation for this model
    boxcox_features = get_boxcox_features(model_name)

    # Step 3: Apply Box-Cox transformation and create _B features
    for feature in boxcox_features:
        if feature in data_df.columns:
            original_value = data_df[feature].values[0]
            transformed_value = apply_boxcox_transform(
                original_value,
                feature,
                boxcox_params
            )
            data_df[f'{feature}_B'] = transformed_value

    # Step 4: Apply standardization to all features
    for col in data_df.columns:
        if col in scaler_params:
            data_df[col] = apply_standardization(
                data_df[col].values[0],
                col,
                scaler_params
            )

    return data_df


def preprocess_data(data, model_name='ln5'):
    """
    Preprocess patient data for prediction.

    Parameters:
    -----------
    data : pd.DataFrame, dict, or pd.Series
        Patient data
    model_name : str, default='ln5'
        Model name, either 'ln5' or 'ln6'

    Returns:
    --------
    pd.DataFrame
        Preprocessed data ready for prediction
    """
    if isinstance(data, pd.DataFrame):
        processed_list = []
        for idx, row in data.iterrows():
            processed = preprocess_single_patient(row, model_name)
            processed_list.append(processed)
        return pd.concat(processed_list, ignore_index=True)
    else:
        return preprocess_single_patient(data, model_name)


def validate_input_features(data):
    """
    Validate that input data contains all required features.

    Parameters:
    -----------
    data : pd.DataFrame or dict
        Patient data

    Returns:
    --------
    tuple
        (is_valid, missing_features)
    """
    required_features = [
        'Age',
        'Sex',
        'Blood_Type_ABO',
        'CEA',
        'CA19_9',
        'AFP',
        'Tumor_Size',
        'Tumor_Location',
        'Circumferential_Extent',
        'Borrmann_Type',
        'Proximal_Resection_Margin',
        'Distal_Resection_Margin',
        'Cell_Differentiation'
    ]

    if isinstance(data, dict):
        available_features = set(data.keys())
    else:
        available_features = set(data.columns)

    missing_features = [f for f in required_features if f not in available_features]
    is_valid = len(missing_features) == 0

    return is_valid, missing_features


if __name__ == "__main__":
    print("Testing preprocessing functions...\n")

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

    is_valid, missing = validate_input_features(example_patient)
    if is_valid:
        print("✓ Input validation passed\n")

        print("Testing feature engineering...")
        df = pd.DataFrame([example_patient])
        df_engineered = engineer_features(df)
        print(f"✓ Original features: {len(df.columns)}")
        print(f"✓ After engineering: {len(df_engineered.columns)}")
        print(f"  - Tumor_Wall_Invasion = {df_engineered['Tumor_Wall_Invasion'].values[0]}")
        print(f"  - Tumor_Type_3A = {df_engineered['Tumor_Type_3A'].values[0]}")
        print(f"  - Tumor_Size_Mucosal_1_3 = {df_engineered['Tumor_Size_Mucosal_1_3'].values[0]}")
        print(f"  - Distal_Margin = {df_engineered['Distal_Margin'].values[0]}")
        print(f"  - CA199 = {df_engineered['CA199'].values[0]}")
        print(f"  - Cell_Type = {df_engineered['Cell_Type'].values[0]}")
        print()

        print("=" * 60)
        print("Preprocessing for LN5 model:")
        print("=" * 60)
        processed_ln5 = preprocess_data(example_patient, model_name='ln5')
        print(f"✓ LN5 preprocessing completed")
        print(f"  Output features: {processed_ln5.shape[1]}")
        print()

        print("=" * 60)
        print("Preprocessing for LN6 model:")
        print("=" * 60)
        processed_ln6 = preprocess_data(example_patient, model_name='ln6')
        print(f"✓ LN6 preprocessing completed")
        print(f"  Output features: {processed_ln6.shape[1]}")
    else:
        print(f"✗ Missing features: {missing}")