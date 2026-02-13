# GAPLN-PGC56: Gastric Cancer Lymph Node Metastasis Prediction Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange.svg)](https://xgboost.readthedocs.io/)

Machine learning models for predicting lymph node metastasis at stations 5 and 6 in proximal gastric cancer patients.

## 📄 Publication

This repository contains the reproducible code for the research presented in:

> **Evolutionary Learning-Based Prediction of No. 5 and No. 6 Lymph Node Metastasis in Proximal Gastric Cancer**  
> *Submitted to International Journal of Surgery (under review)*
> 
> **Authors:**  
> Ching-Yun Kung<sup>1,2†</sup>, Ching-Po Huang<sup>3†</sup>, Kuo-Hung Huang<sup>1,2</sup>, Chew-Wen Wu<sup>1,2</sup>, Shinn-Ying Ho<sup>3,4,5,6*</sup>, Wen-Liang Fang<sup>1,2*</sup>
>
> <sup>1</sup>Division of General Surgery, Department of Surgery, Taipei Veterans General Hospital, Taipei, Taiwan  
> <sup>2</sup>School of Medicine, National Yang Ming Chiao Tung University, Taipei, Taiwan  
> <sup>3</sup>Institute of Bioinformatics and Systems Biology, National Yang Ming Chiao Tung University, Hsinchu, Taiwan  
> <sup>4</sup>Department of Biological Science and Technology, National Yang Ming Chiao Tung University, Hsinchu, Taiwan  
> <sup>5</sup>Center for Intelligent Drug Systems and Smart Bio-devices (IDS2B), National Yang Ming Chiao Tung University, Hsinchu, Taiwan  
> <sup>6</sup>College of Health Sciences, Kaohsiung Medical University, Kaohsiung, Taiwan  
> <sup>†</sup>Both authors contributed equally to this work  
> <sup>*</sup>Co-Corresponding authors

**Corresponding Authors:**
- Shinn-Ying Ho, PhD (syho@nycu.edu.tw)
- Wen-Liang Fang, MD, PhD (wlfang@vghtpe.gov.tw)

## 🌐 Online Tool

An interactive web-based prediction tool is available at:

**https://gapln-pgc.vercel.app/**

This web application provides a user-friendly interface for making predictions without installing any software.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Input Data Format](#input-data-format)
- [Model Details](#model-details)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

This repository provides pre-trained XGBoost models for predicting lymph node metastasis at:
- **Station 5**: Suprapyloric lymph nodes
- **Station 6**: Infrapyloric lymph nodes

The models use clinical and pathological features to predict the risk of lymph node metastasis, which can inform surgical decision-making for proximal gastric cancer patients.

### Key Features

- ✅ **Pre-trained Models**: Ready-to-use XGBoost models for LN5 and LN6 prediction
- ✅ **Automated Preprocessing**: Built-in feature engineering and transformation
- ✅ **Simple API**: Easy-to-use Python functions
- ✅ **Reproducible**: Complete preprocessing parameters included
- ✅ **No Imputation Required**: Works with complete patient data

---

## 🚀 Features

- **Two prediction models**: Station 5 (LN5) and Station 6 (LN6)
- **Automated feature engineering**: Derived features are automatically computed
- **Box-Cox transformation**: Model-specific transformations applied automatically
- **Standardization**: Z-score normalization using training set statistics
- **Flexible input**: Supports single patient or batch prediction
- **Simple output**: Binary prediction (0/1) and risk score (0-1)

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/boris890907/GAPLN-PGC56.git
cd GAPLN-PGC56
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
python -c "from src import predict_ln5, predict_ln6; print('Installation successful!')"
```

---

## ⚡ Quick Start

### Single Patient Prediction
```python
from src import predict_both

# Patient data
patient = {
    'Age': 65,
    'Sex': 1,  # 1=Male, 2=Female
    'Blood_Type_ABO': 1,  # 1=A, 2=B, 3=O, 4=AB
    'CEA': 3.5,
    'CA19_9': 25.0,
    'AFP': 2.1,
    'Tumor_Size': 4.5,
    'Tumor_Location': 4,  # 1-5
    'Circumferential_Extent': 5,  # 1-13
    'Borrmann_Type': 7,  # 1-9
    'Proximal_Resection_Margin': 30.0,
    'Distal_Resection_Margin': 25.0,
    'Cell_Differentiation': 1  # 1=Poor, 2=Moderate, 3=Well
}

# Predict both stations
results = predict_both(patient)

print(f"Station 5: Prediction={results['ln5']['prediction']}, Risk={results['ln5']['risk_score']:.3f}")
print(f"Station 6: Prediction={results['ln6']['prediction']}, Risk={results['ln6']['risk_score']:.3f}")
```

**Output:**
```
Station 5: Prediction=1, Risk=0.734
Station 6: Prediction=0, Risk=0.284
```

---

## 📖 Usage Examples

### Example 1: Single Patient Prediction
```python
from src import predict_ln5, predict_ln6

# Define patient data
patient = {
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

# Predict Station 5
result_ln5 = predict_ln5(patient)
print(f"LN5 Prediction: {result_ln5['prediction']}")
print(f"LN5 Risk Score: {result_ln5['risk_score']:.4f}")

# Predict Station 6
result_ln6 = predict_ln6(patient)
print(f"LN6 Prediction: {result_ln6['prediction']}")
print(f"LN6 Risk Score: {result_ln6['risk_score']:.4f}")
```

### Example 2: Batch Prediction from CSV
```python
import pandas as pd
from src import predict_both

# Load patient data
df = pd.read_csv('examples/sample_input.csv')

# Remove label columns if present
feature_cols = [col for col in df.columns if col not in ['LN_5_Label', 'LN_6_Label']]
df_features = df[feature_cols]

# Predict for all patients
results = predict_both(df_features)

# Add predictions to dataframe
df['LN5_Prediction'] = [r['ln5']['prediction'] for r in results]
df['LN5_Risk_Score'] = [r['ln5']['risk_score'] for r in results]
df['LN6_Prediction'] = [r['ln6']['prediction'] for r in results]
df['LN6_Risk_Score'] = [r['ln6']['risk_score'] for r in results]

# Save results
df.to_csv('predictions.csv', index=False)
print(f"Predictions saved for {len(df)} patients")
```

### Example 3: Using Individual Model Functions
```python
from src import predict_ln5, predict_ln6, predict_both

patient = {...}  # patient data

# Option 1: Predict both stations together (more efficient)
results = predict_both(patient)

# Option 2: Predict stations separately
result_ln5 = predict_ln5(patient)
result_ln6 = predict_ln6(patient)
```

### Example 4: Input Validation
```python
from src.preprocessing import validate_input_features

patient = {
    'Age': 65,
    'Sex': 1,
    # ... some features missing
}

is_valid, missing_features = validate_input_features(patient)

if not is_valid:
    print(f"Error: Missing required features: {missing_features}")
else:
    print("All required features present")
    result = predict_ln5(patient)
```

---

## 📊 Input Data Format

### Required Features (13 features)

| Feature | Type | Description | Valid Values |
|---------|------|-------------|--------------|
| `Age` | Continuous | Age in years | Positive number |
| `Sex` | Categorical | Biological sex | 1=Male, 2=Female |
| `Blood_Type_ABO` | Categorical | ABO blood type | 1=A, 2=B, 3=O, 4=AB |
| `CEA` | Continuous | Carcinoembryonic antigen (ng/mL) | Positive number |
| `CA19_9` | Continuous | Cancer antigen 19-9 (U/mL) | Positive number |
| `AFP` | Continuous | Alpha-fetoprotein (ng/mL) | Positive number |
| `Tumor_Size` | Continuous | Tumor size (cm) | Positive number |
| `Tumor_Location` | Categorical | Tumor location | 1=Cardia, 2=Upper-third, 3=Middle-third, 4=Lower-third, 5=Whole stomach |
| `Circumferential_Extent` | Categorical | Circumferential extent | 1-13 (see encoding below) |
| `Borrmann_Type` | Categorical | Borrmann classification | 1-9 (see encoding below) |
| `Proximal_Resection_Margin` | Continuous | Proximal margin (mm) | Positive number |
| `Distal_Resection_Margin` | Continuous | Distal margin (mm) | Positive number |
| `Cell_Differentiation` | Categorical | Cell differentiation | 1=Poor, 2=Moderate, 3=Well |

### Circumferential Extent Encoding

1. Less curvature (Less)
2. Greater curvature (Gre)
3. Anterior wall (Ant)
4. Posterior wall (Post)
5. Full circumferential
6. Ant + Less
7. Less + Post
8. Post + Gre
9. Gre + Ant
10. Ant + Less + Post
11. Less + Post + Gre
12. Post + Gre + Ant
13. Gre + Ant + Less

### Borrmann Type Encoding

1. Type 0-I
2. Type 0-IIa
3. Type 0-IIb
4. Type 0-IIc
5. Type 0-III
6. Type 1
7. Type 2
8. Type 3
9. Type 4

### Example Input (CSV Format)

See `examples/sample_input.csv` for complete examples:
```csv
Age,Sex,Blood_Type_ABO,CEA,CA19_9,AFP,Tumor_Size,Tumor_Location,Circumferential_Extent,Borrmann_Type,Proximal_Resection_Margin,Distal_Resection_Margin,Cell_Differentiation
65,1,1,3.5,25.0,2.1,4.5,4,5,7,30.0,25.0,1
72,2,2,8.2,45.3,1.8,6.2,3,8,8,25.5,30.2,2
```

---

## 🔬 Model Details

### LN5 Model (Station 5 Prediction)

- **Model Type**: XGBoost Classifier
- **Input Features**: 9 features (after preprocessing)
  - `Tumor_Location`, `Distal_Resection_Margin`, `Age`, `Tumor_Wall_Invasion`
  - `CEA_B`, `AFP_B`, `Tumor_Size`, `CA19_9`, `Cell_Differentiation`
- **Box-Cox Transformation**: Applied to CEA and AFP
- **Performance Metrics**: See publication for detailed performance

### LN6 Model (Station 6 Prediction)

- **Model Type**: XGBoost Classifier
- **Input Features**: 8 features (after preprocessing)
  - `Proximal_Resection_Margin`, `Tumor_Location`, `Tumor_Wall_Invasion`, `Tumor_Type_3A`
  - `CEA_B`, `Tumor_Size_Mucosal_1_3`, `Blood_Type_ABO`, `Sex`
- **Box-Cox Transformation**: Applied to CEA only
- **Performance Metrics**: See publication for detailed performance

### Preprocessing Pipeline

The preprocessing pipeline automatically performs:

1. **Feature Engineering**:
   - `Tumor_Wall_Invasion` ← `Circumferential_Extent`
   - `Tumor_Type_3A` ← `Borrmann_Type`
   - `Tumor_Size_Mucosal_1_3` ← Categorized from `Tumor_Size` (1: >8cm, 2: 4-8cm, 3: <4cm)

2. **Box-Cox Transformation**:
   - LN5: Transforms CEA and AFP
   - LN6: Transforms CEA only
   - Creates `_B` suffix features (e.g., `CEA_B`, `AFP_B`)

3. **Standardization**:
   - Z-score normalization using training set statistics
   - Applied to all features

### Output Format

Each prediction returns:
```python
{
    'prediction': int,      # Binary prediction (0 or 1)
    'risk_score': float     # Risk probability (0.0 to 1.0)
}
```

- `prediction`: 0 = No metastasis predicted, 1 = Metastasis predicted
- `risk_score`: Continuous probability score (higher = higher risk)

---

## 📁 Repository Structure
```
GAPLN-PGC56/
├── README.md                   # This file
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies
├── models/                     # Pre-trained models
│   ├── xgb_model_5.json       # LN5 model
│   └── xgb_model_6.json       # LN6 model
├── preprocessing/              # Preprocessing parameters
│   ├── boxcox_params_ln5.json
│   ├── boxcox_params_ln6.json
│   ├── scaler_params_ln5.json
│   └── scaler_params_ln6.json
├── src/                        # Source code
│   ├── __init__.py            # Package initialization
│   ├── preprocessing.py       # Preprocessing functions
│   └── predict.py             # Prediction functions
└── examples/                   # Usage examples
    └── sample_input.csv       # Sample patient data
```

---

## 📝 Citation

If you use these models in your research, please cite our paper:
```bibtex
@article{kung2024gapln,
  title={Evolutionary Learning-Based Prediction of No. 5 and No. 6 Lymph Node Metastasis in Proximal Gastric Cancer},
  author={Kung, Ching-Yun and Huang, Ching-Po and Huang, Kuo-Hung and Wu, Chew-Wen and Ho, Shinn-Ying and Fang, Wen-Liang},
  journal={International Journal of Surgery},
  note={Under review},
  year={2026}
}
```

**Note**: This manuscript is currently under review at the International Journal of Surgery. Please check back for updated citation information once the paper is published.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Summary

✅ **You can**:
- Use the models for research and clinical applications
- Modify and adapt the code
- Distribute derivative works

❌ **You must**:
- Include the original copyright notice
- Cite the original publication (when available)

---

## 🤝 Contributing

We welcome contributions! If you find bugs or have suggestions for improvements:

1. Open an issue describing the problem or suggestion
2. Submit a pull request with proposed changes

---

## ⚠️ Disclaimer

**Important**: These models are intended for research purposes only. They should not be used as the sole basis for clinical decision-making without proper validation in your specific clinical setting and consultation with qualified healthcare professionals.

---

## 💡 Troubleshooting

### Common Issues

**1. Import Error**
```python
ImportError: cannot import name 'predict_ln5' from 'src'
```

**Solution**: Make sure you're running Python from the repository root directory:
```bash
cd GAPLN-PGC56
python your_script.py
```

**2. Missing Features Error**
```python
ValueError: Missing required features: ['Age', 'Sex']
```

**Solution**: Ensure all 13 required features are present in your input data. Check the [Input Data Format](#input-data-format) section.

**3. File Not Found Error**
```python
FileNotFoundError: [Errno 2] No such file or directory: 'models/xgb_model_5.json'
```

**Solution**: Verify that the `models/` directory contains the model files. The directory structure should match the [Repository Structure](#repository-structure).

---

## 📧 Contact

For questions, issues, or collaboration inquiries:

**Corresponding Authors:**
- **Shinn-Ying Ho, PhD**  
  Institute of Bioinformatics and Systems Biology  
  National Yang Ming Chiao Tung University  
  Email: syho@nycu.edu.tw

- **Wen-Liang Fang, MD, PhD**  
  Division of General Surgery, Department of Surgery  
  Taipei Veterans General Hospital  
  Email: wlfang@vghtpe.gov.tw

**Repository Issues**: https://github.com/boris890907/GAPLN-PGC56/issues

**Web Tool**: https://gapln-pgc.vercel.app/

---

## 🙏 Acknowledgments

This research was conducted at:
- Taipei Veterans General Hospital
- National Yang Ming Chiao Tung University
- Kaohsiung Medical University

We thank all the patients who participated in this study and the clinical staff who contributed to data collection.

Special thanks to the open-source community for the tools that made this work possible:
- [XGBoost](https://github.com/dmlc/xgboost)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [pandas](https://github.com/pandas-dev/pandas)
- [NumPy](https://github.com/numpy/numpy)

---

**Last Updated**: February 2026
