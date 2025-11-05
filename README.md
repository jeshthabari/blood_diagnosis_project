# Automated Screening of Haematological Disorders

This project uses **Machine Learning** to classify blood samples as *Normal* or *Abnormal* based on HPLC test data.

### 🧠 Overview
- **Language:** Python  
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn, joblib  

### ⚙️ How to Run
1. Install dependencies  
   ```bash
   pip install -r requirements.txt
2. Run the script
    ```bash
   python classify_cells.py


### 📊 Dataset

Input: HPLC parameters (HbA0, HbA2, HbF, RBC, HB, etc.)

The project now uses an extended dataset (`HPLC_data_extended.csv`) with 50,000 samples, 
generated synthetically to improve generalization and reduce overfitting.

Output: Diagnosis (Normal / Abnormal)

### 📈 Model Comparison

| Model | Accuracy |
|--------|-----------|
| Random Forest | 96.8% |
| Extra Trees | 96.2% |
| AdaBoost | 94.9% |
| Logistic Regression | 94.3% |
| SVM (RBF) | 94.0% |


Author: Jeshtha Bari
