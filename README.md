# Automated Screening of Haematological Disorders

This project uses **Machine Learning** to classify blood samples as *Normal* or *Abnormal* based on HPLC test data.

### 🧠 Overview
- **Language:** Python  
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn, joblib  
- **Best Model:** Random Forest (≈ 99.9% accuracy)

### ⚙️ How to Run
1. Install dependencies  
   ```bash
   pip install -r requirements.txt
2. Run the script

python classify_cells.py

📊 Dataset

Input: HPLC parameters (HbA0, HbA2, HbF, RBC, HB, etc.)

Output: Diagnosis (Normal / Abnormal)

📈 Results
Model	Accuracy
Random Forest	99.9%
Logistic Regression	99.6%
SVM (RBF)	99.4%

Author: Jeshtha Bari
