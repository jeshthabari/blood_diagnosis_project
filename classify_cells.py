import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =============== STEP 1: Load data ===============
df = pd.read_csv(r"C:\Users\jesht\Downloads\blood_diagnosis_project\HPLC data.csv")


# =============== STEP 2: Create binary target ===============
df['target'] = df['Diagnosis'].apply(lambda x: 0 if 'normal' in str(x).lower() else 1)

# =============== STEP 3: Clean age ===============
def extract_years(a):
    if pd.isna(a): 
        return np.nan
    s = str(a).lower()
    import re
    m = re.search(r'(\d+)', s)
    if m:
        return float(m.group(1))
    try:
        return float(s)
    except:
        return np.nan

df['Age_years'] = df['Age'].apply(extract_years)
df['Age_years'].fillna(df['Age_years'].median(), inplace=True)

# =============== STEP 4: Encode categoricals ===============
cat_cols = ['Gender', 'Jaundice', 'Weekness']
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].fillna('Unknown').astype(str).str.strip()
    else:
        df[c] = 'Unknown'

df_enc = pd.get_dummies(df[cat_cols], drop_first=True)

# =============== STEP 5: Select numeric features ===============
numeric_features = ['HbA0', 'HbA2', 'HbF', 'S-Window', 'Unknown',
                    'RBC', 'HB', 'MCV', 'MCH', 'MCHC', 'RDWcv', 'Age_years']

X = pd.concat([df[numeric_features].reset_index(drop=True),
               df_enc.reset_index(drop=True)], axis=1)

# Fill missing numeric values with median
for f in numeric_features:
    if X[f].isnull().any():
        X[f].fillna(X[f].median(), inplace=True)

y = df['target']

# =============== STEP 6: Split data ===============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# =============== STEP 7: Scale numeric data ===============
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[numeric_features])
X_test_num = scaler.transform(X_test[numeric_features])

# Combine numeric (scaled) + categorical
X_train_final = np.hstack([X_train_num, X_train[df_enc.columns].values])
X_test_final = np.hstack([X_test_num, X_test[df_enc.columns].values])

feature_names = numeric_features + list(df_enc.columns)

# =============== STEP 8: Train models ===============
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
    "SVM (RBF)": SVC(probability=True, class_weight='balanced', gamma='scale')
}

results = {}
for name, model in models.items():
    model.fit(X_train_final, y_train)
    y_pred = model.predict(X_test_final)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    results[name] = (acc, model)

# =============== STEP 9: Random Forest feature importances ===============
rf = results["Random Forest"][1]
importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
print("\nTop 10 important features:\n", importances.head(10))

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, rf.predict(X_test_final)),
            annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()
print("Confusion matrix saved as confusion_matrix.png")


# =============== STEP 10: Save model for reuse ===============
joblib.dump({'model': rf, 'scaler': scaler, 'numeric_features': numeric_features,
             'cat_columns': list(df_enc.columns)}, "best_model_random_forest.pkl")

print("\nModel saved as 'best_model_random_forest.pkl'")
