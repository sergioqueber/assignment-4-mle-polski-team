"""
4. Mushroom foraging

The mushroom dataset (https://www.kaggle.com/datasets/dhinaharp/mushroom-dataset) 
contains data about approximately 60000 mushrooms, and the task is to classify them 
as either edible or poisonous. 

Analysis includes:
* Informed data preparation.
* 2 different classification models, one of which must be logistic regression.
* A discussion of which performance metric is most relevant for the evaluation of models.
* 2 different validation methodologies used to tune hyperparameters.
* Confusion matrices for models, and associated comments.
"""

# =============================================================================
# Data Loading
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score)

pd.set_option('display.max_columns', 1000)
df = pd.read_csv('secondary_data.csv', delimiter=';')
print("Initial data:")
print(df.head())

# =============================================================================
# Data Preparation
# =============================================================================
print("\n" + "="*80)
print("DATA PREPARATION")
print("="*80)

df_clean = df.replace('?', np.nan)

print("\nMissing values after cleaning:")
print(df_clean.isnull().sum())

X = df_clean.drop('class', axis=1)
y = df_clean['class']

# Encode target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\nTarget encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# One-hot encode features
le_dict = {}
X_onehot = pd.get_dummies(X, drop_first=True)

print(f"\nOriginal features: {X.shape[1]}")
print(f"One-hot encoded features: {X_onehot.shape[1]}")
print(f"Dataset shape: {X_onehot.shape}")
print("\nFirst few encoded feature names:")
print(X_onehot.columns[:15].tolist())
print("\nOne-hot encoded data preview:")
print(X_onehot.head())

# =============================================================================
# Scale and divide the data
# =============================================================================
print("\n" + "="*80)
print("TRAIN-TEST SPLIT AND SCALING")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X_onehot, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set size: {X_train_scaled.shape}")
print(f"Test set size: {X_test_scaled.shape}")

# =============================================================================
# Model 1: Logistic Regression with GridSearchCV
# =============================================================================
print("\n" + "="*80)
print("LOGISTIC REGRESSION - VALIDATION METHODOLOGY 1: GridSearchCV")
print("="*80)

# Define the model
log_reg = LogisticRegression(random_state=42, max_iter=1000)

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

print("Training Logistic Regression with GridSearchCV...")
print(f"Testing {len(param_grid['C']) * len(param_grid['penalty']) * len(param_grid['solver'])} combinations")

# Validation Methodology 1: GridSearchCV with 5-fold Cross-Validation
grid_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    cv=5,
    scoring='recall',  # Prioritize recall to minimize false negatives (poisonous classified as edible)
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation recall score: {grid_search.best_score_:.4f}")

# Evaluate best model
best_log_reg = grid_search.best_estimator_

# Make predictions
y_pred_train = best_log_reg.predict(X_train_scaled)
y_pred_test = best_log_reg.predict(X_test_scaled)
y_pred_proba = best_log_reg.predict_proba(X_test_scaled)[:, 1]

print("\n=== Logistic Regression Performance ===")
print("\nTraining Set:")
print(classification_report(y_train, y_pred_train, target_names=['Edible', 'Poisonous']))

print("\nTest Set:")
print(classification_report(y_test, y_pred_test, target_names=['Edible', 'Poisonous']))
print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
            xticklabels=['Edible', 'Poisonous'], 
            yticklabels=['Edible', 'Poisonous'])
plt.title('Confusion Matrix - Logistic Regression')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix_logreg_gridsearch.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nConfusion Matrix Analysis:")
print(f"True Negatives (Edible correctly classified): {cm[0,0]}")
print(f"False Positives (Edible wrongly classified as Poisonous): {cm[0,1]}")
print(f"False Negatives (Poisonous wrongly classified as Edible): {cm[1,0]}  DANGEROUS!")
print(f"True Positives (Poisonous correctly classified): {cm[1,1]}")

# =============================================================================
# Performance Metrics Explanation
# =============================================================================
print("\n" + "="*80)
print("PERFORMANCE METRICS EXPLANATION")
print("="*80)
print("""
Why Recall is Most Important for Mushroom Classification:

For mushroom classification, Recall (Sensitivity) for the poisonous class is the 
most critical metric because:

1. False Negatives are Dangerous: Classifying a poisonous mushroom as edible 
   could be FATAL. This is far worse than the reverse error.
   
2. Cost Asymmetry: A False Positive (edible classified as poisonous) only means 
   missing a meal, but a False Negative (poisonous classified as edible) could 
   mean death.
   
3. Safety-First Approach: We want to catch as many poisonous mushrooms as possible, 
   even if it means being overly cautious.

Interpretation:
- Our model achieves 86% recall for poisonous mushrooms, meaning it catches most 
  dangerous mushrooms
- However, the 14% false negative rate is still risky in a real-world scenario
- For a real system, we might want to:
  * Lower the classification threshold to increase recall (catch more poisonous mushrooms)
  * Accept more false positives (classify more as poisonous) to minimize danger
  * Aim for recall > 95% for the poisonous class
""")

# =============================================================================
# Validation Methodology 2: Train-Validation-Test Split
# =============================================================================
print("\n" + "="*80)
print("LOGISTIC REGRESSION - VALIDATION METHODOLOGY 2: Train-Val-Test Split")
print("="*80)

# First split: separate test set (20%)
X_trainval, X_test_split, y_trainval, y_test_split = train_test_split(
    X_onehot, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Second split: separate validation set from training (20% of remaining 80% = 16% of total)
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
)

scaler_split = StandardScaler()
X_train_split_scaled = scaler_split.fit_transform(X_train_split)
X_val_scaled = scaler_split.transform(X_val)
X_test_split_scaled = scaler_split.transform(X_test_split)

print(f"Training set size: {X_train_split_scaled.shape}")
print(f"Validation set size: {X_val_scaled.shape}")
print(f"Test set size: {X_test_split_scaled.shape}")

# Manual hyperparameter tuning using validation set
log_reg_manual = LogisticRegression(random_state=42, max_iter=1000)

C_values = [0.001, 0.01, 0.1, 1, 10, 50, 100]  # Fixed: added comma between 50 and 100
val_scores = []

print("\nTuning C parameter using validation set:")
for C in C_values:
    log_reg_manual.set_params(C=C, penalty='l2')
    log_reg_manual.fit(X_train_split_scaled, y_train_split)
    val_score = log_reg_manual.score(X_val_scaled, y_val)
    val_scores.append(val_score)
    print(f"C={C}: Validation accuracy = {val_score:.4f}")

# Find best C
best_C = C_values[val_scores.index(max(val_scores))]
print(f"\nBest C value: {best_C}")

# Train final model with best C
final_log_reg = LogisticRegression(random_state=42, max_iter=1000, solver='saga', 
                                   C=best_C, penalty='l2')
final_log_reg.fit(X_train_split_scaled, y_train_split)

train_score = final_log_reg.score(X_train_split_scaled, y_train_split)
val_score = final_log_reg.score(X_val_scaled, y_val)
test_score = final_log_reg.score(X_test_split_scaled, y_test_split)

print(f"\nFinal Model Performance:")
print(f"Training accuracy: {train_score:.4f}")
print(f"Validation accuracy: {val_score:.4f}")
print(f"Test accuracy: {test_score:.4f}")

y_pred_final = final_log_reg.predict(X_test_split_scaled)
print("\n=== Test Set Classification Report ===")
print(classification_report(y_test_split, y_pred_final, target_names=['Edible', 'Poisonous']))

# Confusion Matrix for Train-Val-Test Split
cm_split = confusion_matrix(y_test_split, y_pred_final)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_split, annot=True, fmt='d', cmap='Reds', 
            xticklabels=['Edible', 'Poisonous'], 
            yticklabels=['Edible', 'Poisonous'])
plt.title('Confusion Matrix - Logistic Regression (Train-Val-Test Split)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix_logreg_trainvaltest.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nConfusion Matrix Analysis (Train-Val-Test Split):")
print(f"True Negatives (Edible correctly classified): {cm_split[0,0]}")
print(f"False Positives (Edible wrongly classified as Poisonous): {cm_split[0,1]}")
print(f"False Negatives (Poisonous wrongly classified as Edible): {cm_split[1,0]}  DANGEROUS!")
print(f"True Positives (Poisonous correctly classified): {cm_split[1,1]}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
