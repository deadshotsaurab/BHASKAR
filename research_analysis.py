import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score

# Paths
folder = 'c:/Users/saura/OneDrive/Desktop/bhaskar/ROC_curve'
xlsx_path = os.path.join(folder, 'training_data.xlsx')

# Load Data
df = pd.read_excel(xlsx_path)

# Data Preparation
drop_cols = ['FID', 'BLOCK', 'YIELD__lpm', 'CID', 'LATITUDE', 'LONGITUDE']
existing_cols_to_drop = [col for col in drop_cols if col in df.columns]
if existing_cols_to_drop:
    df = df.drop(columns=existing_cols_to_drop)

X = df.drop(columns=['outcome'])
y = df['outcome']

# 70-30 train-test split for small data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM (SVC)': SVC(probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

results = []

# Output directory for plots
out_dir = 'c:/Users/saura/OneDrive/Desktop/bhaskar'

# 1. Combined ROC Curve
plt.figure(figsize=(12, 8))
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'AUC': roc_auc
    })
    
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('Receiver Operating Characteristic (ROC) - Comparative Analysis', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.savefig(os.path.join(out_dir, 'combined_roc.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Confusion Matrices
sns.set_theme(style="white")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Confusion Matrices of Machine Learning Models", fontsize=16, fontweight='bold')
axes = axes.flatten()

for idx, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False, annot_kws={"size": 16})
    axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Predicted Label', fontsize=10)
    axes[idx].set_ylabel('True Label', fontsize=10)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(out_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
plt.close()

# Print results as markdown
print("## Performance Metrics Table\n")
print("| Model | Accuracy | Precision | Recall | F1-Score | AUC |")
print("|-------|----------|-----------|--------|----------|-----|")
for r in results:
    print(f"| {r['Model']} | {r['Accuracy']:.4f} | {r['Precision']:.4f} | {r['Recall']:.4f} | {r['F1-Score']:.4f} | {r['AUC']:.4f} |")

print("\nDONE")
