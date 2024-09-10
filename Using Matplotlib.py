import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, roc_auc_score, \
    roc_curve, precision_recall_curve, average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# File path setup
data_path = 'D:\Credit\dataset\german_credit_data.csv'
output_dir = 'D:\Credit\Outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the dataset
credit = pd.read_csv(data_path)

# Map categorical values to numeric
sex_mapping = {'male': 1, 'female': 2}
housing_mapping = {'own': 1, 'rent': 2, 'free': 3}
savings_mapping = {'little': 1, 'moderate': 2, 'quite rich': 3, 'rich': 4}
checking_mapping = {'little': 1, 'moderate': 2, 'rich': 3}
purpose_mapping = {'radio/TV': 1, 'furniture/equipment': 2, 'education': 3, 'car': 4, 'business': 5, 'maintenance': 6, 'travel': 7, 'other purposes': 8}
risk_mapping = {'bad': 0, 'good': 1}

if 'Sex' in credit.columns:
    credit['Sex'] = credit['Sex'].map(sex_mapping)

if 'Housing' in credit.columns:
    credit['Housing'] = credit['Housing'].map(housing_mapping)

if 'Saving accounts' in credit.columns:
    credit['Saving accounts'] = credit['Saving accounts'].map(savings_mapping)

if 'Checking account' in credit.columns:
    credit['Checking account'] = credit['Checking account'].map(checking_mapping)

if 'Purpose' in credit.columns:
    credit['Purpose'] = credit['Purpose'].map(purpose_mapping)

if 'Risk' in credit.columns:
    credit['Risk'] = credit['Risk'].map(risk_mapping)

# Convert all columns to numeric, handling errors explicitly
credit = credit.apply(pd.to_numeric, errors='coerce')
credit.fillna(0, inplace=True)

# Prepare Data for Classification
X, y = credit.drop("Risk", axis=1), credit["Risk"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Classification Models
results = []
models = {
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(),
    'Neural Network': MLPClassifier(max_iter=1000),
    'Halving Random Search': HalvingGridSearchCV(estimator=RandomForestClassifier(), param_grid={'n_estimators': [50, 100, 150]})
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    results.append({
        'Model': name,
        'F1 Score': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Confusion Matrix': confusion_matrix(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_proba)
    })

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, 'model_performance.csv'), index=False)

# Visualization
sns.set(style="whitegrid")

# 1. Distribution of features
for column in X.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(credit[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.savefig(os.path.join(output_dir, f'distribution_{column}.png'))
    plt.close()

# 2. Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(credit.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
plt.close()

# 3. Pairplot
sns.pairplot(credit)
plt.title('Pairplot of Features')
plt.savefig(os.path.join(output_dir, 'pairplot.png'))
plt.close()

# 4. Boxplots for each feature
for column in X.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=credit[column])
    plt.title(f'Boxplot of {column}')
    plt.savefig(os.path.join(output_dir, f'boxplot_{column}.png'))
    plt.close()

# 5. Histograms for each feature
for column in X.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(credit[column], kde=True)
    plt.title(f'Histogram of {column}')
    plt.savefig(os.path.join(output_dir, f'histogram_{column}.png'))
    plt.close()

# 6. ROC Curves for classifiers
plt.figure(figsize=(12, 8))
for name, model in models.items():
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_proba):.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
plt.close()

# 7. Confusion Matrices for classifiers
for result in results:
    model_name = result['Model']
    cm = result['Confusion Matrix']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name}.png'))
    plt.close()

# 8. Feature Importance from RandomForest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.savefig(os.path.join(output_dir, 'feature_importances.png'))
plt.close()

# 9. Learning Curves for a chosen model (e.g., RandomForest)
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(), X_train, y_train, cv=5, n_jobs=-1)
plt.figure(figsize=(12, 8))
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='r', label='Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='g', label='Cross-validation score')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.title('Learning Curves (RandomForest)')
plt.legend()
plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
plt.close()

# 10. Precision-Recall Curves for classifiers
plt.figure(figsize=(12, 8))
for name, model in models.items():
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.plot(recall, precision, label=f'{name} (AUC = {average_precision_score(y_test, y_proba):.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()
plt.savefig(os.path.join(output_dir, 'precision_recall_curves.png'))
plt.close()

print("All files and images have been saved successfully.")
