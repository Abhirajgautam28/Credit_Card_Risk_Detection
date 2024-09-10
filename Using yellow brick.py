import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from yellowbrick.classifier import ROCAUC, ConfusionMatrix, ClassPredictionError
from yellowbrick.features import PCA as YellowbrickPCA
import os

# Load the dataset
credit = pd.read_csv('D:\Credit\dataset\german_credit_data.csv')

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')
credit = pd.DataFrame(imputer.fit_transform(credit), columns=credit.columns)

# Encode categorical variables
# Use LabelEncoder for all non-numeric columns (categorical features)
le = LabelEncoder()
for column in credit.columns:
    if credit[column].dtype == 'object':
        credit[column] = le.fit_transform(credit[column].astype(str))

# Define the target and features
X = credit.drop('Risk', axis=1)  # Replace 'Credit_Risk' with the actual target column
y = credit['Risk']  # Replace 'Credit_Risk' with the actual target column

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a KNN model
print("Training KNN model...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = knn.predict(X_test)
print("KNN Classification Report")
print(classification_report(y_test, y_pred))

# Yellowbrick visualizations for KNN
# Confusion Matrix for KNN
cm_viz = ConfusionMatrix(knn)
cm_viz.fit(X_train, y_train)
cm_viz.score(X_test, y_test)
cm_viz.show(outpath="outputs/knn_confusion_matrix.png")  # Save the visualization to 'outputs' folder

# Class Prediction Error for KNN
cpe_viz = ClassPredictionError(knn)
cpe_viz.fit(X_train, y_train)
cpe_viz.score(X_test, y_test)
cpe_viz.show(outpath="outputs/knn_class_prediction_error.png")  # Save the visualization to 'outputs' folder

# Train a RandomForest model for ROC AUC visualization
print("Training RandomForest model for ROC AUC...")
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Make predictions and evaluate RandomForest
y_proba = rf.predict_proba(X_test)[:, 1]
print(f"RandomForest ROC AUC Score: {roc_auc_score(y_test, y_proba):.2f}")

# ROC AUC for RandomForest
roc_auc_viz = ROCAUC(rf)
roc_auc_viz.fit(X_train, y_train)
roc_auc_viz.score(X_test, y_test)
roc_auc_viz.show(outpath="outputs/rf_roc_auc.png")  # Save the visualization to 'outputs' folder

# PCA Visualization using Yellowbrick
pca_viz = YellowbrickPCA(scale=True, proj_dim=2)
pca_viz.fit_transform(X_train, y_train)
pca_viz.show(outpath="outputs/pca_visualization.png")  # Save the visualization to 'outputs' folder

# Save all visualizations in the outputs folder without displaying them on the screen
output_dir = "outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Now let's test the integration of Yellowbrick and ensure everything works smoothly
print("All visualizations saved in the 'outputs' folder.")
