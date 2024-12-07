# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
data = pd.read_csv("//content//BreastCancer.csv")

# Data cleaning
data_cleaned = data.drop(columns=["id", "Unnamed: 32"])  
data_cleaned['diagnosis'] = data_cleaned['diagnosis'].map({'M': 1, 'B': 0})  

# Exploratory Data Analysis
# Distribution of target classes
plt.figure(figsize=(6, 4))
sns.countplot(data=data_cleaned, x='diagnosis', palette='Set2')
plt.title('Distribution of Target Classes')
plt.xticks([0, 1], ['Benign', 'Malignant'])
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(data_cleaned.corr(), cmap='coolwarm', annot=False)
plt.title('Correlation Heatmap')
plt.show()

# Pairplot for important features
important_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'diagnosis']
sns.pairplot(data_cleaned[important_features], hue='diagnosis', palette='Set1')
plt.show()

# Splitting the data into features and target variable
X = data_cleaned.drop(columns=['diagnosis'])
y = data_cleaned['diagnosis']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training a Random Forest classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Feature Importance
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
sorted_importances = feature_importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sorted_importances.head(10).plot(kind='bar', color='teal')
plt.title('Top 10 Feature Importances')
plt.ylabel('Importance Score')
plt.show()

# Predictions and evaluation
y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

# Save the predictions with actual values in a DataFrame
predictions = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred,
    'Probability': y_pred_proba
})
print("Predicted Values with Probabilities:\n", predictions.head())

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='No Skill')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.show()

# Display the first few rows of predictions
print("Sample Predictions:")
print(predictions.head())
