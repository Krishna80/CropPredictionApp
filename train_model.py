import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier

# Load dataset
file_path = "Crop_recommendation.csv"
df = pd.read_csv(file_path)

# Drop missing values
df.dropna(inplace=True)

# Features and target
features = ['temperature', 'humidity', 'ph',]
X = df[features]
y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models dictionary using pipeline
models = {
    'LogisticRegression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=200))
    ]),
    'GaussianNB': Pipeline([
        ('scaler', StandardScaler()),  # Optional for Naive Bayes, but safe
        ('model', GaussianNB())
    ]),
    'SVC': Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC())
    ]),
    'KNeighborsClassifier': Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier())
    ]),
    'DecisionTreeClassifier': Pipeline([
        ('model', DecisionTreeClassifier())
    ]),
    'ExtraTreeClassifier': Pipeline([
        ('model', ExtraTreeClassifier())
    ]),
    'RandomForestClassifier': Pipeline([
        ('model', RandomForestClassifier())
    ]),
    'BaggingClassifier': Pipeline([
        ('model', BaggingClassifier())
    ]),
    'GradientBoostingClassifier': Pipeline([
        ('model', GradientBoostingClassifier())
    ]),
    'AdaBoostClassifier': Pipeline([
        ('model', AdaBoostClassifier())
    ])
}

# Store performance metrics
metrics = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': []
}

# Train and evaluate models
for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average='weighted') * 100
    recall = recall_score(y_test, y_pred, average='weighted') * 100
    f1 = f1_score(y_test, y_pred, average='weighted') * 100
    
    print(f"{name} model with accuracy: {accuracy:.2f}%")
    
    metrics['Model'].append(name)
    metrics['Accuracy'].append(accuracy)
    metrics['Precision'].append(precision)
    metrics['Recall'].append(recall)
    metrics['F1-Score'].append(f1)

# Convert metrics to DataFrame
metrics_df = pd.DataFrame(metrics)

# Plot metrics
metrics_df.set_index('Model').plot(kind='bar', figsize=(12, 6), width=0.8)
plt.ylabel('Percentage')
plt.xlabel('Models')
plt.title('Performance Metrics for Crop Prediction Models')
plt.ylim(0, 100)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Metrics')
plt.tight_layout()
plt.show()

# Select best model (based on F1-score here) - adjust as needed
best_model_name = metrics_df.sort_values(by='F1-Score', ascending=False).iloc[0]['Model']
final_model = models[best_model_name]

# Re-train on full training set
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix Heatmap for {best_model_name}")
plt.show()

# Class-wise accuracy
class_accuracy = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(6, 5))
plt.bar(np.unique(y), class_accuracy, color='green')
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.title(f'Class-wise Accuracy for {best_model_name}')
plt.show()

# Save best model
joblib.dump(final_model, "Short_Model.pkl")
print(f"✅ Best model ({best_model_name}) saved as short_mdoel.pkl")
