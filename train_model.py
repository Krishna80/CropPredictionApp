import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
file_path = "New Crop 50K.csv"
df = pd.read_csv(file_path)

# Check for missing values
if df.isnull().sum().sum() > 0:
    df.dropna(inplace=True)

# Select features
features = ['temperature', 'humidity', 'ph', 'rainfall',] # Consider more relevant features
X = df[features]
y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'LogisticRegression': LogisticRegression(),
    'GaussianNB':GaussianNB(),
    'SVC':SVC(),
    'KNeighborsClassifier':KNeighborsClassifier(),
    'DecisionTreeClassifier':DecisionTreeClassifier(),
    'ExtraTreeClassifier':ExtraTreeClassifier(),
    'RandomForestClassifier':RandomForestClassifier(),
    'BaggingClassifier':BaggingClassifier(),
    'GradientBoostingClassifier':GradientBoostingClassifier(),
    'AdaBoostClassifier':AdaBoostClassifier()
}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"{name} model with accuracy: {score * 100:.2f}%")

# Train model
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)  # Tuned hyperparameters
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# 6. Plot Confusion Matrix as a Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap for Crop Prediction")
plt.show()

import matplotlib.pyplot as plt

# Calculate accuracy for each class (if needed)
class_accuracy = cm.diagonal() / cm.sum(axis=1)  # Diagonal values (correct) / row sum (total)

# Plot class-wise accuracy
plt.figure(figsize=(6, 5))
plt.bar(np.unique(y_test), class_accuracy, color='blue')
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.title('Class-wise Accuracy for Crop Prediction')
plt.show()

metrics = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': []
}

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average='weighted') * 100
    recall = recall_score(y_test, y_pred, average='weighted') * 100
    f1 = f1_score(y_test, y_pred, average='weighted') * 100
    
    metrics['Model'].append(name)
    metrics['Accuracy'].append(accuracy)
    metrics['Precision'].append(precision)
    metrics['Recall'].append(recall)
    metrics['F1-Score'].append(f1)

# Convert metrics to DataFrame for easier plotting
metrics_df = pd.DataFrame(metrics)

# Plotting the metrics in a grouped bar chart
metrics_df.set_index('Model').plot(kind='bar', figsize=(12, 6), width=0.8)

# Adding labels and title
plt.ylabel('Percentage')
plt.xlabel('Models')
plt.title('Performance Metrics for GPU-Accelerated Models')
plt.ylim(0, 100)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Metrics')
plt.tight_layout()

# Show the plot
plt.show()


# Save model
joblib.dump(model, "cropNarayangarh_model.pkl")
print("✅ Model saved as cropNarayangarh_model.pkl")
