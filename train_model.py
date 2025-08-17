import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Load dataset
file_path = "Crop_recommendation.csv"
df = pd.read_csv(file_path)

df.dropna(inplace=True)

features = ['temperature', 'humidity', 'ph']
X = df[features]
y = df['label']

# Encode target labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

pipelines = {
    'RandomForest': Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(random_state=42))
    ]),
    'GradientBoosting': Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingClassifier(random_state=42))
    ]),
    'XGBoost': Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBClassifier(eval_metric='mlogloss', random_state=42))
    ])
}

param_grids = {
    'RandomForest': {
        'model__n_estimators': [100, 200],
        'model__max_depth': [10, 20, None],
        'model__min_samples_split': [2, 5],
    },
    'GradientBoosting': {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.05, 0.1],
        'model__max_depth': [3, 5],
    },
    'XGBoost': {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.05, 0.1],
        'model__max_depth': [3, 5],
    }
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []
best_models = {}

for name, pipeline in pipelines.items():
    print(f"Running GridSearchCV for {name}...")
    grid_search = GridSearchCV(
        pipeline,
        param_grids[name],
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train_resampled, y_train_resampled)
    
    best_models[name] = grid_search.best_estimator_
    y_pred = grid_search.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average='weighted') * 100
    recall = recall_score(y_test, y_pred, average='weighted') * 100
    f1 = f1_score(y_test, y_pred, average='weighted') * 100
    
    print(f"{name} best parameters: {grid_search.best_params_}")
    print(f"{name} Accuracy: {accuracy:.2f}%")
    print(f"{name} Precision: {precision:.2f}%")
    print(f"{name} Recall: {recall:.2f}%")
    print(f"{name} F1 Score: {f1:.2f}%\n")
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

metrics_df = pd.DataFrame(results)
print(metrics_df)
metrics_df.to_csv("model_performance_summary.csv", index=False)

metrics_df.set_index('Model').plot(kind='bar', figsize=(12, 6), width=0.8)
plt.ylabel('Percentage')
plt.xlabel('Model')
plt.title('Performance Metrics for Crop Prediction Models (Using temp, humidity, ph)')
plt.ylim(0, 100)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Metrics')
plt.tight_layout()
plt.show()

# ---- Confusion matrix and Class-wise accuracy ----
best_model_name = metrics_df.sort_values(by='F1-Score', ascending=False).iloc[0]['Model']
final_model = best_models[best_model_name]

# Predict on test set again (just to be sure)
y_pred = final_model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
classes = le.inverse_transform(np.arange(len(np.unique(y_test))))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix Heatmap for {best_model_name}")
plt.show()

# Class-wise accuracy
class_accuracy = cm.diagonal() / cm.sum(axis=1)

plt.figure(figsize=(10, 5))
plt.bar(classes, class_accuracy, color='green')
plt.xlabel("Crop Class")
plt.ylabel("Accuracy")
plt.title(f"Class-wise Accuracy for {best_model_name}")
plt.ylim(0, 1.0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save best model
joblib.dump(final_model, "Short_Model.pkl")
joblib.dump(le, "label_encoder.pkl")
print(f"✅ Best model ({best_model_name}) saved as Short_Model.pkl")
