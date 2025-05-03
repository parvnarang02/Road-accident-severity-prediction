import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("Road.csv")

# Drop columns with more than 40% missing values
threshold = 0.4
df = df.loc[:, df.isnull().mean() < threshold]

# Fill remaining missing values with 'Unknown'
df.fillna('Unknown', inplace=True)

# Separate target and features
y = df['Accident_severity']
X = df.drop('Accident_severity', axis=1)

# One-hot encode categorical features
X_encoded = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Apply SMOTE to training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred = model.predict(X_test)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix with class labels
labels = sorted(y.unique())
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), 
            annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Feature Importance Plot
importances = model.feature_importances_
features = X_encoded.columns
importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
importance_df.sort_values(by="Importance", ascending=False, inplace=True)

plt.figure(figsize=(12, 8))
sns.barplot(x="Importance", y="Feature", data=importance_df.head(20))  # top 20 features
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.show()
