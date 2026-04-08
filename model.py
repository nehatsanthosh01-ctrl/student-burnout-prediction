# -----------------------------
# IMPORT LIBRARIES
# -----------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# LOAD DATASET
# -----------------------------
data = pd.read_csv("dataset.csv")

print("Dataset Loaded Successfully!\n")
print(data.head())

# -----------------------------
# FEATURES & TARGET
# -----------------------------
X = data.drop("Burnout", axis=1)
y = data["Burnout"]

# -----------------------------
# TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# FEATURE SCALING
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# MODEL TRAINING
# -----------------------------
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# PREDICTION
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# EVALUATION
# -----------------------------
print("\n MODEL PERFORMANCE")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

print("\n Feature Importance:")
print(importance)

# -----------------------------
# TEST WITH NEW STUDENT
# -----------------------------
print("\n Predict Burnout for New Student")

study_hours = float(input("Study Hours per day: "))
sleep = float(input("Sleep hours: "))
assignments = int(input("Number of assignments: "))
attendance = float(input("Attendance (%): "))
screen_time = float(input("Screen time (hours): "))
stress_level = int(input("Stress level (1-10): "))

new_data = np.array([[study_hours, sleep, assignments, attendance, screen_time, stress_level]])

new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)

print("\n Result:")
if prediction[0] == 1:
    print(" High Burnout Risk")
else:
    print(" No Burnout Risk")
# -----------------------------
# GRAPH: FEATURE IMPORTANCE
# -----------------------------
import matplotlib.pyplot as plt

# Convert importance to sorted values
importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values()

# Plot graph
plt.figure()
importance.plot(kind='barh')

plt.title("Feature Importance for Burnout Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.show()