import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Number of students
num_samples = 500

# Generate features
study_hours = np.random.randint(1, 12, num_samples)          # 1–11 hours
sleep = np.random.randint(3, 9, num_samples)                 # 3–8 hours
assignments = np.random.randint(1, 10, num_samples)          # 1–9 assignments
attendance = np.random.randint(50, 100, num_samples)         # 50–100%
screen_time = np.random.randint(2, 12, num_samples)          # 2–11 hours
stress_level = np.random.randint(1, 11, num_samples)         # 1–10 scale

# Create burnout logic (REALISTIC RULES)
burnout = []

for i in range(num_samples):
    score = 0
    
    # High study + low sleep → burnout
    if study_hours[i] > 7 and sleep[i] < 5:
        score += 2
        
    # Too many assignments
    if assignments[i] > 6:
        score += 1
        
    # High screen time
    if screen_time[i] > 8:
        score += 1
        
    # High stress
    if stress_level[i] > 7:
        score += 2
        
    # Low attendance
    if attendance[i] < 70:
        score += 1

    # Final decision
    burnout.append(1 if score >= 3 else 0)

# Create DataFrame
data = pd.DataFrame({
    "StudyHours": study_hours,
    "Sleep": sleep,
    "Assignments": assignments,
    "Attendance": attendance,
    "ScreenTime": screen_time,
    "StressLevel": stress_level,
    "Burnout": burnout
})

# Save to CSV
data.to_csv("dataset.csv", index=False)

print("✅ Dataset generated successfully as dataset.csv")
print("\nSample Data:")
print(data.head())