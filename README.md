gStudent Burnout Prediction using Machine Learning

Project Overview

Student burnout is a growing concern in academic environments due to stress, lack of sleep, and heavy workload.
This project uses Machine Learning (Random Forest Classifier) to predict whether a student is at risk of burnout based on behavioral and academic factors.



Objective

To build a model that predicts:

- 0 → No Burnout
- 1 → Burnout Risk

using student-related data.



Features Used

The model uses the following input features:

- Study Hours per Day
- Sleep Duration
- Number of Assignments
- Attendance Percentage
- Screen Time
- Stress Level (survey-based)



Machine Learning Model

- Algorithm Used: Random Forest Classifier
- Why Random Forest?
  - Handles non-linear relationships
  - Reduces overfitting using multiple decision trees
  - Provides feature importance for analysis



Project Workflow

1. Data Generation
   
   - Synthetic dataset created using Python
   - Realistic rules applied (e.g., low sleep + high stress → burnout)

2. Data Preprocessing
   
   - Handling missing values (if any)
   - Feature scaling using StandardScaler

3. Model Training
   
   - Dataset split into:
     - Training Set (80%)
     - Testing Set (20%)

4. Model Evaluation
   
   - Accuracy Score
   - Confusion Matrix
   - Classification Report

5. Prediction
   
   - Takes user input to predict burnout risk

6. Visualization
   
   - Feature Importance Graph using Matplotlib



Sample Output

- Model Accuracy: ~85% (may vary)
- Feature Importance:
  - Stress Level → Highest impact
  - Sleep → Second major factor
  - Study Hours, Attendance → Moderate influence



How to Run the Project

Step 1: Install Dependencies

pip install pandas numpy scikit-learn matplotlib

Step 2: Generate Dataset

python generate_dataset.py

Step 3: Run Model

python burnout_model.py



Project Structure

student-burnout-ml/
│── generate_dataset.py
│── burnout_model.py
│── dataset.csv
│── README.md



Example Prediction

Input:

Study Hours: 8  
Sleep: 4  
Assignments: 7  
Attendance: 65  
Screen Time: 9  
Stress Level: 9  

Output:

High Burnout Risk


Key Insights

- High stress level is the most critical factor in burnout
- Low sleep duration significantly increases risk
- Excessive screen time and workload contribute to fatigue


Future Improvements

- Add real-time data collection
- Build a web application using Flask
- Improve dataset with real student data
- Deploy model for college use


Author

Neha TS



Conclusion

This project demonstrates how Machine Learning can be used to identify student burnout early, helping institutions take preventive measures and improve student well-being.