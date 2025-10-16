import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('attrition_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# App title and description
st.title('🚀 Employee Attrition Predictor')
st.write("Enter employee details to predict the risk of attrition (leaving the company). Based on the tuned Random Forest model (AUC ~0.89).")

# Sidebar for inputs (top features for simplicity; expand as needed)
st.sidebar.header('Employee Profile')

# Key features (top 10 from importance; adjust based on your feature_cols)
age = st.sidebar.slider('Age', min_value=18, max_value=60, value=30)
salary = st.sidebar.slider('Monthly Salary', min_value=1000, max_value=20000, value=5000)
overtime = st.sidebar.selectbox('OverTime (Yes/No)', ['No', 'Yes'])
job_satisfaction = st.sidebar.slider('Job Satisfaction (1-4)', min_value=1, max_value=4, value=3)
work_life_balance = st.sidebar.slider('Work Life Balance (1-4)', min_value=1, max_value=4, value=3)
years_at_company = st.sidebar.slider('Years at Company', min_value=0, max_value=40, value=5)
years_since_promotion = st.sidebar.slider('Years Since Last Promotion', min_value=0, max_value=15, value=2)
education = st.sidebar.slider('Education Level (1-5)', min_value=1, max_value=5, value=3)
distance_from_home = st.sidebar.slider('Distance from Home (miles)', min_value=1, max_value=30, value=10)
job_level = st.sidebar.slider('Job Level (1-5)', min_value=1, max_value=5, value=2)

# Map categorical (OverTime)
overtime_val = 1 if overtime == 'Yes' else 0

# Engineered: SatisfactionScore
satisfaction_score = (job_satisfaction + work_life_balance) / 2

# OverTime_Hours (proxy)
overtime_hours = 2 if overtime == 'Yes' else 0

# Prepare input DataFrame (must match exact feature_cols order from training!)
# WARNING: Order matters! Get from your Colab: print(X.columns.tolist()) and copy-paste here.
# For demo, assuming a subset; replace with full list for accuracy.
feature_names = [
    'Age', 'Salary', 'YearsAtCompany', 'JobSatisfaction', 'WorkLifeBalance', 'OverTime',
    'Education', 'OverTime_Hours', 'SatisfactionScore', 'YearsSinceLastPromotion',
    'DistanceFromHome', 'JobLevel'  # Add more dummies as needed, e.g., 'Department_Research & Development': 0, etc.
    # Full example: Add all 37, set unused dummies to 0 or mean from training.
]

input_data = pd.DataFrame({
    'Age': [age],
    'Salary': [salary],
    'YearsAtCompany': [years_at_company],
    'JobSatisfaction': [job_satisfaction],
    'WorkLifeBalance': [work_life_balance],
    'OverTime': [overtime_val],
    'Education': [education],
    'OverTime_Hours': [overtime_hours],
    'SatisfactionScore': [satisfaction_score],
    'YearsSinceLastPromotion': [years_since_promotion],
    'DistanceFromHome': [distance_from_home],
    'JobLevel': [job_level]
    # Add dummy columns as 0s if not selected, e.g.,
    # 'Department_Research & Development': [0],
    # etc. for all missing.
}, columns=feature_names)  # Ensure columns match exactly!

# Ensure all features are present (pad with 0s if subset)
for col in feature_names:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder to match training
input_data = input_data[feature_names]

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
if st.sidebar.button('Predict Attrition Risk'):
    prob = model.predict_proba(input_scaled)[0][1]  # Probability of Yes (attrition)
    pred = model.predict(input_scaled)[0]
    
    st.subheader('Prediction Results')
    st.metric("Attrition Risk", f"{prob:.1%}", delta=None)
    
    if pred == 1:
        st.error("⚠️ High Risk: Employee is likely to leave!")
    else:
        st.success("✅ Low Risk: Employee is likely to stay.")
    
    # Insights (based on top features)
    st.write("**Quick Insights:**")
    if overtime == 'Yes':
        st.warning("OverTime flagged as a top risk factor.")
    if salary < 6000:
        st.warning("Low Salary correlates with higher attrition.")
    if job_satisfaction < 3:
        st.warning("Improve Job Satisfaction to reduce risk.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit & Scikit-learn. Model from HR Attrition dataset.")