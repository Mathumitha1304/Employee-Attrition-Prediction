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
num_companies_worked = st.sidebar.slider('Number of Companies Worked', min_value=0, max_value=9, value=1)
total_working_years = st.sidebar.slider('Total Working Years', min_value=0, max_value=40, value=10)
environment_satisfaction = st.sidebar.slider('Environment Satisfaction (1-4)', min_value=1, max_value=4, value=3)
performance_rating = st.sidebar.slider('Performance Rating (1-4)', min_value=1, max_value=4, value=3)
relationship_satisfaction = st.sidebar.slider('Relationship Satisfaction (1-4)', min_value=1, max_value=4, value=3)
stock_option_level = st.sidebar.slider('Stock Option Level (0-3)', min_value=0, max_value=3, value=0)

# Additional categoricals (select one category per group for dummies)
st.sidebar.subheader('Categorical Selections')
business_travel = st.sidebar.selectbox('Business Travel', ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
department = st.sidebar.selectbox('Department', ['Sales', 'Research & Development', 'Human Resources'])
education_field = st.sidebar.selectbox('Education Field', ['Life Sciences', 'Marketing', 'Medical', 'Other', 'Technical Degree', 'Human Resources'])
gender = st.sidebar.selectbox('Gender', ['Female', 'Male'])
marital_status = st.sidebar.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
job_role = st.sidebar.selectbox('Job Role', ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])

# Map categoricals to dummies (1 if selected, 0 otherwise)
# BusinessTravel
bt_frequent = 1 if business_travel == 'Travel_Frequently' else 0
bt_rarely = 1 if business_travel == 'Travel_Rarely' else 0

# Department
dept_rd = 1 if department == 'Research & Development' else 0
dept_hr = 1 if department == 'Human Resources' else 0

# EducationField
ef_life = 1 if education_field == 'Life Sciences' else 0
ef_marketing = 1 if education_field == 'Marketing' else 0
ef_medical = 1 if education_field == 'Medical' else 0
ef_other = 1 if education_field == 'Other' else 0
ef_technical = 1 if education_field == 'Technical Degree' else 0

# Gender
gender_male = 1 if gender == 'Male' else 0

# MaritalStatus
ms_married = 1 if marital_status == 'Married' else 0
ms_single = 1 if marital_status == 'Single' else 0

# JobRole
jr_hr = 1 if job_role == 'Human Resources' else 0
jr_lab_tech = 1 if job_role == 'Laboratory Technician' else 0
jr_manager = 1 if job_role == 'Manager' else 0
jr_mfg_dir = 1 if job_role == 'Manufacturing Director' else 0
jr_rd_dir = 1 if job_role == 'Research Director' else 0
jr_rs = 1 if job_role == 'Research Scientist' else 0
jr_sales_exec = 1 if job_role == 'Sales Executive' else 0
jr_sales_rep = 1 if job_role == 'Sales Representative' else 0

# Map OverTime
overtime_val = 1 if overtime == 'Yes' else 0

# Engineered: SatisfactionScore
satisfaction_score = (job_satisfaction + work_life_balance) / 2

# OverTime_Hours (proxy)
overtime_hours = 2 if overtime == 'Yes' else 0

# EXACT feature_names from Colab (37 features)
feature_names = [
    'Age', 'Salary', 'YearsAtCompany', 'JobSatisfaction', 'WorkLifeBalance', 'OverTime',
    'Education', 'OverTime_Hours', 'SatisfactionScore', 'YearsSinceLastPromotion',
    'DistanceFromHome', 'NumCompaniesWorked', 'TotalWorkingYears', 'EnvironmentSatisfaction',
    'JobLevel', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel',
    'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
    'Department_Research & Development', 'EducationField_Life Sciences',
    'EducationField_Marketing', 'EducationField_Medical', 'EducationField_Other',
    'EducationField_Technical Degree', 'Gender_Male', 'MaritalStatus_Married',
    'MaritalStatus_Single', 'JobRole_Human Resources', 'JobRole_Laboratory Technician',
    'JobRole_Manager', 'JobRole_Manufacturing Director', 'JobRole_Research Director',
    'JobRole_Research Scientist', 'JobRole_Sales Executive', 'JobRole_Sales Representative'
]

# Create input_data with ALL features
input_dict = {
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
    'NumCompaniesWorked': [num_companies_worked],
    'TotalWorkingYears': [total_working_years],
    'EnvironmentSatisfaction': [environment_satisfaction],
    'JobLevel': [job_level],
    'PerformanceRating': [performance_rating],
    'RelationshipSatisfaction': [relationship_satisfaction],
    'StockOptionLevel': [stock_option_level],
    'BusinessTravel_Travel_Frequently': [bt_frequent],
    'BusinessTravel_Travel_Rarely': [bt_rarely],
    'Department_Research & Development': [dept_rd],
    'EducationField_Life Sciences': [ef_life],
    'EducationField_Marketing': [ef_marketing],
    'EducationField_Medical': [ef_medical],
    'EducationField_Other': [ef_other],
    'EducationField_Technical Degree': [ef_technical],
    'Gender_Male': [gender_male],
    'MaritalStatus_Married': [ms_married],
    'MaritalStatus_Single': [ms_single],
    'JobRole_Human Resources': [jr_hr],
    'JobRole_Laboratory Technician': [jr_lab_tech],
    'JobRole_Manager': [jr_manager],
    'JobRole_Manufacturing Director': [jr_mfg_dir],
    'JobRole_Research Director': [jr_rd_dir],
    'JobRole_Research Scientist': [jr_rs],
    'JobRole_Sales Executive': [jr_sales_exec],
    'JobRole_Sales Representative': [jr_sales_rep]
}

input_data = pd.DataFrame(input_dict)

# Reorder to EXACTLY match feature_names order
input_data = input_data[feature_names]

# Verify shape matches training (1, 37)
st.sidebar.write(f"Input shape: {input_data.shape} (should be 1 x 37)")

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
