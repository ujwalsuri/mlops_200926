import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_pred_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Subscription Prediction
st.title("Tourism Package Subscription Prediction App")
st.write("The Tourism Package Subscription Prediction App is an internal tool for 'Visit With Us' staff that predicts whether customers will subscribe to the Tourism package pitched to them based on their details.")
st.write("Kindly enter the customer details to check whether they are likely to subscribe.")

# Collect user input
Age = st.number_input("Age (customer's age)", min_value=15, max_value=100, value=35)
TypeOfEnq = st.selectbox("Type Of Enquiry (Self Enquiry or Company Invited)", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier (1/2/3)", ["1", "2", "3"])
Duration = st.number_input("Duratoin of Vacation", min_value=1, max_value=365, value=7)
Occupation = st.selectbox("Occupation (Salaried/Free Lancer/Small Business/Large Business)", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender (Male/Female)", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting ", min_value=1, max_value=20, value=3)
NumberOfFollowups = st.number_input("Number of Followups", min_value=0, max_value=20, value=1)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"])
PreferredPropertyStar = st.selectbox("Preferred Property Star (1/2/3/4/5)", ["1", "2", "3", "4", "5"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "UnMarried"])
NumberOfTrips = st.number_input("Number of Trips", min_value=0, max_value=50, value=1)
Passport = st.selectbox("Passport (Yes = 1/No = 0)", ["1", "0"])
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score (1/2/3/4/5)", ["1", "2", "3", "4", "5"])
OwnCar = st.selectbox("Own Car (Yes = 1/No = 0)", ["1", "0"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
Designation = st.selectbox("Designation (Executive/Manager/Senior Manager/AVP/VP", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Monthly Income", min_value=100, max_value=100000, value=5000)

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    "Age":Age,
    "TypeofContact":TypeOfEnq, 
    "CityTier":CityTier,
    "DurationOfPitch":Duration,
    "Occupation":Occupation,
    "Gender":Gender,
    "NumberOfPersonVisiting":NumberOfPersonVisiting,
    "NumberOfFollowups":NumberOfFollowups,
    "ProductPitched":ProductPitched, 
    "PreferredPropertyStar":PreferredPropertyStar,
    "MaritalStatus":MaritalStatus,
    "NumberOfTrips":NumberOfTrips,
    "Passport":Passport, 
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar":OwnCar,
    "NumberOfChildrenVisiting":NumberOfChildrenVisiting,
    "MonthlyIncome":MonthlyIncome,
    "Designation":Designation
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "Subscribe" if prediction == 1 else "Not Subscribe"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
