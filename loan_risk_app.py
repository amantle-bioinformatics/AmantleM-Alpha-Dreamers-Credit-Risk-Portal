
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the saved Random Forest model
model = joblib.load('loan_risk_model.pkl')

# Define the exact columns the model was trained on 
# (Based on your dataset structure)
FEATURES = [
    'Income', 'Age', 'Experience', 'Married/Single', 'House_Ownership', 
    'Car_Ownership', 'Profession', 'CITY', 'STATE', 'CURRENT_JOB_YRS', 
    'CURRENT_HOUSE_YRS'
]

#--- STAGE 2: APP LAYOUT ---
st.set_page_config(page_title="Alpha Dreamers Credit Risk Portal", layout="wide")

st.title("🛡️ Alpha Dreamers Banking Consortium")
st.subheader("Automated Loan Risk Assessment Portal")
st.write("Enter the applicant's details below to calculate the risk profile.")

# Use columns to organize the input form
col1, col2 = st.columns(2)

with col1:
    st.header("👤 Personal & Professional")
    age = st.number_input("Applicant Age", min_value=18, max_value=100, value=30)
    marital = st.selectbox("Marital Status", ["single", "married"])
    profession = st.selectbox("Profession", [
        "Mechanical_engineer", "Software_Developer", "Technical_writer", "Civil_servant", 
        "Librarian", "Economist", "Flight_attendant", "Architect", "Designer", "Physician", 
        "Financial_Analyst", "Air_traffic_controller", "Politician", "Police_officer", 
        "Artist", "Surveyor", "Design_Engineer", "Chemical_engineer", "Hotel_Manager", 
        "Dentist", "Comedian", "Biomedical_Engineer", "Graphic_Designer", 
        "Computer_hardware_engineer", "Petroleum_Engineer", "Secretary", "Computer_operator", 
        "Chartered_Accountant", "Technician", "Microbiologist", "Fashion_Designer", 
        "Aviator", "Psychologist", "Magistrate", "Lawyer", "Firefighter", "Engineer", 
        "Official", "Analyst", "Geologist", "Drafter", "Statistician", "Web_designer", 
        "Consultant", "Chef", "Army_officer", "Surgeon", "Scientist", "Civil_engineer", 
        "Industrial_Engineer", "Technology_specialist"
    ])
    experience = st.number_input("Years of Professional Experience", min_value=0, max_value=50, value=5)
    job_years = st.number_input("Years in Current Job", min_value=0, max_value=50, value=2)

with col2:
    st.header("💰 Financial & Residential")
    income = st.number_input("Annual Income", min_value=0, value=50000)
    house_ownership = st.selectbox("House Ownership", ["rented", "owned", "norent_noown"])
    car_ownership = st.selectbox("Car Ownership", ["no", "yes"])
    house_years = st.number_input("Years in Current Residence", min_value=0, max_value=50, value=5)
    city = st.text_input("City", "New York")
    state = st.text_input("State", "NY")

# --- STAGE 3: PREDICTION LOGIC ---
if st.button("RUN RISK ASSESSMENT"):
    
    # Internal Encoding Logic (Must match your LabelEncoder training)
    # Note: These values are examples. Ensure they match your specific encoder mapping!
    encoded_marital = 1 if marital == "single" else 0
    encoded_car = 1 if car_ownership == "yes" else 0
    
    house_map = {"rented": 0, "owned": 1, "norent_noown": 2}
    encoded_house = house_map[house_ownership]

    # For Profession, City, and State, your app would typically use 
    # the LabelEncoder.transform() or a saved dictionary.
    # For now, we will use a placeholder logic:
    encoded_prof = 1 # Placeholder: Replace with your specific mapping logic
    encoded_city = 1 # Placeholder
    encoded_state = 1 # Placeholder

    # Create the Feature DataFrame
    input_data = pd.DataFrame([[
        income, age, experience, encoded_marital, encoded_house, 
        encoded_car, encoded_prof, encoded_city, encoded_state, 
        job_years, house_years
    ]], columns=FEATURES)

    # Execute Prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1] # Probability of Risk

    # --- STAGE 4: DISPLAY RESULTS ---
    st.divider()
    if prediction[0] == 1:
        st.error(f"⚠️ **HIGH RISK DETECTED** (Probability: {probability:.2%})")
        st.write("Recommendation: Reject automated approval. Refer to Senior Underwriter.")
    else:
        st.success(f"✅ **LOW RISK PROFILE** (Risk Probability: {probability:.2%})")
        st.write("Recommendation: Proceed with automated loan processing.")
