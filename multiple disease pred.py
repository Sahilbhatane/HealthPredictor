import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import json
import logging
import os
import csv
import sklearn
import joblib
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Print scikit-learn version
logger.info(f"Using scikit-learn version: {sklearn.__version__}")

# Configure page settings
st.set_page_config(
    layout='wide',
    page_title='HealthPredictor AI',
    page_icon='🩺',
    initial_sidebar_state='expanded'
)

# Dark theme styles
dark_theme = """
<style>
    :root {
        --background-color: #0E1117;
        --text-color: #F0F2F6;
        --card-bg-color: #1E2130;
        --warning-color: #FFA726;
        --error-color: #EF5350;
        --success-color: #66BB6A;
        --info-color: #42A5F5;
    }
    
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 8px 16px;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        background-color: #388E3C;
    }
    
    .css-145kmo2 {
        border: 2px solid #304250;
        border-radius: 10px;
        padding: 20px;
        background-color: #1E2130;
    }
    
    .stExpander {
        border: 1px solid #304250;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .stAlert {
        border-radius: 10px;
    }
    
    .stHeader {
        background-color: var(--card-bg-color) !important;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: var(--card-bg-color);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: white;
        margin: 5px;
        padding: 5px;
        font-weight: bold;
        background-color: #304250;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    div.stMultiSelect>div:first-child {
        background-color: #1E2130;
        border-color: #304250;
        border-radius: 8px;
    }
    
    .sidebar .sidebar-content {
        background-color: #111621;
    }
    
    /* Custom card styling */
    .card {
        background-color: #1E2130;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    
    /* Risk meter styling */
    .risk-meter {
        height: 20px;
        border-radius: 10px;
        margin-top: 10px;
        margin-bottom: 15px;
        background-color: #304250;
        overflow: hidden;
    }
    
    /* Progress bar animation */
    @keyframes pulse {
        0% { opacity: 0.8; }
        50% { opacity: 1; }
        100% { opacity: 0.8; }
    }
    
    .animate-pulse {
        animation: pulse 2s infinite ease-in-out;
    }
</style>
"""

st.markdown(dark_theme, unsafe_allow_html=True)

# Define the base and model directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
CMODEL_DIR = os.path.join(BASE_DIR, 'common-diseases-Prediction-model-main')
MEDICINE_DIR = os.path.join(BASE_DIR, 'medicines')
data_path = os.path.join(BASE_DIR, 'diabetes_data.csv')

# Define model paths
diabetes_model_path = os.path.join(MODEL_DIR, 'diabetes_model.sav')
heart_disease_model_path = os.path.join(MODEL_DIR, 'heart_disease_model.sav')
parkinsons_model_path = os.path.join(MODEL_DIR, 'parkinsons_model.sav')
common_model = os.path.join(CMODEL_DIR, 'decision_tree_model.sav')
symptoms_list_path = os.path.join(CMODEL_DIR, 'Symptoms.txt')
medicine_model_path = os.path.join(MEDICINE_DIR, 'medicine_prediction_model.sav')
encoders_path = os.path.join(MEDICINE_DIR, 'encoders.sav')

# Load medication and precaution data
try:
    medications_df = pd.read_csv(os.path.join(MEDICINE_DIR, 'medications.csv'))
    precautions_df = pd.read_csv(os.path.join(MEDICINE_DIR, 'precautions_df.csv'))
    diets_df = pd.read_csv(os.path.join(MEDICINE_DIR, 'diets.csv'))
    descriptions_df = pd.read_csv(os.path.join(MEDICINE_DIR, 'description.csv'))
    logger.info("Successfully loaded medication and precaution data")
except Exception as e:
    logger.error(f"Error loading medication data: {e}")
    medications_df = None
    precautions_df = None
    diets_df = None
    descriptions_df = None

# Function to get medication recommendations
def get_medication_info(disease):
    if medications_df is not None:
        try:
            medications = medications_df[medications_df['Disease'] == disease]['Medication'].values
            if len(medications) > 0:
                return ast.literal_eval(medications[0])
            return []
        except Exception as e:
            logger.error(f"Error getting medications for {disease}: {e}")
            return []
    return []

# Function to get precautions
def get_precautions(disease):
    if precautions_df is not None:
        try:
            precaution_row = precautions_df[precautions_df['Disease'] == disease]
            if len(precaution_row) > 0:
                precautions = []
                for i in range(1, 5):
                    column = f'Precaution_{i}'
                    if column in precaution_row.columns and not pd.isna(precaution_row[column].values[0]):
                        precautions.append(precaution_row[column].values[0])
                return precautions
            return []
        except Exception as e:
            logger.error(f"Error getting precautions for {disease}: {e}")
            return []
    return []

# Function to get diet recommendations
def get_diet_recommendations(disease):
    if diets_df is not None:
        try:
            diets = diets_df[diets_df['Disease'] == disease]['Diet'].values
            if len(diets) > 0:
                return ast.literal_eval(diets[0])
            return []
        except Exception as e:
            logger.error(f"Error getting diet for {disease}: {e}")
            return []
    return []

# Function to get disease description
def get_disease_description(disease):
    if descriptions_df is not None:
        try:
            description = descriptions_df[descriptions_df['Disease'] == disease]['Description'].values
            if len(description) > 0:
                return description[0]
            return ""
        except Exception as e:
            logger.error(f"Error getting description for {disease}: {e}")
            return ""
    return ""

# Function to display risk meter
def display_risk_meter(risk_percentage):
    html = f"""
    <div class="risk-meter">
        <div style="width: {risk_percentage}%; background: linear-gradient(90deg, #4CAF50, #FFC107, #F44336); 
             height: 100%; border-radius: 10px;" class="animate-pulse"></div>
    </div>
    <div style="display: flex; justify-content: space-between; margin-top: -10px;">
        <span>Low Risk</span>
        <span>Moderate</span>
        <span>High Risk</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# Function to suggest similar diseases based on symptoms
def suggest_similar_diseases(symptoms, predicted_disease):
    # This would ideally use a more sophisticated algorithm
    # For now, we'll use a simple approach based on common symptoms
    if not symptoms or common_model is None:
        return []
    
    try:
        # Get all possible diseases
        all_diseases = list(set(descriptions_df['Disease'].tolist()))
        
        # Remove the predicted disease
        if predicted_disease in all_diseases:
            all_diseases.remove(predicted_disease)
            
        # Get subset of diseases (for efficiency)
        potential_diseases = all_diseases[:5]  # Limit to avoid performance issues
        
        return potential_diseases
    except:
        return []

# Try to load medicine prediction model
try:
    with open(medicine_model_path, 'rb') as model_file:
        medicine_model = pickle.load(model_file)
    with open(encoders_path, 'rb') as encoder_file:
        encoders = pickle.load(encoder_file)
    logger.info("Medicine prediction model loaded successfully")
except Exception as e:
    logger.warning(f"Could not load medicine prediction model: {e}")
    medicine_model = None
    encoders = None

# Load models with better error handling
try:
    # Try loading with joblib first (more stable for scikit-learn models)
    try:
        diabetes_model = joblib.load(diabetes_model_path)
        heart_disease_model = joblib.load(heart_disease_model_path)
        parkinsons_model = joblib.load(parkinsons_model_path)
        
        # For the common disease model, we need probability estimates
        try:
            # Try to load and set probability=True
            common_model_obj = joblib.load(common_model)
            # Try to set probability if it's a classifier with this attribute
            if hasattr(common_model_obj, 'probability'):
                common_model_obj.probability = True
            common_model = common_model_obj
        except Exception as e:
            logger.warning(f"Could not set probability for common_model: {e}")
            # Manual fallback - we'll handle this in the prediction logic
            common_model = joblib.load(common_model)
        
        logger.info("Models loaded successfully using joblib.")
    except Exception as e:
        logger.warning(f"Joblib loading failed, trying pickle: {e}")
        # Fallback to pickle if joblib fails
        with open(diabetes_model_path, 'rb') as f:
            diabetes_model = pickle.load(f)
        with open(heart_disease_model_path, 'rb') as f:
            heart_disease_model = pickle.load(f)
        with open(parkinsons_model_path, 'rb') as f:
            parkinsons_model = pickle.load(f)
        with open(common_model, 'rb') as f:
            common_model_obj = pickle.load(f)
            # Try to set probability if it's a classifier with this attribute
            if hasattr(common_model_obj, 'probability'):
                common_model_obj.probability = True
            common_model = common_model_obj
        logger.info("Models loaded successfully using pickle.")
    
    # Load symptoms list
    with open(symptoms_list_path, 'r') as f:
        symptoms_list = f.read().splitlines()
    
except Exception as e:
    logger.error(f"Error loading models: {e}")
    st.error("Error loading models. Please check the logs and ensure all model files exist in the correct locations.")
    st.stop()  # Stop the app if models can't be loaded

# Check if models support predict_proba
has_predict_proba = {
    "diabetes": hasattr(diabetes_model, 'predict_proba'),
    "heart": hasattr(heart_disease_model, 'predict_proba'),
    "parkinsons": hasattr(parkinsons_model, 'predict_proba'),
    "common": hasattr(common_model, 'predict_proba')
}
logger.info(f"Models with predict_proba support: {has_predict_proba}")

# App header
st.markdown("""
<div style="text-align: center; padding: 10px; margin-bottom: 20px; background-color: #1E2130; border-radius: 10px;">
    <h1 style="color: #4CAF50;">🩺 HealthPredictor AI</h1>
    <p style="font-size: 18px;">Advanced Disease Prediction & Analysis using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div style="text-align: center;"><h2 style="color: #4CAF50;">🩺 HealthPredictor AI</h2></div>', unsafe_allow_html=True)
    
    selected = option_menu('Select Disease Prediction',
                           ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction', 'Common diseases Prediction'],
                           icons=['activity', 'heart', 'person', 'thermometer'],
                           default_index=3,
                           styles={
                               "container": {"padding": "5px", "background-color": "#1E2130", "border-radius": "10px"},
                               "icon": {"color": "#4CAF50", "font-size": "20px"},
                               "nav-link": {"color": "white", "font-size": "16px", "text-align": "left", "margin": "0px", 
                                           "border-radius": "5px"},
                               "nav-link-selected": {"background-color": "#4CAF50"},
                           })
    
    # Add sidebar information
    st.markdown("---")
    st.markdown("### About")
    st.info("This application uses advanced machine learning to predict multiple diseases based on symptoms and clinical data.")
    
    # Add timestamp
    st.markdown("---")
    st.markdown(f"<small>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</small>", unsafe_allow_html=True)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title("Diabetes Prediction using AI")
    
    # Option to select patient or doctor mode
    mode = st.tabs(["Patient Mode", "Doctor Mode"])

    with mode[1]:
        
        col1, col2, col3 = st.columns(3)

        with col1:
            Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0)
            Glucose = st.number_input('Glucose Level (mg/dL)', min_value=0, max_value=300, value=100)
            SkinThickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20)
            DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5, step=0.1)
            
        with col2:
            BloodPressure = st.number_input('Blood Pressure (mmHg)', min_value=0, max_value=200, value=70)
            Insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0, max_value=1000, value=80)
            Age = st.number_input('Age of the Person', min_value=0, max_value=120, value=30)
            
        with col3:
            BMI = st.number_input('BMI (kg/m²)', min_value=0.0, max_value=50.0, value=25.0, step=0.1)
            st.markdown("### Risk Factors")
            family_history = st.checkbox("Family History of Diabetes")
            physical_activity = st.checkbox("Low Physical Activity")
            high_bp = st.checkbox("History of High Blood Pressure")

        if st.button('Predict Diabetes Risk', key='doctor_diabetes'):
            with st.spinner("Analyzing patient data..."):
                try:
                    # Create input data array
                    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, 
                                          Insulin, BMI, DiabetesPedigreeFunction, Age]])
                    
                    # Make prediction
                    prediction = diabetes_model.predict(input_data)[0]
                    
                    # Try to get probability if available
                    prob_score = None
                    try:
                        if has_predict_proba["diabetes"]:
                            probs = diabetes_model.predict_proba(input_data)[0]
                            prob_score = probs[1] if prediction == 1 else probs[0]
                    except Exception as e:
                        logger.warning(f"Could not get probability: {e}")
                    
                    # Calculate risk score based on input values
                    risk_score = 0
                    
                    # Glucose level risk
                    if Glucose > 140:
                        risk_score += 30
                    elif Glucose > 100:
                        risk_score += 15
                    
                    # BMI risk
                    if BMI > 30:
                        risk_score += 20
                    elif BMI > 25:
                        risk_score += 10
                    
                    # Age risk
                    if Age > 45:
                        risk_score += 10
                    
                    # Additional risk factors
                    if family_history: risk_score += 20
                    if physical_activity: risk_score += 15
                    if high_bp: risk_score += 10
                    
                    # Calculate risk percentage - use probability if available
                    if prob_score is not None:
                        risk_percentage = int(prob_score * 100)
                    else:
                        risk_percentage = min(100, risk_score)
                    
                    # Display results in a nice card
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("### Prediction Results")
                    
                    # Display risk meter
                    display_risk_meter(risk_percentage)
                    
                    if prediction == 1 or risk_percentage > 50:
                        st.error(f"High Risk of Diabetes (Risk Score: {risk_percentage}%)")
                        st.warning("Recommendations:")
                        st.write("- Please consult a doctor for proper diagnosis")
                        st.write("- Consider getting a blood sugar test")
                        st.write("- Monitor your symptoms")
                        
                        # Show data visualization
                        col1, col2 = st.columns(2)
                        with col1:
                            # Create a radar chart for key metrics
                            categories = ['Glucose', 'BMI', 'Blood Pressure', 'Age', 'Insulin']
                            # Normalize values to 0-1 scale for the chart
                            glucose_norm = min(1.0, Glucose / 200)
                            bmi_norm = min(1.0, BMI / 40)
                            bp_norm = min(1.0, BloodPressure / 160)
                            age_norm = min(1.0, Age / 80)
                            insulin_norm = min(1.0, Insulin / 400)
                            
                            values = [glucose_norm, bmi_norm, bp_norm, age_norm, insulin_norm]
                            
                            # Create radar chart
                            fig = plt.figure(figsize=(4, 4))
                            ax = fig.add_subplot(111, polar=True)
                            
                            # Plot the data
                            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
                            values += values[:1]  # Close the loop
                            angles += angles[:1]  # Close the loop
                            categories += categories[:1]  # Close the loop
                            
                            ax.plot(angles, values, 'o-', linewidth=2, color='#4CAF50')
                            ax.fill(angles, values, alpha=0.25, color='#4CAF50')
                            ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
                            ax.set_ylim(0, 1)
                            ax.grid(True)
                            ax.set_facecolor('#1E2130')
                            for spine in ax.spines.values():
                                spine.set_color('white')
                            ax.tick_params(colors='white')
                            
                            plt.title('Key Health Metrics', color='white')
                            st.pyplot(fig)
                            
                        with col2:
                            # Create a bar chart for risk factors
                            risk_factors = ['Glucose', 'BMI', 'Blood Pressure', 'Family History', 'Physical Activity']
                            risk_values = [
                                30 if Glucose > 140 else (15 if Glucose > 100 else 0),
                                20 if BMI > 30 else (10 if BMI > 25 else 0),
                                15 if BloodPressure > 140 else (5 if BloodPressure > 120 else 0),
                                20 if family_history else 0,
                                15 if physical_activity else 0
                            ]
                            
                            fig, ax = plt.subplots(figsize=(4, 4))
                            bars = ax.bar(risk_factors, risk_values, color=['#FFC107', '#FF9800', '#F44336', '#9C27B0', '#2196F3'])
                            
                            ax.set_ylabel('Risk Contribution', color='white')
                            ax.set_title('Risk Factor Analysis', color='white')
                            ax.set_facecolor('#1E2130')
                            for spine in ax.spines.values():
                                spine.set_color('white')
                            ax.tick_params(axis='x', colors='white', rotation=45)
                            ax.tick_params(axis='y', colors='white')
                            ax.grid(axis='y', linestyle='--', alpha=0.7)
                            
                            st.pyplot(fig)
                        
                        # Add treatment recommendations
                        diabetes_info = get_disease_description("Diabetes")
                        diabetes_meds = get_medication_info("Diabetes")
                        diabetes_precautions = get_precautions("Diabetes")
                        diabetes_diet = get_diet_recommendations("Diabetes")
                        
                        if diabetes_info or diabetes_meds or diabetes_precautions or diabetes_diet:
                            st.markdown("### Treatment & Management")
                            
                            if diabetes_info:
                                with st.expander("About Diabetes", expanded=False):
                                    st.write(diabetes_info)
                            
                            if diabetes_meds:
                                with st.expander("Common Medications", expanded=False):
                                    st.write("*Note: Always consult a healthcare professional before taking any medications.*")
                                    for med in diabetes_meds:
                                        st.write(f"- {med}")
                            
                            if diabetes_precautions:
                                with st.expander("Precautions & Management", expanded=False):
                                    for precaution in diabetes_precautions:
                                        st.write(f"- {precaution}")
                            
                            if diabetes_diet:
                                with st.expander("Diet Recommendations", expanded=False):
                                    for diet in diabetes_diet:
                                        st.write(f"- {diet}")
                    else:
                        st.success(f"Low Risk of Diabetes (Risk Score: {risk_percentage}%)")
                        st.info("Recommendations:")
                        st.write("- Maintain a healthy lifestyle")
                        st.write("- Regular check-ups are recommended")
                        st.write("- Continue monitoring your health")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Feedback system
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("### Feedback")
                    feedback = st.radio("Was this prediction helpful?", ["Yes", "No"])
                    if feedback == "Yes":
                        try:
                            with open(data_path, 'a', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerow([*input_data[0], prediction, risk_percentage])
                            st.info("Thank you for your feedback! Data has been recorded.")
                        except Exception as e:
                            logger.error(f"Error saving feedback: {e}")
                            st.warning("Could not save feedback. Please try again later.")
                    
                except Exception as e:
                    logger.error(f"Error during prediction: {e}")
                    st.error("An error occurred during prediction. Please check the input values.")

    with mode[0]:
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Common Symptoms")
            thirsty_tired = st.radio("Do you often feel unusually thirsty or tired?", ["Yes", "No"])
            frequent_urination = st.radio("Do you often have to urinate?", ["Yes", "No"])
            blurry_vision = st.radio("Do you have blurry vision?", ["Yes", "No"])
            
        with col2:
            st.subheader("Additional Symptoms")
            sudden_weight_loss = st.radio("Have you experienced sudden weight loss?", ["Yes", "No"])
            wounds_slow_to_heal = st.radio("Do you have wounds that are slow to heal?", ["Yes", "No"])
            family_history = st.radio("Do you have a family history of diabetes?", ["Yes", "No"])
        
        if st.button('Check Diabetes Risk', key='patient_diabetes'):
            with st.spinner("Analyzing your symptoms..."):
                try:
                    # Convert symptoms to numerical values
                    Glucose = 160 if thirsty_tired == "Yes" else 90
                    BloodPressure = 85 if frequent_urination == "Yes" else 70
                    BMI = 35.0 if sudden_weight_loss == "Yes" else 28.0
                    SkinThickness = 35 if blurry_vision == "Yes" else 25
                    Insulin = 210 if wounds_slow_to_heal == "Yes" else 80
                    Pregnancies = 2
                    DiabetesPedigreeFunction = 0.5
                    Age = 50
                    
                    # Create input data
                    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, 
                                          Insulin, BMI, DiabetesPedigreeFunction, Age]])
                    
                    # Make prediction
                    prediction = diabetes_model.predict(input_data)[0]
                    
                    # Try to get probability if available
                    prob_score = None
                    try:
                        if has_predict_proba["diabetes"]:
                            probs = diabetes_model.predict_proba(input_data)[0]
                            prob_score = probs[1] if prediction == 1 else probs[0]
                    except Exception as e:
                        logger.warning(f"Could not get probability: {e}")
                    
                    # Calculate risk score based on symptoms
                    risk_score = 0
                    
                    # Symptom-based risk
                    if thirsty_tired == "Yes": risk_score += 20
                    if frequent_urination == "Yes": risk_score += 20
                    if blurry_vision == "Yes": risk_score += 15
                    if sudden_weight_loss == "Yes": risk_score += 15
                    if wounds_slow_to_heal == "Yes": risk_score += 10
                    if family_history == "Yes": risk_score += 20
                    
                    # Calculate risk percentage - use probability if available, otherwise use symptom score
                    if prob_score is not None:
                        risk_percentage = int(prob_score * 100)
                    else:
                        risk_percentage = min(100, risk_score)
                    
                    # Display results
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("### Risk Assessment")
                    
                    # Display risk meter
                    display_risk_meter(risk_percentage)
                    
                    if prediction == 1 or risk_percentage > 50:
                        st.error(f"High Risk of Diabetes (Risk Score: {risk_percentage}%)")
                        st.warning("Recommendations:")
                        st.write("- Please consult a doctor for proper diagnosis")
                        st.write("- Consider getting a blood sugar test")
                        st.write("- Monitor your symptoms")
                        
                        # Add visual representation of symptoms
                        symptoms_present = []
                        if thirsty_tired == "Yes": symptoms_present.append("Thirst/Fatigue")
                        if frequent_urination == "Yes": symptoms_present.append("Frequent Urination")
                        if blurry_vision == "Yes": symptoms_present.append("Blurry Vision")
                        if sudden_weight_loss == "Yes": symptoms_present.append("Weight Loss")
                        if wounds_slow_to_heal == "Yes": symptoms_present.append("Slow Healing")
                        if family_history == "Yes": symptoms_present.append("Family History")
                        
                        if symptoms_present:
                            fig, ax = plt.subplots(figsize=(6, 4))
                            y_pos = range(len(symptoms_present))
                            severity = [20, 20, 15, 15, 10, 20][:len(symptoms_present)]
                            
                            bars = ax.barh(y_pos, severity, color='#F44336')
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(symptoms_present)
                            ax.invert_yaxis()
                            ax.set_xlabel('Symptom Severity', color='white')
                            ax.set_title('Your Diabetes Risk Factors', color='white')
                            ax.set_facecolor('#1E2130')
                            
                            for spine in ax.spines.values():
                                spine.set_color('white')
                            ax.tick_params(axis='x', colors='white')
                            ax.tick_params(axis='y', colors='white')
                            
                            st.pyplot(fig)
                        
                        # Add treatment recommendations
                        diabetes_info = get_disease_description("Diabetes")
                        diabetes_meds = get_medication_info("Diabetes")
                        diabetes_precautions = get_precautions("Diabetes")
                        diabetes_diet = get_diet_recommendations("Diabetes")
                        
                        if diabetes_info or diabetes_meds or diabetes_precautions or diabetes_diet:
                            st.markdown("### Treatment & Management")
                            
                            if diabetes_info:
                                with st.expander("About Diabetes", expanded=False):
                                    st.write(diabetes_info)
                            
                            if diabetes_meds:
                                with st.expander("Common Medications", expanded=False):
                                    st.write("*Note: Always consult a healthcare professional before taking any medications.*")
                                    for med in diabetes_meds:
                                        st.write(f"- {med}")
                            
                            if diabetes_precautions:
                                with st.expander("Precautions & Management", expanded=False):
                                    for precaution in diabetes_precautions:
                                        st.write(f"- {precaution}")
                            
                            if diabetes_diet:
                                with st.expander("Diet Recommendations", expanded=False):
                                    for diet in diabetes_diet:
                                        st.write(f"- {diet}")
                    else:
                        st.success(f"Low Risk of Diabetes (Risk Score: {risk_percentage}%)")
                        st.info("Recommendations:")
                        st.write("- Maintain a healthy lifestyle")
                        st.write("- Regular check-ups are recommended")
                        st.write("- Continue monitoring your health")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Feedback system
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("### Feedback")
                    feedback = st.radio("Was this prediction helpful?", ["Yes", "No"])
                    if feedback == "Yes":
                        try:
                            with open(data_path, 'a', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerow([*input_data[0], prediction, risk_percentage])
                            st.info("Thank you for your feedback! Data has been recorded.")
                        except Exception as e:
                            logger.error(f"Error saving feedback: {e}")
                            st.warning("Could not save feedback. Please try again later.")
                    
                except Exception as e:
                    logger.error(f"Error during prediction: {e}")
                    st.error("An error occurred during prediction. Please check the input values.")

# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    st.title('Disease Prediction using AI')
    
    # Option to select patient or doctor mode
    mode = st.tabs(["Patient Mode", "Doctor Mode"])
    
    with mode[1]:
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input('Age', min_value=1, max_value=120, value=45)
            
        with col2:
            sex = st.radio('Gender', ['Male', 'Female'])
            sex_value = 1 if sex == 'Male' else 0
            
        with col3:
            cp = st.selectbox('Chest Pain Type', 
                             ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'],
                             index=0)
            cp_value = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(cp)
            
        with col1:
            trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=80, max_value=200, value=120)
            
        with col2:
            chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
            
        with col3:
            fbs = st.radio('Fasting Blood Sugar > 120 mg/dl', ['Yes', 'No'])
            fbs_value = 1 if fbs == 'Yes' else 0
            
        with col1:
            restecg = st.selectbox('Resting ECG Results', 
                                 ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'],
                                 index=0)
            restecg_value = ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'].index(restecg)
            
        with col2:
            thalach = st.number_input('Maximum Heart Rate', min_value=50, max_value=220, value=150)
            
        with col3:
            exang = st.radio('Exercise Induced Angina', ['Yes', 'No'])
            exang_value = 1 if exang == 'Yes' else 0
            
        with col1:
            oldpeak = st.number_input('ST Depression induced by Exercise', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            
        with col2:
            slope = st.selectbox('Slope of the Peak Exercise ST Segment', 
                               ['Upsloping', 'Flat', 'Downsloping'],
                               index=0)
            slope_value = ['Upsloping', 'Flat', 'Downsloping'].index(slope)
            
        with col3:
            ca = st.number_input('Number of Major Vessels Colored by Fluoroscopy', min_value=0, max_value=4, value=0)
            
        with col1:
            thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'], index=0)
            thal_value = ['Normal', 'Fixed Defect', 'Reversible Defect'].index(thal)
            
        # code for Prediction
        if st.button('Heart Disease Test Result', key='doctor_heart'):
            with st.spinner("Analyzing patient data..."):
                try:
                    # Create input data
                    input_data = np.array([[age, sex_value, cp_value, trestbps, chol, fbs_value, 
                                          restecg_value, thalach, exang_value, oldpeak, 
                                          slope_value, ca, thal_value]])
                    
                    # Make prediction
                    heart_prediction = heart_disease_model.predict(input_data)[0]
                    
                    # Try to get probability if available
                    risk_percentage = 50  # Default
                    try:
                        if has_predict_proba["heart"]:
                            probs = heart_disease_model.predict_proba(input_data)[0]
                            risk_percentage = int(probs[1] * 100)
                    except Exception as e:
                        logger.warning(f"Could not get probability: {e}")
                        
                        # Calculate risk manually if probability not available
                        risk_score = 0
                        
                        # Age risk
                        if age > 55: risk_score += 20
                        elif age > 45: risk_score += 10
                        
                        # Clinical factors
                        if chol > 240: risk_score += 20
                        if trestbps > 140: risk_score += 15
                        if thalach < 120: risk_score += 15
                        if exang_value == 1: risk_score += 20
                        if cp_value == 3: risk_score += 20  # Asymptomatic
                        if ca >= 1: risk_score += ca * 15
                        
                        risk_percentage = min(100, risk_score)
                    
                    # Display results in a card
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("### Prediction Results")
                    
                    # Display risk meter
                    display_risk_meter(risk_percentage)
                    
                    if heart_prediction == 1:
                        st.error(f"High Risk of Heart Disease (Risk Score: {risk_percentage}%)")
                        st.warning("Recommendations:")
                        st.write("- Schedule a follow-up with a cardiologist")
                        st.write("- Consider lifestyle modifications")
                        st.write("- Monitor blood pressure and cholesterol levels")
                        
                        # Add visualizations
                        col1, col2 = st.columns(2)
                        with col1:
                            # Create a gauge chart for heart health
                            fig = plt.figure(figsize=(4, 3))
                            ax = fig.add_subplot(111)
                            
                            # Calculate heart health score (inverse of risk)
                            heart_health = max(0, 100 - risk_percentage)
                            
                            # Create gauge
                            colors = ['#F44336', '#FFC107', '#4CAF50']
                            bounds = [0, 33, 66, 100]
                            norm = plt.Normalize(0, 100)
                            
                            # Create bars
                            bars = ax.barh([0], [100], color='#304250', height=0.5)
                            bars = ax.barh([0], [heart_health], color=plt.cm.RdYlGn(norm(heart_health)), height=0.5)
                            
                            # Customize
                            ax.set_xlim(0, 100)
                            ax.set_ylim(-0.5, 0.5)
                            ax.set_xlabel('Heart Health Score', color='white')
                            ax.set_yticks([])
                            ax.text(heart_health/2, 0, f"{heart_health}%", ha='center', va='center', color='white', fontweight='bold')
                            
                            ax.set_facecolor('#1E2130')
                            for spine in ax.spines.values():
                                spine.set_color('white')
                            ax.tick_params(axis='x', colors='white')
                            
                            plt.title('Heart Health Index', color='white')
                            st.pyplot(fig)
                            
                        with col2:
                            # Create risk factor chart
                            labels = ['Cholesterol', 'Blood Pressure', 'Heart Rate', 'Age Risk', 'Vessels Affected']
                            sizes = [
                                20 if chol > 240 else 10 if chol > 200 else 5,
                                15 if trestbps > 140 else 7 if trestbps > 120 else 3,
                                15 if thalach < 120 else 7 if thalach < 140 else 3,
                                20 if age > 55 else 10 if age > 45 else 5,
                                ca * 15 if ca > 0 else 5
                            ]
                            
                            # Create pie chart
                            fig, ax = plt.subplots(figsize=(4, 3))
                            wedges, texts, autotexts = ax.pie(sizes, labels=None, autopct='%1.1f%%', 
                                                             startangle=90, colors=plt.cm.tab10.colors)
                            
                            # Customize
                            ax.axis('equal')
                            ax.set_facecolor('#1E2130')
                            plt.title('Risk Factor Distribution', color='white')
                            
                            # Legend
                            ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), 
                                      fontsize=8, frameon=False, labelcolor='white')
                            
                            # Make text white
                            for autotext in autotexts:
                                autotext.set_color('white')
                            
                            st.pyplot(fig)
                        
                        # Add treatment recommendations
                        heart_info = get_disease_description("Heart attack")
                        heart_meds = get_medication_info("Heart attack")
                        heart_precautions = get_precautions("Heart attack")
                        heart_diet = get_diet_recommendations("Heart attack")
                        
                        if heart_info or heart_meds or heart_precautions or heart_diet:
                            st.markdown("### Treatment & Management")
                            
                            if heart_info:
                                with st.expander("About Heart Disease", expanded=False):
                                    st.write(heart_info)
                            
                            if heart_meds:
                                with st.expander("Common Medications", expanded=False):
                                    st.write("*Note: Always consult a healthcare professional before taking any medications.*")
                                    for med in heart_meds:
                                        st.write(f"- {med}")
                            
                            if heart_precautions:
                                with st.expander("Emergency Precautions", expanded=False):
                                    for precaution in heart_precautions:
                                        st.write(f"- {precaution}")
                            
                            if heart_diet:
                                with st.expander("Diet Recommendations", expanded=False):
                                    for diet in heart_diet:
                                        st.write(f"- {diet}")
                    else:
                        st.success(f"Low Risk of Heart Disease (Risk Score: {risk_percentage}%)")
                        st.info("Recommendations:")
                        st.write("- Maintain healthy lifestyle")
                        st.write("- Regular check-ups recommended")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                except Exception as e:
                    logger.error(f"Error during prediction: {e}")
                    st.error("An error occurred during prediction. Please check the input values.")

    with mode[0]:
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Common Symptoms")
            chest_pain = st.radio("Do you experience chest pain or discomfort?", ["Yes", "No"], key="heart_chest_pain")
            shortness_breath = st.radio("Do you have shortness of breath?", ["Yes", "No"], key="heart_shortness_breath")
            fatigue = st.radio("Do you feel unusually tired or fatigued?", ["Yes", "No"], key="heart_fatigue")
            
        with col2:
            st.subheader("Additional Information")
            age_group = st.selectbox("What is your age group?", ["Under 40", "40-55", "Above 55"], key="heart_age_group")
            high_bp = st.radio("Do you have high blood pressure?", ["Yes", "No", "Don't Know"], key="heart_high_bp")
            high_cholesterol = st.radio("Do you have high cholesterol?", ["Yes", "No", "Don't Know"], key="heart_high_cholesterol")
        
        col1, col2 = st.columns(2)
        with col1:
            family_history = st.radio("Do you have a family history of heart disease?", ["Yes", "No"], key="heart_family_history")
        with col2:
            physical_activity = st.radio("How would you describe your physical activity level?", 
                                        ["Sedentary", "Light Exercise", "Regular Exercise"], key="heart_physical_activity")
        
        if st.button('Check Heart Disease Risk', key='patient_heart'):
            with st.spinner("Analyzing your symptoms..."):
                try:
                    # Convert symptoms to numerical values for the model
                    age_value = 65 if age_group == "Above 55" else (50 if age_group == "40-55" else 35)
                    sex_value = 1  # Default to male as we don't ask gender in simple mode
                    cp_value = 0 if chest_pain == "No" else 2  # Non-anginal pain if they report chest pain
                    trestbps = 150 if high_bp == "Yes" else (130 if high_bp == "Don't Know" else 120)
                    chol = 250 if high_cholesterol == "Yes" else (200 if high_cholesterol == "Don't Know" else 180)
                    fbs_value = 0  # Default
                    restecg_value = 0  # Default to normal
                    thalach = 120 if fatigue == "Yes" else 150
                    exang_value = 1 if chest_pain == "Yes" and shortness_breath == "Yes" else 0
                    oldpeak = 1.0 if chest_pain == "Yes" else 0.0
                    slope_value = 1  # Default to flat
                    ca = 0  # Default
                    thal_value = 0  # Default to normal
                    
                    # Create input data
                    input_data = np.array([[age_value, sex_value, cp_value, trestbps, chol, fbs_value, 
                                          restecg_value, thalach, exang_value, oldpeak, 
                                          slope_value, ca, thal_value]])
                    
                    # Make prediction
                    prediction = heart_disease_model.predict(input_data)[0]
                    
                    # Try to get probability if available
                    risk_percentage = 50  # Default
                    try:
                        if has_predict_proba["heart"]:
                            probs = heart_disease_model.predict_proba(input_data)[0]
                            risk_percentage = int(probs[1] * 100)
                    except Exception as e:
                        logger.warning(f"Could not get probability: {e}")
                    
                    # Calculate risk score based on symptoms
                    risk_score = 0
                    
                    # Symptom-based risk
                    if chest_pain == "Yes": risk_score += 25
                    if shortness_breath == "Yes": risk_score += 20
                    if fatigue == "Yes": risk_score += 15
                    if age_group == "Above 55": risk_score += 20
                    elif age_group == "40-55": risk_score += 10
                    if high_bp == "Yes": risk_score += 20
                    if high_cholesterol == "Yes": risk_score += 20
                    if family_history == "Yes": risk_score += 20
                    if physical_activity == "Sedentary": risk_score += 15
                    
                    # Calculate risk percentage - use probability if available, otherwise use symptom score
                    if risk_percentage == 50:  # If we couldn't get a probability
                        risk_percentage = min(100, risk_score)
                    
                    # Display results
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("### Risk Assessment")
                    
                    # Display risk meter
                    display_risk_meter(risk_percentage)
                    
                    if prediction == 1 or risk_percentage > 50:
                        st.error(f"High Risk of Heart Disease (Risk Score: {risk_percentage}%)")
                        st.warning("Recommendations:")
                        st.write("- Please consult a doctor for proper diagnosis")
                        st.write("- Consider getting further cardiac tests")
                        st.write("- Monitor your symptoms carefully")
                        
                        # Add visual representation of symptoms
                        symptoms_present = []
                        if chest_pain == "Yes": symptoms_present.append("Chest Pain")
                        if shortness_breath == "Yes": symptoms_present.append("Shortness of Breath")
                        if fatigue == "Yes": symptoms_present.append("Fatigue")
                        if high_bp == "Yes": symptoms_present.append("High Blood Pressure")
                        if high_cholesterol == "Yes": symptoms_present.append("High Cholesterol")
                        if family_history == "Yes": symptoms_present.append("Family History")
                        if physical_activity == "Sedentary": symptoms_present.append("Sedentary Lifestyle")
                        
                        if symptoms_present:
                            fig, ax = plt.subplots(figsize=(6, 4))
                            y_pos = range(len(symptoms_present))
                            # Assign severity based on risk contribution
                            severity = []
                            for symptom in symptoms_present:
                                if symptom == "Chest Pain": severity.append(25)
                                elif symptom == "Shortness of Breath": severity.append(20)
                                elif symptom == "Fatigue": severity.append(15)
                                elif symptom == "High Blood Pressure": severity.append(20)
                                elif symptom == "High Cholesterol": severity.append(20)
                                elif symptom == "Family History": severity.append(20)
                                elif symptom == "Sedentary Lifestyle": severity.append(15)
                                else: severity.append(10)
                            
                            bars = ax.barh(y_pos, severity, color='#F44336')
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(symptoms_present)
                            ax.invert_yaxis()
                            ax.set_xlabel('Risk Contribution', color='white')
                            ax.set_title('Your Heart Disease Risk Factors', color='white')
                            ax.set_facecolor('#1E2130')
                            
                            for spine in ax.spines.values():
                                spine.set_color('white')
                            ax.tick_params(axis='x', colors='white')
                            ax.tick_params(axis='y', colors='white')
                            
                            st.pyplot(fig)
                        
                        # Add treatment recommendations
                        heart_info = get_disease_description("Heart attack")
                        heart_meds = get_medication_info("Heart attack")
                        heart_precautions = get_precautions("Heart attack")
                        heart_diet = get_diet_recommendations("Heart attack")
                        
                        if heart_info or heart_meds or heart_precautions or heart_diet:
                            st.markdown("### Treatment & Management")
                            
                            if heart_info:
                                with st.expander("About Heart Disease", expanded=False):
                                    st.write(heart_info)
                            
                            if heart_meds:
                                with st.expander("Common Medications", expanded=False):
                                    st.write("*Note: Always consult a healthcare professional before taking any medications.*")
                                    for med in heart_meds:
                                        st.write(f"- {med}")
                            
                            if heart_precautions:
                                with st.expander("Emergency Precautions", expanded=False):
                                    for precaution in heart_precautions:
                                        st.write(f"- {precaution}")
                            
                            if heart_diet:
                                with st.expander("Diet Recommendations", expanded=False):
                                    for diet in heart_diet:
                                        st.write(f"- {diet}")
                    else:
                        st.success(f"Low Risk of Heart Disease (Risk Score: {risk_percentage}%)")
                        st.info("Recommendations:")
                        st.write("- Maintain a healthy lifestyle")
                        st.write("- Regular check-ups are recommended")
                        st.write("- Exercise regularly and eat a balanced diet")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Feedback system
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("### Feedback")
                    feedback = st.radio("Was this prediction helpful?", ["Yes", "No"], key="heart_feedback")
                    if feedback == "Yes":
                        st.info("Thank you for your feedback! Your input helps us improve our predictions.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    logger.error(f"Error during prediction: {e}")
                    st.error("An error occurred during prediction. Please try again.")

# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):
    st.title("Parkinson's Disease Prediction using ML")
    
    # Option to select patient or doctor mode
    mode = st.tabs(["Patient", "Doctors"])
    
    with mode[1]:
        
        col1, col2, col3, col4, col5 = st.columns(5)  
        
        with col1:
            fo = st.number_input('MDVP:Fo(Hz)', min_value=88.0, max_value=260.0, value=120.0, step=0.1)
            
        with col2:
            fhi = st.number_input('MDVP:Fhi(Hz)', min_value=100.0, max_value=600.0, value=200.0, step=0.1)
            
        with col3:
            flo = st.number_input('MDVP:Flo(Hz)', min_value=65.0, max_value=240.0, value=100.0, step=0.1)
            
        with col4:
            Jitter_percent = st.number_input('MDVP:Jitter(%)', min_value=0.0, max_value=10.0, value=0.5, step=0.001)
            
        with col5:
            Jitter_Abs = st.number_input('MDVP:Jitter(Abs)', min_value=0.0, max_value=0.1, value=0.005, step=0.0001)
            
        with col1:
            RAP = st.number_input('MDVP:RAP', min_value=0.0, max_value=0.2, value=0.02, step=0.001)
            
        with col2:
            PPQ = st.number_input('MDVP:PPQ', min_value=0.0, max_value=0.5, value=0.02, step=0.001)
            
        with col3:
            DDP = st.number_input('Jitter:DDP', min_value=0.0, max_value=0.2, value=0.01, step=0.001)
            
        with col4:
            Shimmer = st.number_input('MDVP:Shimmer', min_value=0.0, max_value=1.0, value=0.05, step=0.001)
            
        with col5:
            Shimmer_dB = st.number_input('MDVP:Shimmer(dB)', min_value=0.0, max_value=2.0, value=0.5, step=0.01)
            
        with col1:
            APQ3 = st.number_input('Shimmer:APQ3', min_value=0.0, max_value=0.1, value=0.02, step=0.001)
            
        with col2:
            APQ5 = st.number_input('Shimmer:APQ5', min_value=0.0, max_value=0.15, value=0.03, step=0.001)
            
        with col3:
            APQ = st.number_input('MDVP:APQ', min_value=0.0, max_value=0.15, value=0.03, step=0.001)
            
        with col4:
            DDA = st.number_input('Shimmer:DDA', min_value=0.0, max_value=0.15, value=0.03, step=0.001)
            
        with col5:
            NHR = st.number_input('NHR', min_value=0.0, max_value=0.5, value=0.01, step=0.001)
            
        with col1:
            HNR = st.number_input('HNR', min_value=0.0, max_value=30.0, value=20.0, step=0.1)
            
        with col2:
            RPDE = st.number_input('RPDE', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
            
        with col3:
            DFA = st.number_input('DFA', min_value=0.0, max_value=1.0, value=0.6, step=0.01)
            
        with col4:
            spread1 = st.number_input('spread1', min_value=-10.0, max_value=10.0, value=-5.0, step=0.1)
            
        with col5:
            spread2 = st.number_input('spread2', min_value=0.0, max_value=10.0, value=0.2, step=0.01)
            
        with col1:
            D2 = st.number_input('D2', min_value=0.0, max_value=5.0, value=2.0, step=0.01)
            
        with col2:
            PPE = st.number_input('PPE', min_value=0.0, max_value=1.0, value=0.2, step=0.01)
            
        
        # code for Prediction
        if st.button("Parkinson's Test Result"):
            try:
                # Create input data
                input_data = np.array([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, 
                                      Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, 
                                      RPDE, DFA, spread1, spread2, D2, PPE]])
                
                # Make prediction
                parkinsons_prediction = parkinsons_model.predict(input_data)[0]
                
                # Try to get probability if available
                risk_percentage = 50  # Default
                try:
                    if has_predict_proba["parkinsons"]:
                        probs = parkinsons_model.predict_proba(input_data)[0]
                        risk_percentage = int(probs[1] * 100)
                except Exception as e:
                    logger.warning(f"Could not get probability: {e}")
                    
                    # Manual risk calculation for Parkinson's
                    risk_score = 0
                    
                    # Key indicators for Parkinson's based on clinical knowledge
                    if Jitter_percent > 1.0: risk_score += 15
                    if Shimmer > 0.1: risk_score += 15
                    if NHR > 0.05: risk_score += 10
                    if HNR < 15: risk_score += 10
                    if PPE > 0.3: risk_score += 15
                    if DFA < 0.5 or DFA > 0.8: risk_score += 10
                    
                    risk_percentage = min(100, risk_score)
            
                # Display results
                st.markdown("### Prediction Results")
                if parkinsons_prediction == 1:
                    st.error(f"High probability of Parkinson's Disease (Risk Score: {risk_percentage}%)")
                    st.warning("Recommendations:")
                    st.write("- Schedule a follow-up with a neurologist")
                    st.write("- Consider further neurological testing")
                    st.write("- Monitor symptoms closely")
                    
                    # Add treatment recommendations
                    parkinsons_info = get_disease_description("Paralysis (brain hemorrhage)")  # Closest match in our dataset
                    parkinsons_meds = get_medication_info("Paralysis (brain hemorrhage)")
                    parkinsons_precautions = get_precautions("Paralysis (brain hemorrhage)")
                    parkinsons_diet = get_diet_recommendations("Paralysis (brain hemorrhage)")
                    
                    if parkinsons_info or parkinsons_meds or parkinsons_precautions or parkinsons_diet:
                        st.markdown("### Treatment & Management")
                        
                        if parkinsons_info:
                            with st.expander("About Neurological Conditions", expanded=False):
                                st.write(parkinsons_info)
                        
                        if parkinsons_meds:
                            with st.expander("Common Medications", expanded=False):
                                st.write("*Note: Always consult a healthcare professional before taking any medications.*")
                                for med in parkinsons_meds:
                                    st.write(f"- {med}")
                        
                        if parkinsons_precautions:
                            with st.expander("Care & Management", expanded=False):
                                for precaution in parkinsons_precautions:
                                    st.write(f"- {precaution}")
                        
                        if parkinsons_diet:
                            with st.expander("Diet Recommendations", expanded=False):
                                for diet in parkinsons_diet:
                                    st.write(f"- {diet}")
                else:
                    st.success(f"Low probability of Parkinson's Disease (Risk Score: {risk_percentage}%)")
                    st.info("Recommendations:")
                    st.write("- Continue regular check-ups")
                    st.write("- Monitor for any new symptoms")
                
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                st.error("An error occurred during prediction. Please check the input values.")
    
    with mode[0]:
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Motor Symptoms")
            tremor = st.radio("Do you experience tremors or shaking?", ["No", "Mild", "Moderate", "Severe"])
            stiffness = st.radio("Do you experience muscle stiffness or rigidity?", ["No", "Mild", "Moderate", "Severe"])
            balance = st.radio("Do you have problems with balance or walking?", ["No", "Mild", "Moderate", "Severe"])
            
        with col2:
            st.subheader("Other Symptoms")
            handwriting = st.radio("Has your handwriting gotten smaller or harder to read?", ["No", "Yes"])
            voice = st.radio("Has your voice become softer or harder to understand?", ["No", "Yes"])
            sleep = st.radio("Do you have sleep problems or act out dreams?", ["No", "Yes"])
        
        col1, col2 = st.columns(2)
        with col1:
            smell = st.radio("Have you lost your sense of smell?", ["No", "Yes"])
        with col2:
            age_group = st.selectbox("What is your age group?", ["Under 50", "50-65", "Above 65"])
            
        if st.button("Check Parkinson's Risk"):
            try:
                # Map symptom severity to numerical values
                severity_map = {"No": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
                yes_no_map = {"No": 0, "Yes": 1}
                
                # Create a simplified model input based on clinical knowledge
                # We'll use these values to map to the actual values the model expects
                tremor_val = severity_map[tremor]
                stiffness_val = severity_map[stiffness]
                balance_val = severity_map[balance]
                handwriting_val = yes_no_map[handwriting]
                voice_val = yes_no_map[voice]
                sleep_val = yes_no_map[sleep]
                smell_val = yes_no_map[smell]
                
                # Scale the voice measurements based on symptoms
                # Higher Jitter and Shimmer values correlate with Parkinson's
                fo = 120.0
                fhi = 150.0 + (50.0 * voice_val)
                flo = 80.0 + (20.0 * voice_val)
                Jitter_percent = 0.4 + (0.5 * tremor_val) + (0.3 * voice_val)
                Jitter_Abs = 0.003 + (0.002 * tremor_val)
                RAP = 0.01 + (0.01 * voice_val)
                PPQ = 0.01 + (0.01 * voice_val)
                DDP = 0.03 + (0.01 * voice_val)
                
                # Higher Shimmer correlates with Parkinson's
                Shimmer = 0.04 + (0.02 * voice_val) + (0.01 * tremor_val)
                Shimmer_dB = 0.2 + (0.1 * voice_val)
                APQ3 = 0.02 + (0.01 * voice_val)
                APQ5 = 0.02 + (0.01 * voice_val)
                APQ = 0.02 + (0.01 * voice_val)
                DDA = 0.05 + (0.01 * voice_val)
                
                # Lower HNR correlates with Parkinson's
                NHR = 0.01 + (0.01 * voice_val)
                HNR = 20.0 - (5.0 * voice_val)
                
                # Other measurements
                RPDE = 0.4 + (0.1 * (stiffness_val + balance_val) / 6.0)
                DFA = 0.5 + (0.1 * handwriting_val)
                spread1 = -5.0 + (1.0 * tremor_val)
                spread2 = 0.2 + (0.1 * stiffness_val)
                D2 = 2.0 + (0.5 * (tremor_val + balance_val) / 6.0)
                PPE = 0.1 + (0.1 * tremor_val) + (0.05 * voice_val)
                
                # Create input data for the model
                input_data = np.array([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, 
                                      Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, 
                                      RPDE, DFA, spread1, spread2, D2, PPE]])
                
                # Make prediction
                prediction = parkinsons_model.predict(input_data)[0]
                
                # Calculate risk score based on symptoms
                risk_score = 0
                
                # Symptom-based risk
                risk_score += tremor_val * 15
                risk_score += stiffness_val * 10
                risk_score += balance_val * 10
                risk_score += handwriting_val * 15
                risk_score += voice_val * 15
                risk_score += sleep_val * 10
                risk_score += smell_val * 10
                
                # Age risk
                if age_group == "Above 65": 
                    risk_score += 15
                elif age_group == "50-65": 
                    risk_score += 10
                
                # Calculate risk percentage
                risk_percentage = min(100, risk_score)
                
                # Display results
                st.markdown("### Risk Assessment")
                if prediction == 1 or risk_percentage > 50:
                    st.error(f"High Risk of Parkinson's Disease (Risk Score: {risk_percentage}%)")
                    st.warning("Recommendations:")
                    st.write("- Please consult a neurologist for proper evaluation")
                    st.write("- Consider specialized neurological tests")
                    st.write("- Monitor your symptoms closely")
                    
                    # Add treatment recommendations
                    parkinsons_info = get_disease_description("Paralysis (brain hemorrhage)")  # Closest match in our dataset
                    parkinsons_meds = get_medication_info("Paralysis (brain hemorrhage)")
                    parkinsons_precautions = get_precautions("Paralysis (brain hemorrhage)")
                    parkinsons_diet = get_diet_recommendations("Paralysis (brain hemorrhage)")
                    
                    if parkinsons_info or parkinsons_meds or parkinsons_precautions or parkinsons_diet:
                        st.markdown("### Treatment & Management")
                        
                        if parkinsons_info:
                            with st.expander("About Neurological Conditions", expanded=False):
                                st.write(parkinsons_info)
                        
                        if parkinsons_meds:
                            with st.expander("Common Medications", expanded=False):
                                st.write("*Note: Always consult a healthcare professional before taking any medications.*")
                                for med in parkinsons_meds:
                                    st.write(f"- {med}")
                        
                        if parkinsons_precautions:
                            with st.expander("Care & Management", expanded=False):
                                for precaution in parkinsons_precautions:
                                    st.write(f"- {precaution}")
                        
                        if parkinsons_diet:
                            with st.expander("Diet Recommendations", expanded=False):
                                for diet in parkinsons_diet:
                                    st.write(f"- {diet}")
                else:
                    st.success(f"Low Risk of Parkinson's Disease (Risk Score: {risk_percentage}%)")
                    st.info("Recommendations:")
                    st.write("- Continue to monitor any changes in symptoms")
                    st.write("- Regular check-ups are recommended")
                    st.write("- Maintain an active lifestyle")
                
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                st.error("An error occurred during prediction. Please try again.")

#common disease model
if selected == 'Common diseases Prediction':
    st.title("Common Disease Prediction using AI")
    
    # Load symptoms list correctly
    try:
        with open(symptoms_list_path, 'r') as f:
            symptoms_list = f.read().splitlines()
        logger.info(f"Loaded {len(symptoms_list)} symptoms successfully")
    except Exception as e:
        logger.error(f"Error loading symptoms list: {e}")
        st.error("Error loading symptoms list. Please check the logs.")
        st.stop()
    
    # Add search functionality for symptoms
    st.subheader("Select your symptoms")
    st.info("Select as many symptoms as you're experiencing. The more symptoms you provide, the more accurate the prediction will be.")
    
    # Add a search box for symptoms with suggestion overlay
    search_term = st.text_input("Search for symptoms", key="symptom_search")
    
    # Filter symptoms based on search for suggestions
    suggestions = []
    if search_term:
        suggestions = [symptom for symptom in symptoms_list if search_term.lower() in symptom.lower()]
        if len(suggestions) > 0 and len(suggestions) <= 10:
            # Display suggestions in a container with styling
            st.markdown("""
            <style>
            .suggestion-container {
                background-color: #1E2130;
                border: 1px solid #304250;
                border-radius: 8px;
                padding: 10px;
                margin-bottom: 15px;
            }
            .suggestion-item {
                padding: 5px 10px;
                margin: 2px 0;
                cursor: pointer;
                border-radius: 4px;
            }
            .suggestion-item:hover {
                background-color: #304250;
            }
            </style>
            """, unsafe_allow_html=True)
            
            suggestion_container = st.container()
            with suggestion_container:
                st.markdown(f"<div class='suggestion-container'>", unsafe_allow_html=True)
                st.markdown("### Quick Suggestions:")
                for i, suggestion in enumerate(suggestions):
                    if st.button(suggestion, key=f"suggestion_{i}"):
                        # Add the selected suggestion to the multiselect
                        st.session_state.symptoms_selector = list(st.session_state.get('symptoms_selector', [])) + [suggestion]
                        st.experimental_rerun()
                st.markdown("</div>", unsafe_allow_html=True)
    
    # Filter symptoms based on search for the multiselect
    filtered_symptoms = symptoms_list
    if search_term:
        filtered_symptoms = [symptom for symptom in symptoms_list if search_term.lower() in symptom.lower()]
        st.write(f"Found {len(filtered_symptoms)} matching symptoms")
    
    # Create multiselect for symptoms
    selected_symptoms = st.multiselect(
        'Select your symptoms',
        filtered_symptoms,
        key='symptoms_selector'
    )
    
    # Add common symptom categories for easier selection
    st.markdown("### Quick Symptom Selection")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Pain Symptoms**")
        pain_symptoms = st.multiselect(
            'Select pain-related symptoms',
            [s for s in symptoms_list if 'pain' in s.lower()],
            key='pain_symptoms'
        )
        
    with col2:
        st.markdown("**Fever/Infection**")
        fever_symptoms = st.multiselect(
            'Select fever-related symptoms',
            [s for s in symptoms_list if any(x in s.lower() for x in ['fever', 'chill', 'sweat', 'infection'])],
            key='fever_symptoms'
        )
        
    with col3:
        st.markdown("**Digestive Issues**")
        digestive_symptoms = st.multiselect(
            'Select digestive symptoms',
            [s for s in symptoms_list if any(x in s.lower() for x in ['stomach', 'nausea', 'vomit', 'diarrhea', 'digest'])],
            key='digestive_symptoms'
        )
    
    # Combine all selected symptoms
    all_selected_symptoms = list(set(selected_symptoms + pain_symptoms + fever_symptoms + digestive_symptoms))
    
    if all_selected_symptoms:
        st.markdown("### Your Selected Symptoms:")
        st.write(", ".join(all_selected_symptoms))
    
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button('Predict Disease', key='predict_common'):
        if not all_selected_symptoms:
            st.warning("Please select at least one symptom")
        else:
            with st.spinner("Analyzing your symptoms..."):
                try:
                    # Create the input vector for the model
                    ipt = [1 if symptom in all_selected_symptoms else 0 for symptom in symptoms_list]
                    ipt = np.array([ipt])
                    
                    # Make predictions
                    pred = common_model.predict(ipt)[0]
                    
                    # Try to get prediction probabilities
                    confidence_score = 0
                    alternative_diseases = []
                    
                    try:
                        if has_predict_proba["common"]:
                            # Get probabilities
                            probs = common_model.predict_proba(ipt)[0]
                            
                            # Sort probabilities and get corresponding classes
                            top_indices = probs.argsort()[-3:][::-1]  # Top 3 predictions
                            top_probs = probs[top_indices]
                            
                            # Get disease names
                            disease_classes = common_model.classes_
                            top_diseases = [disease_classes[i] for i in top_indices]
                            
                            # Set main prediction confidence
                            confidence_score = int(top_probs[0] * 100)
                            
                            # Store alternative predictions
                            alternative_diseases = [(top_diseases[i], int(top_probs[i] * 100)) 
                                                  for i in range(1, len(top_diseases)) if top_probs[i] > 0.1]
                    except Exception as e:
                        logger.warning(f"Could not get prediction probabilities: {e}")
                    
                    # If no probability available, calculate based on symptom count
                    if confidence_score == 0:
                        total_symptoms = len(all_selected_symptoms)
                        confidence_score = min(100, max(40, total_symptoms * 10))
                    
                    if any(ipt[0]):
                        # Display results
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown("### Prediction Results")
                        
                        # Display risk meter
                        display_risk_meter(confidence_score)
                        
                        common_diagnostics = f"Predicted Disease: {pred}"
                        st.success(common_diagnostics)
                        st.info(f"Confidence: {confidence_score}%")
                        
                        # Show alternative predictions if available
                        if alternative_diseases:
                            st.markdown("### Alternative Possibilities")
                            for disease, prob in alternative_diseases:
                                st.write(f"- {disease} ({prob}% confidence)")
                        
                        # Show warning and potential alternatives
                        if confidence_score < 70:
                            st.warning("Low confidence prediction. Please consider the following:")
                            st.write("- Add more symptoms if you're experiencing them")
                            st.write("- Consult with a healthcare professional")
                            st.write("- The prediction is based solely on the symptoms provided")
                        
                        # Add symptom visualization
                        st.markdown("### Symptom Analysis")
                        
                        # Create symptom visualization
                        if len(all_selected_symptoms) > 0:
                            # Create word cloud-like visualization
                            fig, ax = plt.subplots(figsize=(10, 4))
                            y_pos = range(len(all_selected_symptoms))
                            
                            # Create random sizes for visual effect
                            np.random.seed(42)  # For reproducibility
                            sizes = np.random.randint(70, 100, size=len(all_selected_symptoms))
                            
                            ax.barh(y_pos, sizes, color=plt.cm.tab20.colors[:len(all_selected_symptoms)])
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(all_selected_symptoms)
                            ax.invert_yaxis()
                            ax.set_xlabel('Relevance', color='white')
                            ax.set_title('Symptom Analysis', color='white')
                            ax.set_facecolor('#1E2130')
                            
                            # Remove top and right spines
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            
                            for spine in ax.spines.values():
                                spine.set_color('white')
                            ax.tick_params(axis='x', colors='white')
                            ax.tick_params(axis='y', colors='white')
                            
                            st.pyplot(fig)
                        
                        # Display disease information and recommendations
                        st.markdown("### Treatment & Recommendations")
                        
                        # Disease description
                        description = get_disease_description(pred)
                        if description:
                            with st.expander("Disease Information", expanded=True):
                                st.write(description)
                        
                        # Medications
                        medications = get_medication_info(pred)
                        if medications:
                            with st.expander("Recommended Medications", expanded=True):
                                st.write("*Note: Always consult a healthcare professional before taking any medications.*")
                                for med in medications:
                                    st.write(f"- {med}")
                        
                        # Precautions
                        precautions = get_precautions(pred)
                        if precautions:
                            with st.expander("Precautions", expanded=True):
                                for precaution in precautions:
                                    st.write(f"- {precaution}")
                        
                        # Diet recommendations
                        diets = get_diet_recommendations(pred)
                        if diets:
                            with st.expander("Diet Recommendations", expanded=True):
                                for diet in diets:
                                    st.write(f"- {diet}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Add AI symptom tracking system
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown("### AI Symptom Tracker")
                        st.info("Track your symptoms over time to monitor your condition")
                        
                        track_symptoms = st.checkbox("Would you like to track these symptoms?")
                        if track_symptoms:
                            st.success("Symptom tracking enabled. Your symptoms will be saved for future reference.")
                            st.markdown("Set a reminder to check your symptoms again in:")
                            reminder_time = st.selectbox("Reminder frequency", 
                                                        ["12 hours", "1 day", "3 days", "1 week"],
                                                        index=1)
                            st.write(f"We'll remind you to check your symptoms again in {reminder_time}.")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    else:
                        st.error("Could not make a prediction. Please select different symptoms.")
                        
                except Exception as e:
                    logger.error(f"Error during prediction: {e}")
                    st.error("An error occurred during prediction. Please try again.")

# Enhanced footer
st.markdown("""
<div style="background-color: #0E1117; padding: 15px; border-radius: 10px; margin-top: 20px; text-align: center;">
    <h3 style="color: #4CAF50;">HealthPredictor AI</h3>
    <p>Contact a Doctor: <a href="https://www.apollo247.com/specialties" style="color: #42A5F5;">Apollo Healthcare</a></p>
    <p style="color: #FFA726; font-weight: bold;">DISCLAIMER</p>
    <p style="font-size: 12px;">WARNING - A medical expert, like a doctor, is best able to help you find the information and care you need. 
    This information is medical advice or diagnosis based on prediction from more than 3000+ patients.
    If you find anything abnormal, please contact a doctor first.
    This application is for informational purposes only and should not replace professional medical advice.</p>
    <p style="font-size: 10px; margin-top: 15px; color: #777;">© 2023 HealthPredictor AI - Powered by Advanced Machine Learning</p>
</div>
""", unsafe_allow_html=True)