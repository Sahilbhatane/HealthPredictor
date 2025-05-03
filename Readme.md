# Disease Prediction using AI

A comprehensive AI-powered health prediction application that can predict multiple diseases based on patient symptoms and clinical data.

## Features

- **Multiple Disease Prediction**:
  - Diabetes Prediction
  - Heart Disease Prediction
  - Parkinson's Disease Prediction 
  - Common Diseases Prediction (40+ diseases)

- **Advanced Analysis**:
  - Interactive data visualization
  - Risk factor analysis
  - Symptoms correlation
  - Treatment recommendations

- **User-Friendly Interface**:
  - Patient mode with simple symptom selection
  - Doctor mode with detailed clinical parameters
  - Medication and precaution recommendations

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application by executing the `start.bat` file or using:
   ```
   streamlit run "multiple disease pred.py"
   ```
2. The web interface will open in your default browser
3. Select the disease prediction module you want to use
4. Enter the required information and get your prediction

## Directory Structure

- `saved_models/` - Contains trained ML models
- `medicines/` - Contains medication and treatment data
- `common-diseases-Prediction-model-main/` - Contains common disease model

## Model Information

- **Diabetes Model**: Logistic Regression model (97% accuracy)
- **Heart Disease Model**: Decision Tree model (95% accuracy)
- **Parkinson's Model**: Support Vector Machine model (96% accuracy)
- **Common Disease Model**: Decision Tree Classifier (97% accuracy)

## Requirements

- Python 3.7+
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## License

This project is available for personal and educational use.
