import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def save_model_and_encoders(model, disease_encoder, medicine_encoder):
    model_filename = 'medicine_prediction_model.sav'
    with open(model_filename, 'wb') as model_file:
        pickle.dump(model, model_file)

    encoder_filename = 'encoders.sav'
    with open(encoder_filename, 'wb') as encoder_file:
        pickle.dump({'disease_encoder': disease_encoder, 'medicine_encoder': medicine_encoder}, encoder_file)

    print(f'Model saved as {model_filename}')
    print(f'Encoders saved as {encoder_filename}')

def load_model_and_encoders():
    with open('medicine_prediction_model.sav', 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    with open('encoders.sav', 'rb') as encoder_file:
        encoders = pickle.load(encoder_file)
        disease_encoder = encoders['disease_encoder']
        medicine_encoder = encoders['medicine_encoder']

    return loaded_model, disease_encoder, medicine_encoder

def train_model():
    data = pd.read_csv("medicines\medical data.csv")
    data_col = data[['Disease', 'Medicine']].dropna()

    disease_encoder = LabelEncoder()
    medicine_encoder = LabelEncoder()

    data_col['disease_encoded'] = disease_encoder.fit_transform(data_col['Disease'])
    data_col['medicine_encoded'] = medicine_encoder.fit_transform(data_col['Medicine'])

    X = data_col[['disease_encoded']]
    Y = data_col['medicine_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    save_model_and_encoders(model, disease_encoder, medicine_encoder)

def predict_medicine(new_disease):
    model, disease_encoder, medicine_encoder = load_model_and_encoders()

    try:
        new_disease_encoded = disease_encoder.transform([new_disease])[0]
        predicted_medicine_encoded = model.predict([[new_disease_encoded]])
        predicted_medicine = medicine_encoder.inverse_transform(predicted_medicine_encoded)
        print(f'Predicted Medicine for {new_disease}: {predicted_medicine[0]}')
    except ValueError:
        print(f'Disease "{new_disease}" not found in the dataset.')

if __name__ == "__main__":
    train_model()
    new_disease = 'Common Cold'
    predict_medicine(new_disease)
