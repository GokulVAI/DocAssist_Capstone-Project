import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Function to generate synthetic data
def generate_data(num_records=1000):
    np.random.seed(0)
    data = pd.DataFrame({
        'Age': np.random.randint(20, 70, num_records),
        'BMI': np.random.normal(28, 8, num_records),
        'Glucose': np.random.normal(100, 20, num_records),
        'BloodPressure': np.random.normal(120, 15, num_records),
        'Cholesterol': np.random.normal(200, 30, num_records),
        'Response': np.random.choice([0, 1], num_records)  # 0 for No Disease, 1 for Disease
    })
    return data

# Preprocess data function
def preprocess_data(features):
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return features

# Load and preprocess initial data
data = generate_data()
features = data.drop('Response', axis=1)
labels = data['Response']
features = preprocess_data(features)

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prepare for Streamlit app
scaler = StandardScaler().fit(features)  # Refit scaler on all data to use for incoming single data points

def main():
    st.title('DocAssist: Medical Decision Support System')

    with st.form("patient_input"):
        st.write("Enter patient data:")
        age = st.number_input("Age", min_value=20, max_value=100, value=50)
        bmi = st.number_input("BMI", value=28.0)
        glucose = st.number_input("Glucose", value=100.0)
        bp = st.number_input("Blood Pressure", value=120.0)
        cholesterol = st.number_input("Cholesterol", value=200.0)
        submitted = st.form_submit_button("Submit")

        if submitted:
            new_data = pd.DataFrame([{
                'Age': age, 'BMI': bmi, 'Glucose': glucose,
                'BloodPressure': bp, 'Cholesterol': cholesterol
            }])
            new_data = scaler.transform(new_data)  # Use the same scaler to transform new data
            prediction = model.predict(new_data)[0]
            probability = model.predict_proba(new_data)[0][prediction]

            st.write(f"### Treatment Recommendation: {'Disease' if prediction == 1 else 'No Disease'}")
            st.write(f"##### Probability of Prediction: {probability*100:.2f}%")

if __name__ == "__main__":
    main()
