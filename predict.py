import joblib

# Load saved objects
rf_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- USER INPUT PART ---------------- #

print("\nEnter patient details:")

pregnancies = float(input("Pregnancies: "))
glucose = float(input("Glucose level: "))
bp = float(input("Blood Pressure: "))
skin = float(input("Skin Thickness: "))
insulin = float(input("Insulin: "))
bmi = float(input("BMI: "))
dpf = float(input("Diabetes Pedigree Function: "))
age = float(input("Age: "))

# Create input array
user_data = [[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]]

# Scale user input
user_data_scaled = scaler.transform(user_data)

# Prediction
prediction = rf_model.predict(user_data_scaled)

print("\nChoosen model: Random Forest Classifier")

# Output
if prediction[0] == 1:
    print("\nPrediction: Person is likely DIABETIC")
else:
    print("\nPrediction: Person is likely NOT DIABETIC")