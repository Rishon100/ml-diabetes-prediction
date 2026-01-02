# 🩺 Diabetes Prediction using Machine Learning

This project is an **end-to-end Machine Learning system** that predicts whether a person is likely to have diabetes based on medical input features.

The goal of this project was to **relearn Machine Learning from scratch** and understand how models are trained, evaluated, saved, and reused in real-world scenarios.

---

## 🚀 Project Overview

* Built a complete ML pipeline from data preprocessing to prediction
* Used **Logistic Regression** as a baseline model
* Improved performance using a **Random Forest Classifier**
* Implemented **model persistence** by saving and loading trained models
* Enabled real-time prediction using user input from the terminal

---

## 🧠 Machine Learning Workflow

1. Load and inspect the dataset
2. Preprocess data (feature scaling using StandardScaler)
3. Split data into training and testing sets
4. Train baseline model (Logistic Regression)
5. Train and tune Random Forest model
6. Compare model performance using accuracy
7. Save the final trained model and scaler
8. Load saved model for prediction without retraining

---

## 📊 Models Used

### 1️⃣ Logistic Regression (Baseline Model)

* Used as a simple baseline for comparison
* Helps understand initial model performance
* Logistic Regression Accuracy: ~75%

### 2️⃣ Random Forest Classifier (Final Model)

* Ensemble model using multiple decision trees
* Tuned using `n_estimators` and `max_depth`
* Selected as the final model based on better accuracy
* Random Forest Accuracy: ~76%

---

## 📁 Project Structure

```
ml-diabetes-prediction/
├── train_model.py        # Trains models and saves final model
├── predict.py            # Loads saved model and predicts using user input
├── diabetes.csv          # Dataset
├── random_forest_model.pkl  # Saved Random Forest model
├── scaler.pkl            # Saved StandardScaler
├── README.md
```

---

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* VS Code

---

## ▶️ How to Run the Project

### 1️⃣ Clone the repository

```bash
git clone <your-github-repo-link>
cd ml-diabetes-prediction
```

### 2️⃣ Install dependencies

```bash
pip install pandas numpy scikit-learn joblib
```

### 3️⃣ Train and save the model (run once)

```bash
python train_model.py
```

### 4️⃣ Run prediction using saved model

```bash
python predict.py
```

Enter patient details when prompted to get a prediction.

---

## 🧪 Sample Input

```
Pregnancies: 2
Glucose: 150
Blood Pressure: 80
Skin Thickness: 30
Insulin: 100
BMI: 32
Diabetes Pedigree Function: 0.5
Age: 45
```

---

## ✅ Sample Output

```
Prediction: Person is likely DIABETIC
```

---

## 🎯 Key Learnings

* Difference between training and inference in ML
* Importance of feature scaling
* Model comparison and selection
* Saving and loading ML models using `joblib`
* Real-world ML project structure

---

## 👤 Author

** V Rishon Anand**
Machine Learning Student | AIML
Learning ML from scratch with a focus on strong fundamentals

---

## 🏁 Final Notes

This project represents my effort to **bridge the gap between theory and practical Machine Learning** by building a real, usable ML system step by step.

---
