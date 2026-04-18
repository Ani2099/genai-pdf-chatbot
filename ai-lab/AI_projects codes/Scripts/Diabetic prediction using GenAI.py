#!pip uninstall -y google-generativeai
#!pip install -q --upgrade google-generativeai

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import google.generativeai as genai



# --- Step 1: Dataset Preparation ---

df = pd.read_csv("diabetes.csv")

X = df[['Age', 'BMI', 'Glucose', 'BloodPressure']]
y = df['Outcome']

# --- Step 2: Train Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# --- Step 3: Predict from real patient input ---
def get_patient_input():
    print("\nEnter patient details:")
    age = float(input("Age: "))
    bmi = float(input("BMI: "))
    glucose = float(input("Glucose Level (mg/dL): "))
    bp = float(input("Blood Pressure (mmHg): "))
    return [[age, bmi, glucose, bp]]

patient_data = get_patient_input()
prediction = model.predict(patient_data)[0]

label_reverse = {0: 'Non-Diabetic', 1: 'Pre-Diabetic', 2: 'Diabetic'}
predicted_class = label_reverse[prediction]
print(f"\n🩺 Prediction: The patient is likely {predicted_class}.")


os.environ["GOOGLE_API_KEY"] = " "
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# --- Step 4: Use OpenAI to suggest diet ---
def get_diet_suggestion(condition):
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
    Suggest a healthy and practical daily diet plan for a person who is {condition}.
    Include:
    - Breakfast
    - Lunch
    - Dinner
    - Snacks
    Also list foods they should avoid.
    """
    response = model.generate_content(prompt)
    return response.text

diet_plan = get_diet_suggestion(predicted_class.lower())
print("\n🍽️ Suggested Diet Plan:\n")
print(diet_plan)

models = genai.list_models()
for model in models:
    print("Model Name:", model.name)
    print("Supported Methods:", model.supported_generation_methods)
    print("-" * 50)

!pip uninstall -y google-generativeai
!pip install -q --upgrade google-generativeai

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import google.generativeai as genai



# --- Step 1: Dataset Preparation ---

df = pd.read_csv("diabetes.csv")

X = df[['Age', 'BMI', 'Glucose', 'BloodPressure']]
y = df['Outcome']

# --- Step 2: Train Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# --- Step 3: Predict from real patient input ---
def get_patient_input():
    print("\nEnter patient details:")
    age = float(input("Age: "))
    bmi = float(input("BMI: "))
    glucose = float(input("Glucose Level (mg/dL): "))
    bp = float(input("Blood Pressure (mmHg): "))
    return [[age, bmi, glucose, bp]]

patient_data = get_patient_input()
prediction = model.predict(patient_data)[0]

label_reverse = {0: 'Non-Diabetic', 1: 'Pre-Diabetic', 2: 'Diabetic'}
predicted_class = label_reverse[prediction]
print(f"\n🩺 Prediction: The patient is likely {predicted_class}.")


os.environ["GOOGLE_API_KEY"] = " "
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# --- Step 4: Use OpenAI to suggest diet ---
def get_diet_suggestion(condition):
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
    Suggest a healthy and practical daily diet plan for a person who is {condition}.
    Include:
    - Breakfast
    - Lunch
    - Dinner
    - Snacks
    Also list foods they should avoid.
    """
    response = model.generate_content(prompt)
    return response.text

diet_plan = get_diet_suggestion(predicted_class.lower())
print("\n🍽️ Suggested Diet Plan:\n")
print(diet_plan)



!pip uninstall -y google-generativeai
!pip install -q --upgrade google-generativeai


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import google.generativeai as genai



# --- Step 1: Dataset Preparation ---

df = pd.read_csv("diabetes.csv")

X = df[['Age', 'BMI', 'Glucose', 'BloodPressure']]
y = df['Outcome']

# --- Step 2: Train Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# --- Step 3: Predict from real patient input ---
def get_patient_input():
    print("\nEnter patient details:")
    age = float(input("Age: "))
    bmi = float(input("BMI: "))
    glucose = float(input("Glucose Level (mg/dL): "))
    bp = float(input("Blood Pressure (mmHg): "))
    return [[age, bmi, glucose, bp]]

os.environ["GOOGLE_API_KEY"] = " "
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# --- Step 4: Use OpenAI to suggest diet ---
def get_diet_suggestion(condition):
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
    Suggest a healthy and practical daily diet plan for a person who is {condition}.
    Include:
    - Breakfast
    - Lunch
    - Dinner
    - Snacks
    Also list foods they should avoid.
    """
    response = model.generate_content(prompt)
    return response.text


while True:
    patient_data = get_patient_input()
    prediction = model.predict(patient_data)[0]

    label_reverse = {0: 'Non-Diabetic', 1: 'Pre-Diabetic', 2: 'Diabetic'}
    predicted_class = label_reverse[prediction]
    print(f"\n🩺 Prediction: The patient is likely {predicted_class}.")

    diet_plan = get_diet_suggestion(predicted_class.lower())
    print("\n🍽️ Suggested Diet Plan:\n")
    print(diet_plan)

    another_patient = input("\nDo you want to enter details for another patient? (yes/no): ").lower()
    if another_patient != 'yes':
        break
