import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Lifestyle ML App", layout="centered")
st.title("🎓 Student Lifestyle & Performance Classifier")

# Load pre-uploaded dataset
file_path = "student_lifestyle_dataset..csv"  # Ensure this file is in the same directory

try:
    df = pd.read_csv(file_path)

    # Preprocessing
    def grade_to_category(grade):
        if grade >= 8.0:
            return "High"
        elif grade >= 6.0:
            return "Medium"
        else:
            return "Low"

    df["Grade_Category"] = df["Grades"].apply(grade_to_category)

    st.subheader("📊 Class Distribution")
    st.dataframe(df["Grade_Category"].value_counts(normalize=True).rename("Proportion").to_frame())

    label_encoders = {}
    for col in ["Stress_Level", "Gender", "Grade_Category"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop(columns=["Student_ID", "Grades", "Grade_Category"])
    y = df["Grade_Category"]

    # Scale features ...
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

    # Use class_weight balanced
    rf = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=3, class_weight='balanced', random_state=42)
    ab = AdaBoostClassifier(n_estimators=100, learning_rate=0.8, random_state=42)

    # Voting classifier
    voting_clf = VotingClassifier(estimators=[('rf', rf), ('ab', ab)], voting='soft')
    voting_clf.fit(X_train, y_train)
    preds = voting_clf.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    st.success(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

    st.subheader("📋 Classification Report")
    report = classification_report(y_test, preds, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("🔍 Confusion Matrix")
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoders["Grade_Category"].classes_, yticklabels=label_encoders["Grade_Category"].classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    st.subheader("📈 Feature Importance (Random Forest)")
    rf.fit(X_train, y_train)
    feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.bar_chart(feat_imp)

    st.subheader("💡 Predict Student Grade Category")
    study = st.slider("Study Hours Per Day", 0.0, 12.0, 6.0)
    extra = st.slider("Extracurricular Hours Per Day", 0.0, 6.0, 2.0)
    sleep = st.slider("Sleep Hours Per Day", 0.0, 12.0, 7.0)
    social = st.slider("Social Hours Per Day", 0.0, 6.0, 2.0)
    physical = st.slider("Physical Activity Hours Per Day", 0.0, 6.0, 1.0)
    stress = st.selectbox("Stress Level", label_encoders["Stress_Level"].classes_)
    gender = st.selectbox("Gender", label_encoders["Gender"].classes_)

    stress_enc = label_encoders["Stress_Level"].transform([stress])[0]
    gender_enc = label_encoders["Gender"].transform([gender])[0]

    user_input = scaler.transform([[study, extra, sleep, social, physical, stress_enc, gender_enc]])

    if st.button("Predict Grade Category"):
        prediction = voting_clf.predict(user_input)[0]
        pred_label = label_encoders["Grade_Category"].inverse_transform([prediction])[0]
        st.info(f"🧠 Predicted Grade Category: {pred_label}")

except FileNotFoundError:
    st.error("The dataset file 'student_lifestyle_dataset..csv' was not found. Please make sure it exists in the app directory.")
