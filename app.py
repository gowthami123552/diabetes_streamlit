
import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

# --------------------------------------------------
# 2. LOAD SAVED MODEL
# --------------------------------------------------

MODEL_PATH = "models/diabetes_prediction_pipeline.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)


bundle = load_model()

# --------------------------------------------------
# 3. APPLICATION HEADER
# --------------------------------------------------

st.title("🩺 Diabetes Prediction System")

st.write(
    "Enter the health indicators below to predict "
    "the diabetes category using a Machine Learning model."
)

st.divider()

if bundle is None:
    st.error(
        "Model file not found! Please check whether "
        "models/diabetes_prediction_pipeline.pkl exists."
    )
    st.stop()

model = bundle["pipeline"]
feature_names = bundle["feature_names"]
class_labels = bundle["class_labels"]

# --------------------------------------------------
# 4. INPUT FORM
# --------------------------------------------------

st.subheader("Personal Health Information")

st.info(
    "Please provide the information accurately. "
    "All fields are required."
)

with st.form("diabetes_prediction_form"):

    col1, col2 = st.columns(2)

    input_data = {}

    # ----------------------------------------------
    # COLUMN 1
    # ----------------------------------------------

    with col1:

        st.markdown("### General Information")

        input_data["HighBP"] = 1 if st.selectbox(
            "High Blood Pressure",
            ["No", "Yes"]
        ) == "Yes" else 0

        input_data["HighChol"] = 1 if st.selectbox(
            "High Cholesterol",
            ["No", "Yes"]
        ) == "Yes" else 0

        input_data["CholCheck"] = 1 if st.selectbox(
            "Cholesterol Check (within 5 years)",
            ["No", "Yes"]
        ) == "Yes" else 0

        input_data["BMI"] = st.number_input(
            "Body Mass Index (BMI)",
            min_value=12.0,
            max_value=98.0,
            value=25.0,
            step=0.1
        )

        input_data["Smoker"] = 1 if st.selectbox(
            "Have you smoked at least 100 cigarettes in your life?",
            ["No", "Yes"]
        ) == "Yes" else 0

        input_data["Stroke"] = 1 if st.selectbox(
            "History of Stroke",
            ["No", "Yes"]
        ) == "Yes" else 0

        input_data["HeartDiseaseorAttack"] = 1 if st.selectbox(
            "Coronary Heart Disease or Heart Attack",
            ["No", "Yes"]
        ) == "Yes" else 0

        input_data["PhysActivity"] = 1 if st.selectbox(
            "Physical Activity (outside work)",
            ["No", "Yes"]
        ) == "Yes" else 0

        input_data["Fruits"] = 1 if st.selectbox(
            "Consume Fruits One or More Times Per Day",
            ["No", "Yes"]
        ) == "Yes" else 0

        input_data["Veggies"] = 1 if st.selectbox(
            "Consume Vegetables One or More Times Per Day",
            ["No", "Yes"]
        ) == "Yes" else 0

        input_data["HvyAlcoholConsump"] = 1 if st.selectbox(
            "Heavy Alcohol Consumption",
            ["No", "Yes"]
        ) == "Yes" else 0

    # ----------------------------------------------
    # COLUMN 2
    # ----------------------------------------------

    with col2:

        st.markdown("### Health and Lifestyle Information")

        input_data["AnyHealthcare"] = 1 if st.selectbox(
            "Do You Have Healthcare Coverage?",
            ["No", "Yes"]
        ) == "Yes" else 0

        input_data["NoDocbcCost"] = 1 if st.selectbox(
            "Could Not Visit Doctor Due to Cost",
            ["No", "Yes"]
        ) == "Yes" else 0

        input_data["GenHlth"] = st.selectbox(
            "General Health",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "Excellent",
                2: "Very Good",
                3: "Good",
                4: "Fair",
                5: "Poor"
            }[x]
        )

        input_data["MentHlth"] = st.slider(
            "Number of Days Mental Health Was Not Good (last 30 days)",
            min_value=0,
            max_value=30,
            value=0
        )

        input_data["PhysHlth"] = st.slider(
            "Number of Days Physical Health Was Not Good (last 30 days)",
            min_value=0,
            max_value=30,
            value=0
        )

        input_data["DiffWalk"] = 1 if st.selectbox(
            "Difficulty Walking or Climbing Stairs",
            ["No", "Yes"]
        ) == "Yes" else 0

        input_data["Sex"] = st.selectbox(
            "Sex",
            options=[0, 1],
            format_func=lambda x: {
                0: "Female",
                1: "Male"
            }[x]
        )

        input_data["Age"] = st.selectbox(
            "Age Category",
            options=list(range(1, 14)),
            format_func=lambda x: {
                1: "18-24",
                2: "25-29",
                3: "30-34",
                4: "35-39",
                5: "40-44",
                6: "45-49",
                7: "50-54",
                8: "55-59",
                9: "60-64",
                10: "65-69",
                11: "70-74",
                12: "75-79",
                13: "80 or older"
            }[x]
        )

        input_data["Education"] = st.selectbox(
            "Education Level",
            options=list(range(1, 7)),
            format_func=lambda x: {
                1: "Never attended / Kindergarten only",
                2: "Grades 1-8",
                3: "Grades 9-11",
                4: "Grade 12 / GED",
                5: "College 1-3 years",
                6: "College 4 or more years"
            }[x]
        )

        input_data["Income"] = st.selectbox(
            "Annual Household Income",
            options=list(range(1, 9)),
            format_func=lambda x: {
                1: "Less than $10,000",
                2: "$10,000-$14,999",
                3: "$15,000-$19,999",
                4: "$20,000-$24,999",
                5: "$25,000-$34,999",
                6: "$35,000-$49,999",
                7: "$50,000-$74,999",
                8: "$75,000 or more"
            }[x]
        )

    st.divider()

    # ----------------------------------------------
    # SUBMIT BUTTON
    # ----------------------------------------------

    submitted = st.form_submit_button(
        "Predict Diabetes",
        use_container_width=True
    )

# --------------------------------------------------
# 5. MAKE PREDICTION
# --------------------------------------------------

if submitted:

    try:

        # Create input DataFrame in training column order
        input_df = pd.DataFrame(
            [input_data],
            columns=feature_names
        )

        # Predict using complete saved pipeline
        prediction = model.predict(input_df)

        predicted_class = int(prediction[0])

        result = class_labels[predicted_class]

        st.divider()

        st.subheader("Prediction Result")

        if predicted_class == 0:

            st.success(
                "Prediction: No Diabetes"
            )

        elif predicted_class == 1:

            st.warning(
                "Prediction: Prediabetes"
            )

        elif predicted_class == 2:

            st.error(
                "Prediction: Diabetes"
            )

        st.write("**Predicted Category:**", result)

        st.caption(
            "This result is generated by a machine learning "
            "model and is not a medical diagnosis. "
            "Please consult a qualified healthcare professional "
            "for medical advice."
        )

    except Exception as e:

        st.error("Prediction failed!")

        st.exception(e)

# --------------------------------------------------
# 6. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Diabetes Prediction System | "
    "Machine Learning Project | "
    "Scikit-learn and Streamlit"
)
