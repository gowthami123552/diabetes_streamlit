
# Diabetes Prediction Using Machine Learning

## Project Overview
The Diabetes Prediction System is a Machine Learning project developed to predict diabetes categories based on health indicators and lifestyle information.

The system performs three-class classification:
- 0 - No Diabetes
- 1 - Prediabetes
- 2 - Diabetes

The project uses multiple classification algorithms and selects the best-performing model based on evaluation metrics. The final model is deployed through an interactive Streamlit application.

## Dataset Information
- Dataset: CDC Diabetes Health Indicators (BRFSS 2015)
- Total Records: 253,680
- Total Features: 21
- Target Variable: Diabetes_012
- Problem Type: Multiclass Classification

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

## Machine Learning Workflow

1. Data Collection
2. Data Exploration and Analysis
3. Missing Value Checking
4. Duplicate Detection
5. Outlier Detection
6. Data Preprocessing
7. Feature Scaling
8. Data Transformation
9. Handling Class Imbalance
10. Feature Selection using SelectKBest
11. Feature Extraction using PCA
12. Train-Test Splitting
13. Model Training
14. Model Evaluation
15. Model Selection
16. Model Saving
17. Streamlit Deployment

## Algorithms Used

The following classification algorithms are implemented:

- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Linear Support Vector Machine (SVM)

The best-performing model is selected based on Macro F1 Score.

## Model Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Balanced Accuracy
- Confusion Matrix
- Classification Report

## Project Structure

Diabetes_Prediction/
│
├── models/
│   ├── diabetes_prediction_pipeline.pkl
│   └── model_comparison.csv
│
├── app.py
├── requirements.txt
└── README.md

## Installation and Setup

### 1. Clone the Repository

git clone https://github.com/your-username/diabetes-prediction.git

### 2. Navigate to the Project Directory

cd diabetes-prediction

### 3. Install Required Libraries

pip install -r requirements.txt

### 4. Run the Streamlit Application

streamlit run app.py

## Application Features

- Interactive health information input form
- Three-class diabetes prediction
- Automated data preprocessing
- Saved Machine Learning pipeline
- User-friendly Streamlit interface
- Real-time prediction results

## Model Deployment

The trained model pipeline is saved using Joblib in the models directory.

The Streamlit application loads the saved pipeline and performs predictions on user-provided health information.

## Future Enhancements

- Improve prediction performance
- Add more health indicators
- Enhance application interface
- Integrate additional model evaluation visualizations

## Disclaimer

This project is developed for educational purposes only. The predictions are not a substitute for professional medical diagnosis or treatment.

## Author

Developed as a Machine Learning and Streamlit deployment project.
