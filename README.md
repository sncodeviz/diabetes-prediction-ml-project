# Diabetes Prediction Project
12.12.2025

Sheikh Abu Nasher<br/>
Suffolk University<br/>
ISOM 835 Predictive Analytics and Machine Learning<br/>
Professor Hasan Arslan

## Overview
This healthcare analytics-themed project analyzes a real-world clinical dataset to develop a machine learning model that predicts the likelihood of an individual having diabetes based on basic health and demographic features. 
Using a structured predictive analytics workflow, the project examines factors such as age, BMI, blood glucose level, and Hemoglobin A1 that contribute to diabetes risk. The project demonstrates the full lifecycle of applied data science, from raw data to actionable insights, while also highlighting the importance of model interpretability, ethical considerations, and responsible machine learning use in health-related analytics.

## Dataset
Diabetes Prediction Dataset on [Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset).

## Code Notebook
Project Code Notebook detailing Phases 1-6 from Google Collab on [GitHub](https://github.com/sncodeviz/diabetes-prediction-ml-project/blob/main/Project_Google_Collab_Notebook.ipynb).

## Streamlit App
This project includes an interactive Streamlit app for real-time diabetes risk prediction. [Try it here](https://diabetes-prediction-ml-project-jpaj2crdcspxpc2tvbvcwu.streamlit.app/)

_**Note**: Due to version and library limitations on Streamlit Cloud (Python 3.13), the deployed app uses a custom SMOTE-like oversampling method instead of the original imbalanced-learn implementation used during offline model development in Colab._

## Data Flow Diagram
This diagram represents the end-to-end workflow of the diabetes prediction project, including data cleaning, exploratory analysis, preprocessing, model training, evaluation, and deployment. [View the diagram on GitHub](https://github.com/sncodeviz/diabetes-prediction-ml-project/blob/main/assets/diagrams/End-to-End%20ML%20Pipeline%20(Data%20Flow%20Diagram).png)

## Executive Summary
Diabetes is a growing and important public health concern not only in the United States of America but also worldwide. In healthcare analytics, early detection of diabetes is an effective way to reduce long-term health complications and healthcare costs, especially with Americans already struggling with their daily costs and high health insurance premiums. 

Many Americans and individuals globally who are at-risk of diabetes remain undiagnosed until mid or severe symptoms appear, which delays treatment. The purpose of this project is to develop a predictive model that can help identify individuals who may be at higher risk of diabetes based on basic clinical and demographic data. 

The dataset used for this project was a large, real-world diabetes prediction dataset containing over 100,000 patient records obtained from Kaggle. Each record included features such as age, gender, BMI, smoking history, blood glucose level, HbA1c level, hypertension, and heart disease status. The target variable indicated whether a patient had been diagnosed with diabetes. The dataset was highly imbalanced, with only about 8% of patients having diabetes, which required special handling with SMOTE during the modeling phase.

The project consisted of six phases, which included exploratory initial checks, data analysis, data cleaning and preprocessing, class imbalance handling, model development, and evaluation. Key preprocessing steps included imputing missing values, encoding categorical features, scaling numerical variables, and stratifying the train/test split. 
During modeling, three different models were evaluated:
- Logistic Regression (baseline model)
- Decision Tree
- Random Forest

_(each combined with SMOTE to balance the minority class)_

Performance was compared using precision, recall, F1-score, and overall accuracy. The results showed that tree-based models outperformed the linear model, particularly in identifying diabetic patients. **The Decision Tree Model** achieved the best balance between accuracy and recall for the minority class, making it the most effective option for this dataset. 

Feature importance analysis of the Decision Tree Model confirmed that blood glucose level and HbA1c level were the strongest predictors, which aligns with clinical standards for diagnosing diabetes. Additional contributors included age and BMI.

Based on these findings, healthcare providers could use a model like this as a supportive risk assessment tool. It can help flag patients who may require additional screening, support early intervention efforts, and allocate clinical resources more efficiently. 

However, most importantly, the model should not be used as a standalone diagnostic system replacing medically trained professionals. Instead, it should support medical decision-making while ensuring that patient privacy, fairness, and ethical guidelines are maintained.

## **Full Report on [Github](https://github.com/sncodeviz/diabetes-prediction-ml-project/blob/main/Project%20Report.pdf).**



