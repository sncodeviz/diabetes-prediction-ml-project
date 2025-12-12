# Diabetes Prediction Project
12.12.2025

Sheikh Abu Nasher<br/>
Suffolk University<br/>
ISOM 835 Predictive Analytics and Machine Learning<br/>
Professor Hasan Arslan

## Project Webpage
View the full interactive project webpage is available on [GitHub Pages](https://sncodeviz.github.io/diabetes-prediction-ml-project/). 

## Overview
This project analyzes a large medical dataset to develop a predictive model that predicts the likelihood of an individual having diabetes. Using a predictive analytics workflow, the project examines factors such as age, BMI, blood glucose level, and Hemoglobin A1 that contribute to diabetes risk. The project also highlights the importance of model interpretability, ethical considerations, and responsible machine learning use in health-related analytics.

## Dataset
Diabetes Prediction Dataset on [Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset).

## Code Notebook
Project Code Notebook detailing Phases 1-6 from Google Collab on [GitHub](https://github.com/sncodeviz/diabetes-prediction-ml-project/blob/main/Project_Google_Collab_Notebook.ipynb).

## Streamlit App
This project includes an interactive Streamlit app for real-time diabetes risk prediction. [Try it here](https://diabetes-prediction-ml-project-jpaj2crdcspxpc2tvbvcwu.streamlit.app/)

_**Note**: Due to version and library limitations on Streamlit Cloud (Python 3.13), the deployed app uses a custom SMOTE-like oversampling method instead of the original imbalanced-learn implementation used during offline model development in Collab._

## Data Flow Diagram
This diagram represents the end-to-end workflow of the diabetes prediction project, including data cleaning, exploratory analysis, preprocessing, model training, evaluation, and deployment. [View the diagram on GitHub](https://github.com/sncodeviz/diabetes-prediction-ml-project/blob/main/assets/diagrams/End-to-End%20ML%20Pipeline%20(Data%20Flow%20Diagram).png)

## Executive Summary
Diabetes is a growing and important public health concern not only in the United States of America but also worldwide. In healthcare analytics, early detection of diabetes is an effective way to reduce long-term health complications and healthcare costs, especially with Americans already struggling with their daily costs and high health insurance premiums. 

Many Americans and individuals globally who are at-risk of diabetes go undiagnosed until mid or severe symptoms appear. The purpose of this project is to develop a predictive model that can help identify individuals who may be at higher risk of diabetes based on commonly gathered medical data. 

The dataset used for this project was a diabetes prediction dataset containing over 100,000 patient records obtained from Kaggle. Each record included features such as age, gender, BMI, smoking history, blood glucose level, HbA1c level, hypertension, and heart disease status. The target variable indicated whether a patient had been diagnosed with diabetes. The dataset was highly imbalanced, with only about 8% of patients having diabetes, which required special handling with SMOTE during the modeling phase.

The project consisted of six phases, which included exploratory initial checks, data analysis, data cleaning and preprocessing, class imbalance handling, model development, and evaluation. Key preprocessing steps included imputing missing values, encoding categorical features, scaling numerical variables, and stratifying the training and testing splits. 

During modeling, three different models were evaluated:

- Logistic Regression (baseline model)
- Decision Tree
- Random Forest

_(each combined with SMOTE to balance the minority class)_

**The Decision Tree Model** was chosen to be the final model due to balanced scores on accuracy, precision, and recall compared to Logistic Regression and Random Forest. A feature analysis on the final model revealed that blood glucose and HbA1c levels were the strongest predictors of diabetes, which is already known in medical research. Additionally, patient BMI and age contributed on a small scale. 

The final model of this project would be a huge supportive tool for healthcare professionals to identify patients who are most likely to have diabetes in the near future and need more support and additional medical tests. However, this model should not be a standalone diabetes diagnostic tool replacing medical professionals and standard testing. It should be a supportive tool while making sure patient privacy and ethical guidelines are followed. 

## **Full Report on [Github](https://github.com/sncodeviz/diabetes-prediction-ml-project/blob/main/Project_Report.pdf).**



