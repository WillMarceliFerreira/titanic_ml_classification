# Titanic Dataset Analysis and Modeling Project

## Overview
This project aims to analyze and model the Titanic dataset, a classic dataset in machine learning. It uses Streamlit to create an interactive web application for exploring different stages of machine learning, from data understanding to prediction.

## Requirements
To run this project, you need:

- Python
- Streamlit
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- joblib
- pickle

## Installation
Clone the repository and install the required libraries:

```bash
pip install -r requirements.txt
```
## Usage
Run the Streamlit application:

´´´bash
streamlit run app.py
´´´
## Features
- Interactive sidebar to navigate through different steps of the analysis and modeling process.
- Visualization of the Titanic dataset including univariate and bivariate analysis.
- Data preprocessing steps including feature extraction, engineering, and scaling.
- Training of multiple machine learning models with hyperparameter tuning.
- Evaluation of models using various metrics like ROC-AUC and Log Loss.
- A prediction feature where users can input passenger data to predict survival.

## File Descriptions
- **app.py**: Main Streamlit application script.
- **page_contents.py**: Contains functions that define the content for each page of the Streamlit app.
- **visualization.py**: Includes the EasyVisualize class for creating various types of plots.
- **scaler.pkl**: Pre-trained scaler model for data preparation.
- **model.pkl**: Pre-trained machine learning model for making predictions.

## Dataset
The Titanic dataset used in this project is sourced from Kaggle. It includes passenger details such as age, sex, fare, class, and survival status.