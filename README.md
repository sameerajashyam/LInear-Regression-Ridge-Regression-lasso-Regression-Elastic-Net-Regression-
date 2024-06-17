# LInear-Regression-Ridge-Regression-lasso-Regression-Elastic-Net-Regression-


# Algerian Forest Fires Prediction Project
This project aims to predict the Fire Weather Index (FWI) using various regression models on a cleaned dataset of Algerian forest fires. The dataset includes various meteorological and fire-related features. The project involves data preprocessing, feature selection, feature scaling, model training, and evaluation using different regression techniques.

# Dataset
The dataset used for this project is the Algerian Forest Fires Cleaned Dataset. It includes the following features:

Temp: Temperature
RH: Relative Humidity
Ws: Wind Speed
Rain: Rainfall
FFMC: Fine Fuel Moisture Code
DMC: Duff Moisture Code
DC: Drought Code
ISI: Initial Spread Index
BUI: Buildup Index
Classes: Binary class indicating fire occurrence (encoded as 0 for 'not fire' and 1 for 'fire')
The target variable is FWI (Fire Weather Index).

# Project Steps
# 1. Data Loading and Initial Exploration
Load the dataset using pandas.
Inspect the first and last few rows of the dataset.
Drop irrelevant columns (day, month, year).
# 2. Data Preprocessing
Encode the Classes column to binary values (0 for 'not fire', 1 for 'fire').
Split the dataset into independent variables (x) and dependent variable (y).
# 3. Train-Test Split
Split the dataset into training and testing sets using train_test_split from sklearn.
# 4. Feature Selection and Multicollinearity Check
Check for multicollinearity among the features using a correlation matrix and heatmap.
Drop features with correlation coefficients greater than 0.85 to avoid multicollinearity.
# 5. Feature Scaling
Scale the features using StandardScaler from sklearn.
Visualize the effect of scaling using box plots.
# 6. Model Training and Evaluation
Train and evaluate multiple regression models:
Linear Regression
Lasso Regression
Ridge Regression 
ElasticNet Regression
For each model, calculate and print the Mean Absolute Error (MAE) and R-squared (R²) score.

Use LassoCV and ElasticNetCV for cross-validation.

# Results
Linear Regression
Mean Absolute Error:
R² Score:
Lasso Regression
Mean Absolute Error:
R² Score:
Ridge Regression
Mean Absolute Error:
R² Score:
ElasticNet Regression
Mean Absolute Error:
R² Score:
Conclusion
This project demonstrates the process of predicting the Fire Weather Index (FWI) using regression models. It includes data preprocessing steps like encoding, feature selection, and scaling, followed by training and evaluating various regression models. The project highlights the importance of handling multicollinearity and the effectiveness of different regression techniques in predicting the FWI.

# How to Run the Code
Clone the repository.
Ensure you have the necessary libraries installed (pandas, numpy, matplotlib, seaborn, sklearn).
Load the dataset.
Follow the steps in the Jupyter Notebook or Python script.
# Dependencies
pandas
numpy
matplotlib
seaborn
scikit-learn
bash

 pip install pandas numpy matplotlib seaborn scikit-learn
# Acknowledgements
The dataset was sourced from the UCI Machine Learning Repository. Special thanks to the contributors for making it available for research and educational purposes.

This README file provides an overview of the project, including the dataset, steps taken, and results obtained. It also includes instructions on how to run the code and a list of dependencies needed to replicate the project.
