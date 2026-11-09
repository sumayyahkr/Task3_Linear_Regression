# Task 3: Linear Regression

## Project Overview
This project demonstrates Simple and Multiple Linear Regression using Python and Scikit-learn.

The Diabetes dataset provided by Scikit-learn is used to predict a continuous disease progression measure.

## Tools and Libraries
- Python
- Pandas
- Scikit-learn
- Matplotlib

## Project Steps
1. Loaded and explored the Diabetes dataset.
2. Separated input features and target variable.
3. Split the data into training and testing sets (80:20).
4. Built a Simple Linear Regression model using BMI.
5. Built a Multiple Linear Regression model using all 10 features.
6. Evaluated both models using MAE, MSE and R².
7. Visualized the simple regression line.
8. Compared model performance using a bar chart.

## Model Results

| Metric | Simple Regression | Multiple Regression |
|---|---:|---:|
| MAE | 52.26 | 42.79 |
| MSE | 4061.83 | 2900.19 |
| R² | 0.2334 | 0.4526 |

## Key Findings
- Multiple Linear Regression performed better on the test set.
- The multiple regression model achieved an R² score of approximately 0.453.
- Using multiple features improved prediction compared with BMI alone.

## Project Files
- `linear_regression.py` – Python implementation.
- `simple_regression_plot.png` – Actual values and fitted regression line.
- `model_comparison.png` – Comparison of model R² scores.

## Conclusion
This project demonstrates how simple and multiple linear regression models can be implemented, evaluated and interpreted using Scikit-learn.