# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes

# Load the dataset
diabetes = load_diabetes()

# Convert dataset into a DataFrame
df = pd.DataFrame(
    diabetes.data,
    columns=diabetes.feature_names
)

# Add target column
df["target"] = diabetes.target

# Display dataset
print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nStatistical summary:")
print(df.describe())
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Separate input features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)
# SIMPLE LINEAR REGRESSION

# Select only BMI as input
X_simple = df[["bmi"]]

# Split the data
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

# Create and train the model
simple_model = LinearRegression()
simple_model.fit(X_train_s, y_train_s)

# Make predictions
simple_predictions = simple_model.predict(X_test_s)

# Evaluate the model
print("\nSIMPLE LINEAR REGRESSION")
print("MAE:", mean_absolute_error(y_test_s, simple_predictions))
print("MSE:", mean_squared_error(y_test_s, simple_predictions))
print("R2 Score:", r2_score(y_test_s, simple_predictions))

# Display coefficient
print("BMI Coefficient:", simple_model.coef_[0])
print("Intercept:", simple_model.intercept_)
# MULTIPLE LINEAR REGRESSION

# Create and train the model
multiple_model = LinearRegression()
multiple_model.fit(X_train, y_train)

# Make predictions
multiple_predictions = multiple_model.predict(X_test)

# Evaluate the model
print("\nMULTIPLE LINEAR REGRESSION")
print("MAE:", mean_absolute_error(y_test, multiple_predictions))
print("MSE:", mean_squared_error(y_test, multiple_predictions))
print("R2 Score:", r2_score(y_test, multiple_predictions))

# Display coefficients
print("\nModel Coefficients:")
for feature, coefficient in zip(X.columns, multiple_model.coef_):
    print(feature, ":", coefficient)

print("Intercept:", multiple_model.intercept_)
# VISUALIZATION: SIMPLE LINEAR REGRESSION

plt.figure(figsize=(8, 5))

plt.scatter(X_test_s["bmi"], y_test_s, label="Actual values")

# Sort BMI values so the regression line displays correctly
sorted_indices = X_test_s["bmi"].argsort()

plt.plot(
    X_test_s["bmi"].iloc[sorted_indices],
    simple_predictions[sorted_indices],
    color="red",
    label="Regression line"
)

plt.xlabel("BMI")
plt.ylabel("Disease Progression")
plt.title("Simple Linear Regression: BMI vs Target")
plt.legend()

plt.savefig("simple_regression_plot.png")
plt.show()
# COMPARE MODEL PERFORMANCE

model_names = ["Simple Regression", "Multiple Regression"]

r2_scores = [
    r2_score(y_test_s, simple_predictions),
    r2_score(y_test, multiple_predictions)
]

plt.figure(figsize=(8, 5))
plt.bar(model_names, r2_scores)

plt.ylabel("R2 Score")
plt.title("Comparison of Regression Models")
plt.ylim(0, 1)

plt.savefig("model_comparison.png")
plt.show()