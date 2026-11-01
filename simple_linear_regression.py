import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Load and clean data
df = pd.read_csv("student_info.csv")
df = df.dropna()
pd.set_option("display.max_rows", None)

# 2. Define features and target
X = df[['study_hours']]
y = df['student_marks']

# 3. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict for full dataset (for regression line)
y_line = model.predict(X)

# 6. Plot regression line with actual data
fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot()
ax.set_xlabel("Hours Studied")
ax.set_ylabel("Student Marks")
ax.scatter(X, y, color="blue", label="Actual")
plt.plot(X, y_line, color="red", label="Regression Line")
plt.legend()
plt.show()

# 7. Predict on test set
y_pred = model.predict(X_test)

# 8. Print evaluation metrics
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))