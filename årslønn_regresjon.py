import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Data provided
data = {
    'Year': np.array([2022, 2021, 2020, 2019, 2018, 2017]),
    'Number': np.array([347.0109431, 346.1777037, 337.475827, 333.6591867, 327.8507638, 308.6742405])
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Linear regression requires a 2D array for the independent variable
# Reshape year data to 2D array
X = df['Year'].values.reshape(-1, 1)
y = df['Number'].values

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Predict the number for the year 1999
prediction_for_1999 = model.predict(np.array([[1999]]))

# Generate a sequence of years for predictions
years = np.arange(1999, 2023).reshape(-1, 1)
predicted_numbers = model.predict(years)

# Plot the actual data
plt.scatter(df['Year'], df['Number'], color='black', label='% (forhold)')

# Plot the regression line
plt.plot(years, predicted_numbers, color='blue', label='Regresjonslinje')

# Mark the prediction for 1999
plt.scatter(1999, prediction_for_1999, color='red', label='Prediksjon for 1999')

# Add title and labels
#plt.title('Linear Regression on Data')
plt.title('Forhold mellom gjeld og brutto inntekt (gjennomsnittlig)')
plt.xlabel('År')
plt.ylabel('%')

# Show legend
plt.legend()

# Display the plot
plt.show()

# Output the prediction for 1999
prediction_for_1999[0]
