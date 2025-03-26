import pandas as pd
import numpy as np

# Sample DataFrame
data = {
    'A': [1, 2, np.nan, np.nan, 5, 6],
    'B': [2, 3, np.nan, 5, 6, 7],
    'C': [3, np.nan, 5, 6, 7, 8],
    'D': [4, 5, 6, np.nan, 8, 9],
    'E': [5, 6, 7, 8, np.nan, np.nan]
}
df = pd.DataFrame(data)

# Function to interpolate missing values in a column using 2nd-degree polynomial
def interpolate_column(column):
    # Extract non-missing values
    non_missing_indices = column.notna()
    x = np.arange(len(column))[non_missing_indices]
    y = column[non_missing_indices]

    # Fit a 2nd-degree polynomial
    coefficients = np.polyfit(x, y, 2)
    poly = np.poly1d(coefficients)
    
    # Predict missing values
    missing_indices = np.arange(len(column))[~non_missing_indices]
    predicted_values = poly(missing_indices)
    
    # Fill the missing values in the column
    column[missing_indices] = predicted_values
    return column

# Apply the function to each column in the DataFrame
df = df.apply(interpolate_column, axis=0)

print(df)