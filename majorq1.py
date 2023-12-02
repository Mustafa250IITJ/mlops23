from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import numpy as np

digits = load_digits()
data = digits.data

scaler = StandardScaler()

normalized_data = scaler.fit_transform(data)

# Print the original data and the normalized data for comparison
print("Original Data:")
print(data)
print("\nNormalized Data:")
print(normalized_data)
