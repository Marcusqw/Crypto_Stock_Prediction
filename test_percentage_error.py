import numpy as np

# Given values
stock_price = 689286
mse = 62715685

# Calculate RMSE
rmse = np.sqrt(mse)

# Calculate error percentage
error_percentage = (rmse / stock_price) * 100

print(f"RMSE: {rmse}")
print(f"Error Percentage: {error_percentage}%")