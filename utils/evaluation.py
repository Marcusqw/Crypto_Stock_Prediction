import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def evaluate_model(true_prices, predicted_prices):
    mse = mean_squared_error(true_prices, predicted_prices)
    print(f"Mean Squared Error: {mse}")

def plot_predictions(dates, true_prices, predicted_prices):
    plt.figure(figsize=(10, 6))
    plt.plot(dates.index, true_prices, label="Actual Prices")
    plt.plot(dates.index, predicted_prices, label="Predicted Prices", linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()
