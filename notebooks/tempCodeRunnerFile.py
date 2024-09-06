import pandas as pd
from utils.data_loader import get_historical_data
from utils.feature_engineering import add_technical_indicators, add_lagged_features
from utils.model_training import prepare_lstm_data, train_lstm_model
from utils.evaluation import evaluate_model, plot_predictions

# Step 1: Load the historical data
ticker = input("Enter the stock ticker (e.g., NVDA): ")
period = input("Enter the historical period (e.g., '1y', '6mo', '3mo'): ")
data = get_historical_data(ticker, period)

# Step 2: Add technical indicators and lagged features
data = add_technical_indicators(data)
data = add_lagged_features(data)

# Step 3: Prepare the data for LSTM
X_train, y_train, scaler = prepare_lstm_data(data)

# Step 4: Train the LSTM model
model = train_lstm_model(X_train, y_train, epochs=50, batch_size=32)

# Step 5: Make predictions on the test set
test_data = data[-len(X_train):]  # Taking a small part of data for prediction
scaled_test_data = scaler.transform(test_data['Close'].values.reshape(-1, 1))

predicted_prices = model.predict(X_train)  # Predict on training data for simplicity
predicted_prices = scaler.inverse_transform(predicted_prices)  # Inverse scaling

# Step 6: Evaluate the model and plot predictions
evaluate_model(data['Close'].values[-len(predicted_prices):], predicted_prices.flatten())
plot_predictions(data[-len(predicted_prices):], data['Close'].values[-len(predicted_prices):], predicted_prices.flatten())
