# test_import.py
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.data_loader import get_historical_data
from utils.feature_engineering import add_technical_indicators, add_lagged_features
from utils.model_training import prepare_lstm_data, train_lstm_model
from utils.scaling import scale_data, inverse_scale, scale_new_data
from utils.evaluation import evaluate_model, plot_predictions


# Step 1: Load the historical data
ticker = input("Enter the stock ticker (e.g., NVDA): ")
period = input("Enter the historical period (e.g., '1y', '6mo', '3mo'): ")
data = get_historical_data(ticker, period)

# Step 2: Add technical indicators and lagged features
data = add_technical_indicators(data)
data = add_lagged_features(data)

# Step 3: Scale the data
scaled_data, scaler = scale_data(data['Close'])

# Step 4: Prepare the data for LSTM
X_train, y_train, _ = prepare_lstm_data(data)

# Step 5: Train the LSTM model
model = train_lstm_model(X_train, y_train, epochs=50, batch_size=32)

# Step 6: Make predictions on the test set
predicted_prices_scaled = model.predict(X_train)
predicted_prices = inverse_scale(predicted_prices_scaled, scaler)

# Step 7: Evaluate the model and plot predictions
evaluate_model(data['Close'].values[-len(predicted_prices):], predicted_prices.flatten())
plot_predictions(data[-len(predicted_prices):], data['Close'].values[-len(predicted_prices):], predicted_prices.flatten())
