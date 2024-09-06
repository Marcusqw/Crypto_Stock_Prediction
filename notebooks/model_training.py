import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def prepare_lstm_data(data, sequence_length=60):
    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Prepare data for LSTM
    X_train, y_train = [], []
    for i in range(sequence_length, len(scaled_data)):
        X_train.append(scaled_data[i-sequence_length:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    return X_train, y_train, scaler

def train_lstm_model(X_train, y_train, epochs=50, batch_size=32):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))  # Dropout to prevent overfitting
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=0.001)  # Using a custom learning rate for better convergence
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[early_stop])

    return model

def predict_next_day(model, data, scaler, sequence_length=60):
    # Get the last sequence_length days for the prediction
    last_sequence = data['Close'].values[-sequence_length:]
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))

    X_input = np.array([last_sequence_scaled])
    X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))

    predicted_scaled_price = model.predict(X_input)
    predicted_price = scaler.inverse_transform(predicted_scaled_price)

    return predicted_price[0][0]
