from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def train_lstm_model(X_train, y_train, epochs=100, batch_size=32):
    model = Sequential()

    # First LSTM layer with more units and dropout
    model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.3))  # Dropout to reduce overfitting
    model.add(BatchNormalization())  # Batch normalization to stabilize training

    # Second LSTM layer with more units
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.3))  # Dropout layer
    model.add(BatchNormalization())  # Batch normalization layer

    # Third LSTM layer
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.3))  # Dropout layer

    # Dense layer for final output
    model.add(Dense(1))

    # Optimizer with a lower learning rate
    optimizer = Adam(learning_rate=0.0005)  # Reduce learning rate for better convergence
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Early stopping to avoid overfitting
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[early_stop])

    return model
