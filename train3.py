import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd
from joblib import dump, load


# Data Loading
data = pd.read_csv('simulation_data/simsimulation_data_NMPC-RTI.csv')
X = data[['X1', 'X2', 'X3', 'U1', 'U2']]  # Features
y = data[['X_ref1','X_ref2','X_ref3','U_ref1','U_ref2']]  # Targets

# Data Preprocessing
scaler_X = RobustScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = RobustScaler()
y_scaled = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Model Design
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(y_train.shape[1])
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Training
model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2)

# Evaluation
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Saving the Model
model.save('cstr_model.keras')
dump(scaler_X, 'scaler_X.joblib') 
dump(scaler_y, 'scaler_y.joblib') 



loaded_model = tf.keras.models.load_model('cstr_model.h5')

# Prediction
new_X = np.array([[0.41137016775970414,296.1595673206998,0.3543881814968072,295.3865639162796,0.08500000000549597]])  # New input data
new_X_scaled = scaler_X.transform(new_X)
predicted_y_scaled = loaded_model.predict(new_X_scaled)
predicted_U = scaler_y.inverse_transform(predicted_y_scaled)

print("Predicted U:", predicted_U)
