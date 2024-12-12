import numpy as np
from sklearn.preprocessing import MinMaxScaler
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generate synthetic data
np.random.seed(42)
size = np.random.uniform(500, 4000, 100)  # House size in square feet
location_quality = np.random.uniform(1, 10, 100)  # Quality rating (1-10)
rooms = np.random.randint(1, 10, 100)  # Number of rooms
price = 50000 + (size * 30) + (location_quality * 10000) + (rooms * 5000) + np.random.normal(0, 25000, 100)

# Normalize the data (important for neural networks)
scaler = MinMaxScaler()
data = np.column_stack((size, location_quality, rooms, price))
data = scaler.fit_transform(data)
X, y = data[:, :-1], data[:, -1]

# Define fuzzy variables
size_fuzzy = ctrl.Antecedent(np.arange(0, 1, 0.01), 'size')
location_fuzzy = ctrl.Antecedent(np.arange(0, 1, 0.01), 'location_quality')
rooms_fuzzy = ctrl.Antecedent(np.arange(0, 1, 0.01), 'rooms')
price_fuzzy = ctrl.Consequent(np.arange(0, 1, 0.01), 'price')

# Membership functions
size_fuzzy['small'] = fuzz.trimf(size_fuzzy.universe, [0, 0, 0.5])
size_fuzzy['medium'] = fuzz.trimf(size_fuzzy.universe, [0, 0.5, 1])
size_fuzzy['large'] = fuzz.trimf(size_fuzzy.universe, [0.5, 1, 1])

location_fuzzy['poor'] = fuzz.trimf(location_fuzzy.universe, [0, 0, 0.5])
location_fuzzy['average'] = fuzz.trimf(location_fuzzy.universe, [0, 0.5, 1])
location_fuzzy['good'] = fuzz.trimf(location_fuzzy.universe, [0.5, 1, 1])

rooms_fuzzy['few'] = fuzz.trimf(rooms_fuzzy.universe, [0, 0, 0.5])
rooms_fuzzy['moderate'] = fuzz.trimf(rooms_fuzzy.universe, [0, 0.5, 1])
rooms_fuzzy['many'] = fuzz.trimf(rooms_fuzzy.universe, [0.5, 1, 1])

price_fuzzy['low'] = fuzz.trimf(price_fuzzy.universe, [0, 0, 0.5])
price_fuzzy['medium'] = fuzz.trimf(price_fuzzy.universe, [0, 0.5, 1])
price_fuzzy['high'] = fuzz.trimf(price_fuzzy.universe, [0.5, 1, 1])

# Define rules
rule1 = ctrl.Rule(size_fuzzy['small'] & location_fuzzy['poor'] & rooms_fuzzy['few'], price_fuzzy['low'])
rule2 = ctrl.Rule(size_fuzzy['large'] & location_fuzzy['good'] & rooms_fuzzy['many'], price_fuzzy['high'])

# Create a control system
price_ctrl = ctrl.ControlSystem([rule1, rule2])
price_simulation = ctrl.ControlSystemSimulation(price_ctrl)

# Create a neural network model
model = Sequential([
    Dense(10, activation='relu', input_shape=(X.shape[1],)),
    Dense(10, activation='relu'),
    Dense(1)  # Output layer
])
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=100, batch_size=10, verbose=1)

# Define a function for hybrid inference
def hybrid_inference(input_data):
    # Neural Network Prediction
    nn_output = model.predict(input_data)
    
    # Set up Fuzzy Inference inputs
    price_simulation.input['size'] = input_data[0, 0]
    price_simulation.input['location_quality'] = input_data[0, 1]
    price_simulation.input['rooms'] = input_data[0, 2]
    
    # Perform fuzzy computation
    try:
        price_simulation.compute()
        # Ensure 'price' exists in output
        if 'price' in price_simulation.output:
            fuzzy_output = price_simulation.output['price']
        else:
            print("Fuzzy output 'price' not found. Check rule definitions and input values.")
            fuzzy_output = nn_output[0][0]  # Fallback to NN output if fuzzy output is missing
    except Exception as e:
        print("Error in fuzzy computation:", e)
        fuzzy_output = nn_output[0][0]  # Fallback in case of error
    
    # Combine fuzzy and neural network output
    hybrid_output = (nn_output[0][0] + fuzzy_output) / 2
    return hybrid_output

# Test the model
test_data = X[:5]  # Example test data
for sample in test_data:
    prediction = hybrid_inference(sample.reshape(1, -1))
    print("Hybrid Model Prediction:", prediction)