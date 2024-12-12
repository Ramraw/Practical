import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define the universe of discourse for temperature and humidity
temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'temperature')  # 0 to 40 degrees Celsius
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')        # 0 to 100 percent

# Define the output variable for AC power
ac_power = ctrl.Consequent(np.arange(0, 101, 1), 'ac_power')        # 0 to 100 percent operation

# Membership functions for temperature
temperature['cold'] = fuzz.trimf(temperature.universe, [0, 0, 20])
temperature['comfortable'] = fuzz.trimf(temperature.universe, [15, 25, 35])
temperature['hot'] = fuzz.trimf(temperature.universe, [30, 40, 40])

# Membership functions for humidity
humidity['dry'] = fuzz.trimf(humidity.universe, [0, 0, 50])
humidity['comfortable'] = fuzz.trimf(humidity.universe, [30, 50, 70])
humidity['humid'] = fuzz.trimf(humidity.universe, [60, 100, 100])

# Membership functions for AC power
ac_power['off'] = fuzz.trimf(ac_power.universe, [0, 0, 25])
ac_power['low'] = fuzz.trimf(ac_power.universe, [20, 40, 60])
ac_power['medium'] = fuzz.trimf(ac_power.universe, [50, 70, 90])
ac_power['high'] = fuzz.trimf(ac_power.universe, [80, 100, 100])

# Define fuzzy rules
rule1 = ctrl.Rule(temperature['cold'] & humidity['dry'], ac_power['off'])
rule2 = ctrl.Rule(temperature['cold'] & humidity['comfortable'], ac_power['low'])
rule3 = ctrl.Rule(temperature['cold'] & humidity['humid'], ac_power['medium'])
rule4 = ctrl.Rule(temperature['comfortable'] & humidity['dry'], ac_power['low'])
rule5 = ctrl.Rule(temperature['comfortable'] & humidity['comfortable'], ac_power['medium'])
rule6 = ctrl.Rule(temperature['comfortable'] & humidity['humid'], ac_power['medium'])
rule7 = ctrl.Rule(temperature['hot'] & humidity['humid'], ac_power['high'])
rule8 = ctrl.Rule(temperature['hot'] & humidity['comfortable'], ac_power['high'])
rule9 = ctrl.Rule(temperature['hot'] & humidity['dry'], ac_power['medium'])

# Create the control system
ac_power_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
ac_power_simulation = ctrl.ControlSystemSimulation(ac_power_control)

# Input values for testing
ac_power_simulation.input['temperature'] = 35  # Comfortable temperature
ac_power_simulation.input['humidity'] = 70     # Comfortable humidity

# Compute the output
ac_power_simulation.compute()

# Print the output
print("AC Power Level: ", ac_power_simulation.output['ac_power'])