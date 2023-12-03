import numpy as np
from tqdm import tqdm
from Perfrom_Regression_and_Plot import perform_linear_regression_and_plot

coefficients = perform_linear_regression_and_plot()

y_intercept = coefficients[0]
beta_1 = coefficients[1]
beta_2 = coefficients[2]
beta_3 = coefficients[3]
beta_4 = coefficients[4]

predicted_gasflow_mean = float(input('Enter the Predicted Gas Flow Mean: '))
predicted_gasflow_std_dev = float(input('Enter the Standard Deviation of the Predicted Gas Flow: '))

predicted_GOR_mean = float(input('Enter the Predicted Gas-oil Ratio Mean: '))
predicted_GOR_std_dev = float(input('Enter the Standard Deviation of the Predicted Gas-Oil Ratio: '))

failure_criterion = float(input('Enter the failure criterion i.e generation capacity: '))

num_calculations = int(input('Enter the number of calculations to be performed: '))

# Lists to store results
predicted_gasflows = []
predicted_oilflows = []
predicted_duties = []

# Counter for failures
failures = 0

# Use tqdm to create a progress bar
for sim in tqdm(range(num_calculations), desc='Simulations', unit='sim'):
    predicted_GOR = np.random.normal(loc=predicted_GOR_mean, scale=predicted_GOR_std_dev)
    predicted_gasflow = np.random.normal(loc=predicted_gasflow_mean, scale=predicted_gasflow_std_dev)
    predicted_oilflow = (predicted_gasflow / predicted_GOR) * 10 ** 6

    # Use ** for exponentiation instead of ^
    predicted_duty = y_intercept + (predicted_gasflow * beta_1) + (predicted_oilflow * beta_2) + \
                     (predicted_gasflow ** 2 * beta_3) + (predicted_oilflow ** 2 * beta_4)

    # Append results to lists
    predicted_gasflows.append(predicted_gasflow)
    predicted_oilflows.append(predicted_oilflow)
    predicted_duties.append(predicted_duty)

    # Check for failure
    if predicted_duty > failure_criterion:
        failures += 1

# Calculate the probability of failure
probability_of_failure = (failures / num_calculations) * 100

print(f"Probability of Failure {probability_of_failure:.2f}%")
