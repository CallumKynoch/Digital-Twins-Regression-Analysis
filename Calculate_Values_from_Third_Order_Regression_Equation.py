import numpy as np
from tqdm import tqdm
from Perfrom_Regression_and_Plot import perform_third_order_regression_and_plot
import matplotlib.pyplot as plt
import sys
from tkinter import filedialog as fd

coefficients = perform_third_order_regression_and_plot()

# β Coefficients from Regression
y_intercept = coefficients[0][0]
beta_1 = coefficients[0][1]
beta_2 = coefficients[0][2]
beta_3 = coefficients[0][3]
beta_4 = coefficients[0][4]
beta_5 = coefficients[0][5]
beta_6 = coefficients[0][6]
interaction_1 = coefficients[0][7]
interaction_2 = coefficients[0][8]
interaction_3 = coefficients[0][9]

# Initialise Normal Distributions
predicted_gasflow_mean = float(input('Enter the Predicted Gas Flow Mean in MMSCFD: '))
predicted_gasflow_std_dev = float(input('Enter the Standard Deviation of the Predicted Gas Flow in MMSCFD: '))

predicted_GOR_mean = float(input('Enter the Predicted Gas-oil Ratio Mean: '))
predicted_GOR_std_dev = float(input('Enter the Standard Deviation of the Predicted Gas-Oil Ratio: '))

capacity_mean = float(input('Enter the mean capacity per steam turbine in MW: '))
capacity_GOR = float(input('Enter the Standard Deviation of the capacity per steam turbine in MW: '))

# Enter the number of calculations to be performed
num_calculations = int(input('Enter the number of calculations to be performed: '))

# Lists to store results
predicted_gasflows = []
predicted_oilflows = []
predicted_duties = []
predicted_capacity = []

# Counter for failures
failures = 0

# Lists to store indices of failures exceeding total capacity
indices_of_failures_exceeding_capacity = []

# Use tqdm to create a progress bar
for sim in tqdm(range(num_calculations), desc='Simulations', unit='sim'):
    # Generate Well Flows
    predicted_GOR = np.random.normal(loc=predicted_GOR_mean, scale=predicted_GOR_std_dev)
    predicted_gasflow = np.random.normal(loc=predicted_gasflow_mean, scale=predicted_gasflow_std_dev)
    predicted_oilflow = (predicted_gasflow / predicted_GOR) * 10 ** 6

    # Generate Steam Turbine Duties
    random_capacities = np.random.normal(loc=capacity_mean, scale=capacity_GOR, size=4)
    total_capacity = np.sum(random_capacities)

    # Calculate the Interaction Term
    interaction_term_1 = predicted_gasflow * predicted_oilflow
    interaction_term_2 = (predicted_gasflow ** 2) * predicted_oilflow
    interaction_term_3 = predicted_gasflow * (predicted_gasflow ** 2)

    # Calculate the Duty
    predicted_duty = (
            y_intercept +
            (predicted_gasflow * beta_1) +
            (predicted_oilflow * beta_2) +
            (predicted_gasflow ** 2 * beta_3) +
            (predicted_oilflow ** 2 * beta_4) +
            (predicted_gasflow ** 3 * beta_5) +
            (predicted_oilflow ** 3 * beta_6) +
            (interaction_term_1 * interaction_1) +
            (interaction_term_2 * interaction_2) +
            (interaction_term_3 * interaction_3)
    )

    # Append results to lists
    predicted_gasflows.append(predicted_gasflow)
    predicted_oilflows.append(predicted_oilflow)
    predicted_duties.append(predicted_duty)

    # Apply the Failure Criterion
    failure_criterion = total_capacity - predicted_duty

    # Check the Failure Criterion
    if failure_criterion <= 0:
        failures += 1

# Calculate the Probability of Failure
probability_of_failure = (failures / num_calculations) * 100

print(f"Probability of Failure {probability_of_failure:.2f}%")

create_outcrossing_graph = input('Would you like to create an outcrossing graph? Y or N: ')

if create_outcrossing_graph.upper() == 'Y':
    # Plot the predicted duty vs. number of simulations
    plt.plot(range(1, num_calculations + 1), predicted_duties, label='Predicted Duty')

    # Plot the total capacity line
    total_capacity_line = predicted_capacity
    plt.plot(range(1, num_calculations + 1), total_capacity_line, label='Total Capacity', linestyle='--')

    # Highlight failures exceeding total capacity with red dots
    plt.scatter(indices_of_failures_exceeding_capacity,
                [predicted_duties[i] for i in indices_of_failures_exceeding_capacity],
                color='red', marker='o', label='Exceed Total Capacity')

    # Set labels and title
    plt.xlabel('Number of Simulations')
    plt.ylabel('Predicted Duty (MW)')

    # Add major gridlines
    plt.grid(True, linestyle='-', alpha=0.7)

    # Add legend
    plt.legend()

    pth = fd.asksaveasfilename(
        title='Choose a location to save the graph',
        initialdir=(r'C:\Users\callk\Documents\02 University of Edinburgh\01 MEng Project\00 Digital '
                    r'Twins\Graphs\Outcrossing Graphs'),
        filetypes=[('Picture', '.png')]
    )
    plt.savefig(pth, dpi=1200)

    # Show the plot
    plt.show()

else:
    sys.exit()
