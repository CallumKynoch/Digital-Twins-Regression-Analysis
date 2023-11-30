import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


def perform_regression_and_plot():
    # Specify the path to your Excel file
    excel_file_path = r'C:\Users\Callum Kynoch\Documents\01 University of Edinburgh\MEng Project - Digital Twins\04 ' \
                      r'Experimental Data\example_2.xlsx'

    # Read the Excel file into a DataFrame
    df = pd.read_excel(excel_file_path)

    # Extract columns from the DataFrame
    x1_values = df['X'].values
    x2_values = df['Y'].values
    y_values = df['Z'].values

    # Create higher-order terms (polynomial terms)
    x1_squared = x1_values ** 2
    x2_squared = x2_values ** 2
    x1_cubed = x1_values ** 3
    x2_cubed = x2_values ** 3

    # Add a column of ones for the intercept
    x = sm.add_constant(np.column_stack((x1_values, x2_values, x1_squared, x2_squared, x1_cubed, x2_cubed)))

    # Fit the model
    model = sm.OLS(y_values, x).fit()

    # Get the regression coefficients
    coefficients = model.params

    print("Intercept:", coefficients[0])
    print("Coefficients for independent variables:", coefficients[1:])

    # Plot the actual data points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x1_values, x2_values, y_values, label='Actual data')

    # Create a meshgrid for the surface representing the regression model
    x1_range = np.linspace(min(x1_values), max(x1_values), 100)
    x2_range = np.linspace(min(x2_values), max(x2_values), 100)
    x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
    x_mesh = sm.add_constant(np.column_stack((x1_mesh.flatten(), x2_mesh.flatten(),
                                              x1_mesh.flatten() ** 2, x2_mesh.flatten() ** 2,
                                              x1_mesh.flatten() ** 3, x2_mesh.flatten() ** 3)))
    y_pred_mesh = np.dot(x_mesh, coefficients)

    # Reshape the predictions to the shape of the meshgrid
    y_pred_mesh = y_pred_mesh.reshape(x1_mesh.shape)

    # Plot the regression surface
    ax.plot_surface(x1_mesh, x2_mesh, y_pred_mesh, alpha=0.5, color='red', label='Regression surface')

    # Add labels and legend
    ax.set_xlabel('Gas Flow')
    ax.set_ylabel('Oil Flow')
    ax.set_zlabel('Total Duty')

    # Create a legend with the scatter plot and a proxy artist
    legend_labels = ['Actual data', 'Regression surface']
    legend_proxy = [plt.Line2D([0], [0], linestyle='none', c=scatter.get_facecolor(), marker='o'),
                    plt.Rectangle((0, 0), 1, 1, fc='red', alpha=0.5)]

    ax.legend(legend_proxy, legend_labels)

    # Set the view to a specific angle
    # ax.view_init(elev=0, azim=00)

    # Show the 3D plot
    plt.show()

    return coefficients
