import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tkinter import filedialog as fd
from tabulate import tabulate
import os


def perform_first_order_regression_and_plot():
    # Specify the path to your Excel file
    try:
        initial_dir = (r'C:\Users\callk\Documents\02 University of Edinburgh\01 MEng Project\00 Digital Twins'
                       r'\Simulation Data')

        if os.path.exists(initial_dir):
            excel_file_path = fd.askopenfilename(
                title='Choose dataset to perform regression on',
                initialdir=initial_dir
            )
        else:
            raise FileNotFoundError(f"Directory does not exist: {initial_dir}")
    except Exception as e:
        print(f"Error: {e}. Unable to locate the specified directory. Opening file explorer...")
        excel_file_path = fd.askopenfilename(
            title='Choose dataset to perform regression on',
            initialdir='/'
        )

    # Read the Excel file into a DataFrame
    df = pd.read_excel(excel_file_path)

    # Extract columns from the DataFrame
    x1_values = df['X'].values
    x2_values = df['Y'].values
    y_values = df['Z'].values
    x1x2_interaction = x1_values * x2_values

    # Add a column of ones for the intercept and include the interaction term
    x = sm.add_constant(np.column_stack((x1_values, x2_values, x1x2_interaction)))

    # Fit the model
    model = sm.OLS(y_values, x).fit()

    # Get the regression coefficients
    coefficients = model.params

    # Make predictions using the model
    y_pred = model.predict(x)

    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_values - y_pred) / y_values)) * 100

    # Calculate R-squared (R^2)
    r_squared = r2_score(y_values, y_pred)

    table_data = [
        ["Intercept", f"{coefficients[0]:.2e}"],
        ["MAPE", f"{mape:.2f}"],
        ["R-squared (R\u00b2)", f"{r_squared:.2f}"]
    ]

    # Add coefficients to the table
    for i, coef in enumerate(coefficients[1:], start=1):
        table_data.append([f"Coefficient {i}", f"{coef:.2e}"])

    # Print the table
    print(tabulate(table_data, headers=["Parameter", "Value"], tablefmt="grid"))

    # Plot the data points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x1_values, x2_values, y_values, label='Actual data')

    # Create a meshgrid for the surface representing the regression model
    x1_range = np.linspace(min(x1_values), max(x1_values), 100)
    x2_range = np.linspace(min(x2_values), max(x2_values), 100)
    x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)

    # Create the interaction term for the meshgrid
    interaction_mesh = x1_mesh.flatten() * x2_mesh.flatten()

    x_mesh = sm.add_constant(np.column_stack((x1_mesh.flatten(), x2_mesh.flatten(), interaction_mesh)))
    y_pred_mesh = np.dot(x_mesh, coefficients)

    # Reshape the predictions to the shape of the meshgrid
    y_pred_mesh = y_pred_mesh.reshape(x1_mesh.shape)

    # Plot the regression surface
    ax.plot_surface(x1_mesh, x2_mesh, y_pred_mesh, alpha=0.5, color='red', label='Response Surface')

    # Add labels and legend
    ax.zaxis.set_rotate_label(False)
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.labelpad = 20
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 35
    ax.set_xlabel('Gas Flow\n(MMSCFD)')
    ax.set_ylabel('Oil Flow\n(x10$^{3}$ bbl/d)\n\n\n\n\n\n\n\n\n', linespacing=1)
    ax.set_zlabel('Total Duty\n(MW)', rotation=0)
    y_max_graph = 80000
    y_tick_size = 20000

    y_ticks = [20000 * numnum for numnum in range(int((y_max_graph / y_tick_size) + 1))]
    ax.set_ylim(0, y_max_graph)  # Oil Flow
    ax.set_xlim(0, 45)  # Gas Flow
    ax.set_yticks(y_ticks)

    # Create a legend with the scatter plot and a proxy artist
    legend_labels = ['Simulation Data', 'Response Surface']
    legend_proxy = [plt.Line2D([0], [0], linestyle='none', c=scatter.get_facecolor(), marker='o'),
                    plt.Rectangle((0, 0), 1, 1, fc='red', alpha=0.5)]

    ax.legend(legend_proxy, legend_labels)
    textstr = '\n'.join((
        r'     R$^2=%.3f$' % r_squared,
        r'MAPE$=%.2f$%%' % mape
    ))
    props = dict(boxstyle='round', facecolor='white', edgecolor='lightgray', alpha=1)

    ax.text(x=0, y=80, z=160, s=textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    # Adjust y-axis labels
    oil_labels = ax.get_yticklabels()
    for label in oil_labels:
        cur_val = float(label.get_text())
        fixed_val = cur_val / 1000
        label.set_text(f'{int(fixed_val)}')  # Format to one decimal place

    ax.set_yticklabels(oil_labels,
                       rotation=0,
                       verticalalignment='baseline',
                       horizontalalignment='left'
                       )

    # Set the view to a specific angle
    # ax.view_init(elev=0, azim=90)

    pth = fd.asksaveasfilename(
        title='Choose dir to save',
        initialdir=(r'C:\Users\callk\Documents\02 University of Edinburgh\01 MEng Project\00 Digital '
                    r'Twins\Graphs\Response Surfaces'),
        filetypes=[('Picture', '.png')]
    )
    plt.savefig(pth, dpi=1200)

    # Show the 3D plot
    plt.show()

    return coefficients, mape, r_squared


def perform_second_order_regression_and_plot():
    # Specify the path to your Excel file
    try:
        initial_dir = (r'C:\Users\callk\Documents\02 University of Edinburgh\01 MEng Project\00 Digital Twins'
                       r'\Simulation Data')

        if os.path.exists(initial_dir):
            excel_file_path = fd.askopenfilename(
                title='Choose dataset to perform regression on',
                initialdir=initial_dir
            )
        else:
            raise FileNotFoundError(f"Directory does not exist: {initial_dir}")
    except Exception as e:
        print(f"Error: {e}. Unable to locate the specified directory. Opening file explorer...")
        excel_file_path = fd.askopenfilename(
            title='Choose dataset to perform regression on',
            initialdir='/'
        )

    # Read the Excel file into a DataFrame
    df = pd.read_excel(excel_file_path)

    # Extract columns from the DataFrame
    x1_values = df['X'].values
    x2_values = df['Y'].values
    y_values = df['Z'].values

    # Create higher-order terms (polynomial terms)
    x1_squared = x1_values ** 2
    x2_squared = x2_values ** 2
    x1x2_interaction = x1_values * x2_values

    # Add a column of ones for the intercept
    x = sm.add_constant(np.column_stack((x1_values, x2_values, x1_squared, x2_squared, x1x2_interaction)))

    # Fit the model
    model = sm.OLS(y_values, x).fit()

    # Get the regression coefficients
    coefficients = model.params

    # Make predictions using the model
    y_pred = model.predict(x)

    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_values - y_pred) / y_values)) * 100

    # Calculate R-squared (R^2)
    r_squared = r2_score(y_values, y_pred)

    table_data = [
        ["Intercept", f"{coefficients[0]:.2e}"],
        ["MAPE", f"{mape:.2f}"],
        ["R-squared (R\u00b2)", f"{r_squared:.2f}"]
    ]

    # Add coefficients to the table
    for i, coef in enumerate(coefficients[1:], start=1):
        table_data.append([f"Coefficient {i}", f"{coef:.2e}"])

    # Print the table
    print(tabulate(table_data, headers=["Parameter", "Value"], tablefmt="grid"))

    # Plot the actual data points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x1_values, x2_values, y_values, label='Actual data')

    # Create a meshgrid for the surface representing the regression model
    x1_range = np.linspace(min(x1_values), max(x1_values), 100)
    x2_range = np.linspace(min(x2_values), max(x2_values), 100)
    x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)

    # Create interaction term for the meshgrid
    x1x2_interaction_mesh = x1_mesh.flatten() * x2_mesh.flatten()

    x_mesh = sm.add_constant(np.column_stack((x1_mesh.flatten(), x2_mesh.flatten(),
                                              x1_mesh.flatten() ** 2, x2_mesh.flatten() ** 2,
                                              x1x2_interaction_mesh)))
    y_pred_mesh = np.dot(x_mesh, coefficients)

    # Reshape the predictions to the shape of the meshgrid
    y_pred_mesh = y_pred_mesh.reshape(x1_mesh.shape)

    # Plot the regression surface
    ax.plot_surface(x1_mesh, x2_mesh, y_pred_mesh, alpha=0.5, color='red', label='Regression surface')

    # Add labels and legend
    ax.zaxis.set_rotate_label(False)
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.labelpad = 20
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 35
    ax.set_xlabel('Gas Flow\n(MMSCFD)')
    ax.set_ylabel('Oil Flow\n(x10$^{3}$ bbl/d)\n\n\n\n\n\n\n\n\n', linespacing=1)
    ax.set_zlabel('Total Duty\n(MW)', rotation=0)
    y_max_graph = 80000
    y_tick_size = 20000

    y_ticks = [20000 * numnum for numnum in range(int((y_max_graph / y_tick_size) + 1))]
    ax.set_ylim(0, y_max_graph)  # Oil Flow
    ax.set_xlim(0, 45)  # Gas Flow
    ax.set_yticks(y_ticks)  # Add labels and legend
    ax.zaxis.set_rotate_label(False)
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.labelpad = 20
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 35
    ax.set_xlabel('Gas Flow\n(MMSCFD)')
    ax.set_ylabel('Oil Flow\n(x10$^{3}$ bbl/d)\n\n\n\n\n\n\n\n\n', linespacing=1)
    ax.set_zlabel('Total Duty\n(MW)', rotation=0)
    y_max_graph = 80000
    y_tick_size = 20000

    y_ticks = [20000 * numnum for numnum in range(int((y_max_graph / y_tick_size) + 1))]
    ax.set_ylim(0, y_max_graph)  # Oil Flow
    ax.set_xlim(0, 45)  # Gas Flow
    ax.set_yticks(y_ticks)

    # Create a legend with the scatter plot and a proxy artist
    legend_labels = ['Simulation Data', 'Response Surface']
    legend_proxy = [plt.Line2D([0], [0], linestyle='none', c=scatter.get_facecolor(), marker='o'),
                    plt.Rectangle((0, 0), 1, 1, fc='red', alpha=0.5)]

    ax.legend(legend_proxy, legend_labels)

    textstr = '\n'.join((
        r'     R$^2=%.3f$' % r_squared,
        r'MAPE$=%.2f$%%' % mape
    ))
    props = dict(boxstyle='round', facecolor='white', edgecolor='lightgray', alpha=1)

    ax.text(x=0, y=80, z=160, s=textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    # Adjust y-axis labels
    oil_labels = ax.get_yticklabels()
    for label in oil_labels:
        cur_val = float(label.get_text())
        fixed_val = cur_val / 1000
        label.set_text(f'{int(fixed_val)}')  # Format to one decimal place

    ax.set_yticklabels(oil_labels,
                       rotation=0,
                       verticalalignment='baseline',
                       horizontalalignment='left'
                       )

    # Set the view to a specific angle
    # ax.view_init(elev=30, azim=30)

    pth = fd.asksaveasfilename(
        title='Choose dir to save',
        initialdir=(r'C:\Users\callk\Documents\02 University of Edinburgh\01 MEng Project\00 Digital '
                    r'Twins\Graphs\Response Surfaces'),
        filetypes=[('Picture', '.png')]
    )
    plt.savefig(pth, dpi=1200)

    # Show the 3D plot
    plt.show()

    return coefficients, mape, r_squared


def perform_third_order_regression_and_plot():
    # Specify the path to your Excel file
    try:
        initial_dir = (r'C:\Users\callk\Documents\02 University of Edinburgh\01 MEng Project\00 Digital Twins'
                       r'\Simulation Data')

        if os.path.exists(initial_dir):
            excel_file_path = fd.askopenfilename(
                title='Choose dataset to perform regression on',
                initialdir=initial_dir
            )
        else:
            raise FileNotFoundError(f"Directory does not exist: {initial_dir}")
    except Exception as e:
        print(f"Error: {e}. Unable to locate the specified directory. Opening file explorer...")
        excel_file_path = fd.askopenfilename(
            title='Choose dataset to perform regression on',
            initialdir='/'
        )

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

    # Interaction terms
    x1_x2 = x1_values * x2_values
    x1_squared_x2 = x1_values ** 2 * x2_values
    x1_x2_squared = x1_values * x2_values ** 2

    # Add a column of ones for the intercept
    x = sm.add_constant(np.column_stack((x1_values, x2_values, x1_squared, x2_squared, x1_cubed, x2_cubed,
                                         x1_x2, x1_squared_x2, x1_x2_squared)))

    # Fit the model
    model = sm.OLS(y_values, x).fit()

    # Get the regression coefficients
    coefficients = model.params

    # Make predictions using the model
    y_pred = model.predict(x)

    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_values - y_pred) / y_values)) * 100

    # Calculate R-squared (R^2)
    r_squared = r2_score(y_values, y_pred)

    table_data = [
        ["Intercept", f"{coefficients[0]:.2e}"],
        ["MAPE", f"{mape:.2f}"],
        ["R-squared (R\u00b2)", f"{r_squared:.2f}"]
    ]

    # Add coefficients to the table
    for i, coef in enumerate(coefficients[1:], start=1):
        table_data.append([f"Coefficient {i}", f"{coef:.2e}"])

    # Print the table
    print(tabulate(table_data, headers=["Parameter", "Value"], tablefmt="grid"))

    # Plot the actual data points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x1_values, x2_values, y_values, label='Actual data')

    # Create a meshgrid for the surface representing the regression model
    x1_range = np.linspace(min(x1_values), max(x1_values), 100)
    x2_range = np.linspace(min(x2_values), max(x2_values), 100)
    x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)

    # Interaction terms for the meshgrid
    x1_x2_mesh = x1_mesh.flatten() * x2_mesh.flatten()
    x1_squared_x2_mesh = x1_mesh.flatten() ** 2 * x2_mesh.flatten()
    x1_x2_squared_mesh = x1_mesh.flatten() * x2_mesh.flatten() ** 2

    x_mesh = sm.add_constant(np.column_stack((x1_mesh.flatten(), x2_mesh.flatten(),
                                              x1_mesh.flatten() ** 2, x2_mesh.flatten() ** 2,
                                              x1_mesh.flatten() ** 3, x2_mesh.flatten() ** 3,
                                              x1_x2_mesh, x1_squared_x2_mesh, x1_x2_squared_mesh)))
    y_pred_mesh = np.dot(x_mesh, coefficients)

    # Reshape the predictions to the shape of the meshgrid
    y_pred_mesh = y_pred_mesh.reshape(x1_mesh.shape)

    # Plot the regression surface
    ax.plot_surface(x1_mesh, x2_mesh, y_pred_mesh, alpha=0.5, color='red', label='Regression surface')

    # Add labels and legend
    ax.zaxis.set_rotate_label(False)
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.labelpad = 20
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 35
    ax.set_xlabel('Gas Flow\n(MMSCFD)')
    ax.set_ylabel('Oil Flow\n(x10$^{3}$ bbl/d)\n\n\n\n\n\n\n\n\n', linespacing=1)
    ax.set_zlabel('Total Duty\n(MW)', rotation=0)
    y_max_graph = 80000
    y_tick_size = 20000

    y_ticks = [20000 * numnum for numnum in range(int((y_max_graph / y_tick_size) + 1))]
    ax.set_ylim(0, y_max_graph)  # Oil Flow
    ax.set_xlim(0, 45)  # Gas Flow
    ax.set_yticks(y_ticks)

    # Create a legend with the scatter plot and a proxy artist
    legend_labels = ['Simulation Data', 'Response Surface']
    legend_proxy = [plt.Line2D([0], [0], linestyle='none', c=scatter.get_facecolor(), marker='o'),
                    plt.Rectangle((0, 0), 1, 1, fc='red', alpha=0.5)]

    ax.legend(legend_proxy, legend_labels)
    textstr = '\n'.join((
        r'     R$^2=%.3f$' % r_squared,
        r'MAPE$=%.2f$%%' % mape
    ))
    props = dict(boxstyle='round', facecolor='white', edgecolor='lightgray', alpha=1)

    ax.text(x=0, y=80, z=245, s=textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    # Adjust y-axis labels
    oil_labels = ax.get_yticklabels()
    for label in oil_labels:
        cur_val = float(label.get_text())
        fixed_val = cur_val / 1000
        label.set_text(f'{int(fixed_val)}')  # Format to one decimal place

    ax.set_yticklabels(oil_labels,
                       rotation=0,
                       verticalalignment='baseline',
                       horizontalalignment='left'
                       )

    # Set the view to a specific angle
    # ax.view_init(elev=0, azim=90)

    pth = fd.asksaveasfilename(
        title='Choose dir to save',
        initialdir=(r'C:\Users\callk\Documents\02 University of Edinburgh\01 MEng Project\00 Digital '
                    r'Twins\Graphs\Response Surfaces'),
        filetypes=[('Picture', '.png')]
    )
    plt.savefig(pth, dpi=1200)

    # Show the 3D plot
    plt.show()

    return coefficients, mape, r_squared
