import numpy as np
import pandas as pd
import win32com.client
import time
import pyautogui

export_variables = [["gas_flow_rate", "B50"], ["oil_flow_rate", "B6"]]
import_variables = []
simulation_results = np.empty(0)


def wait_for_ref_value_change():
    current_reference_value = ExcelApp.Range(reference_cell).Value
    while current_reference_value == prev_reference_value:
        time.sleep(1)
        current_reference_value = ExcelApp.Range(reference_cell).Value


try:
    # Create an instance of the Excel Application & make it visible.
    ExcelApp = win32com.client.GetActiveObject("Excel.Application")
    ExcelApp.Visible = True

    # Open the desired workbook
    workbook = ExcelApp.Workbooks.Open(r"C:\Users\Symmetry Machine\Documents\MEng Project\Digital Twins\compressor_test.xlsx")

    # Create a Normal Distribution for Gas Flow in MMSCFD
    gasflow_mean = int(input("Enter the Mean Gas Glow: "))
    gasflow_std_dev = int(input("Enter the Gas Flow's standard deviation: "))

    # Create a Normal Distribution for Gas-Oil Ratio in Mbbl/d (12,000 bbl/d = 0.012 Mbbl/d)
    GOR_mean = int(input("Enter the Mean Gas-Oil Ratio: "))
    GOR_std_dev = int(input("Enter the Gas-Oil Ratio's standard deviation: "))

    num_simulations = int(input("Enter the number of simulations: "))  # User input for the number of iterations

    num_variables = int(input("Enter the number of measured variables: "))
    for i in range(0, num_variables):
        import_variables.append(input("Enter the import variables in format -> name cell: ").split())

    simulation_results = np.empty((num_simulations, len(export_variables) + len(import_variables) + 1))

    # Read the Reference Value taking cell from first import variable
    if len(import_variables) > 1:
        reference_cell = import_variables[1][1]
    else:
        reference_cell = "B93"

    prev_reference_value = ExcelApp.Range(reference_cell).Value

    # Randomly generate initial gas flow
    initial_gasflow = np.random.normal(loc=gasflow_mean, scale=gasflow_std_dev)

    # Update Excel to initial gas flow
    ExcelApp.Range("B50").Value = initial_gasflow

    # Randomly generate initial GOR
    initial_GOR = np.random.normal(loc=GOR_mean, scale=GOR_std_dev)

    # Determine Oil Flow in bbl/d by dividing Gas Flow by GOR
    initial_oilflow = (initial_gasflow / initial_GOR) * 10**6

    ExcelApp.Range("B6").Value = initial_oilflow

    # Tabs out of Python over to Excel for keyboard press simulation
    pyautogui.hotkey('alt', 'tab')

    # Wait for a short duration to ensure focus is on Excel
    time.sleep(2)

    for sim in range(num_simulations):
        start = time.time()
        # Print a message to indicate that the script is about to press the button
        print("Pressing the 'All' button...")
        # Navigate the Excel ribbon to locate the "all" button for importing and exporting data
        pyautogui.press(['alt', 'y', '2', 'y', '3'])
        # Print a message to indicate that the button has been pressed
        print(f"Button pressed. Waiting for {reference_cell} to change...")

        try:
            # Wait for the Reference Value to change
            wait_for_ref_value_change()
            gas_flow_cell = export_variables[0][1]
            oil_flow_cell = export_variables[1][1]

            # Read the current values of Gas Flow, GOR and Oil Flow
            gasflow_value = ExcelApp.Range(gas_flow_cell).Value
            oilflow_value = ExcelApp.Range(oil_flow_cell).Value
            GOR_value = (gasflow_value / oilflow_value) * 10**6

            # Update previous reference value to current reference value
            prev_reference_value = ExcelApp.Range(reference_cell).Value

            # Randomly select a Gas Flow and GOR, and calculate Oil Flow
            next_gasflow_value = np.random.normal(loc=gasflow_mean, scale=gasflow_std_dev)
            next_GOR = (np.random.normal(loc=GOR_mean, scale=GOR_std_dev)) / 10**6
            next_oilflow = next_gasflow_value / next_GOR

            ExcelApp.Range(gas_flow_cell).Value = next_gasflow_value
            ExcelApp.Range(oil_flow_cell).Value = next_oilflow

            # Update values of variables in results array
            # gasflow_list.append(gasflow_value)
            simulation_results[sim, 0] = gasflow_value

            # oilflow_list.append(oilflow_value)
            simulation_results[sim, 1] = oilflow_value

            # GOR_list.append(next_GOR)
            simulation_results[sim, 2] = GOR_value

            for i, variable in enumerate(import_variables):
                cell = variable[1]
                simulation_results[sim, i+3] = ExcelApp.Range(cell).Value

            end = time.time()

            print(f"SIMULATION {sim + 1} COMPLETED. Time taken: {end-start:.2f} s")

        except Exception as e:
            print("Error while updating cells gas and oil flow:", e)

        # Wait for a short duration to prevent rapid changes
        time.sleep(1)  # You can adjust the wait time as needed

except Exception as e:
    print("Error:", e)

finally:
    try:
        # Explicitly release COM objects
        del ExcelApp
        del workbook
    except Exception as e:
        print("Error while releasing COM objects:", e)

    headers = []
    headers.append(export_variables[0][0])
    headers.append(export_variables[1][0])
    headers.append('GOR')
    for i, variable in enumerate(import_variables):
        headers.append(variable[0])

    # Create a new Excel workbook and write collected data to separate columns
    df = pd.DataFrame(simulation_results)
    df.to_excel(f'simulation_{time.strftime("%Y%m%d-%H%M%S")}.xlsx', index=False, header=headers, engine='openpyxl')