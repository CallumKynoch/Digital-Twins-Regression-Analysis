import numpy as np
import pandas as pd
import win32com.client
import time
import pyautogui
import itertools

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
    workbook = ExcelApp.Workbooks.Open(r'C:\Users\Callum Kynoch\Documents\01 University of Edinburgh\MEng Project - '
                                       r'Digital Twins\04 Experimental Data\example_2.xlsx')

    min_value_range1 = int(input("Enter the minimum value for Range 1: "))
    max_value_range1 = int(input("Enter the maximum value for Range 1: "))
    step_range1 = int(input("Enter the step change for Range 1: "))

    min_value_range2 = int(input("Enter the minimum value for Range 2: "))
    max_value_range2 = int(input("Enter the maximum value for Range 2: "))
    step_range2 = int(input("Enter the step change for Range 2: "))

    # Generate lists of values for Range 1 and Range 2
    value_range1 = list(range(min_value_range1, max_value_range1 + 1, step_range1))
    value_range2 = list(range(min_value_range2, max_value_range2 + 1, step_range2))

    combinations = list(itertools.product(value_range1, value_range2))

    num_simulations = len(combinations)

    num_variables = int(input("Enter the number of measured variables: "))
    for i in range(0, num_variables):
        import_variables.append(input("Enter the import variables in format -> name cell: ").split())

    simulation_results = np.empty((num_simulations, len(export_variables) + len(import_variables)))

    # Read the Reference Value taking cell from first import variable
    if len(import_variables) > 1:
        reference_cell = import_variables[1][1]
    else:
        reference_cell = "B93"

    prev_reference_value = ExcelApp.Range(reference_cell).Value

    # Randomly generate initial gas flow
    initial_gasflow = min_value_range1

    # Update Excel to initial gas flow
    ExcelApp.Range("B50").Value = initial_gasflow

    # Randomly generate initial GOR
    initial_oilflow = min_value_range2

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

            # Update previous reference value to current reference value
            prev_reference_value = ExcelApp.Range(reference_cell).Value

            # Extract gas and oil values from the combinations list
            next_gasflow_value, next_oilflow_value = combinations[sim+1]

            ExcelApp.Range(gas_flow_cell).Value = next_gasflow_value
            ExcelApp.Range(oil_flow_cell).Value = next_oilflow_value

            # Update values of variables in results array
            # gasflow_list.append(gasflow_value)
            simulation_results[sim, 0] = gasflow_value

            # oilflow_list.append(oilflow_value)
            simulation_results[sim, 1] = oilflow_value

            for i, variable in enumerate(import_variables):
                cell = variable[1]
                simulation_results[sim, i + 2] = ExcelApp.Range(cell).Value

            end = time.time()

            print(f"SIMULATION {sim + 1} COMPLETED. Time taken: {end - start:.2f} s")

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

    headers = [export_variables[0][0], export_variables[1][0]]
    for i, variable in enumerate(import_variables):
        headers.append(variable[0])

    # Create a new Excel workbook and write collected data to separate columns
    df = pd.DataFrame(simulation_results)
    print(df.shape)
    df.to_excel(f'simulation_{time.strftime("%Y%m%d-%H%M%S")}.xlsx', index=False, header=headers, engine='openpyxl')
