import pandas as pd

# Load the CSV file
file_path = 'data/drift_data.csv'  # Replace with your file path
column_name = 'Dependents'  # Replace with your column name

# Read the CSV file
data = pd.read_csv(file_path)

# Ensure the column exists
if column_name in data.columns:
    # Calculate the halfway point of the DataFrame
    halfway_index = len(data) // 2
    
    # Set the first half of the column's values to 0
    data.loc[:halfway_index - 1, column_name] = 0
else:
    print(f"Column '{column_name}' not found in the CSV file.")

# Save the updated CSV file
output_file_path = 'data/drift_data2.csv'  # Replace with your desired output file path
data.to_csv(output_file_path, index=False)

print(f"All values in the column '{column_name}' have been set to 0 and saved to '{output_file_path}'.")
