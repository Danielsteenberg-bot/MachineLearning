import numpy as np
import pandas as pd

# Create and populate a 5x2 NumPy array.
my_data = np.array([[0, 3], [10, 7], [20, 9], [30, 14], [40, 15]])

# Create a Python list that holds the names of the two columns.
my_column_names = ['temperature', 'activity']

# Create a DataFrame.
my_dataframe = pd.DataFrame(data=my_data, columns=my_column_names)

# Print the entire DataFrame
print(my_dataframe)

# Opret en 3x4 matrix med tilfældige heltal mellem 0 og 100
random_data = np.random.randint(0, 101, size=(3, 4))  

# Definer kolonne navne
columnName = ['Eleanor', 'Chidi', 'Tahani', 'Jason']

# Opret DataFrame med de tilfældige tal og kolonne navne
name_frame = pd.DataFrame(data=random_data, columns=columnName)
firstElanor = name_frame.loc[1, 'Eleanor']

# Tilføj en 5 kollone 
name_frame["janet"] = name_frame["Tahani"] + name_frame["Jason"]


print("\nTilfældig 3x4 DataFrame:")
print(name_frame, "name_Frame")
print(firstElanor, 'firstElanor')
