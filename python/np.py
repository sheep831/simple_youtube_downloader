import pandas as pd

# Create a DataFrame with missing values
data = {'A': [1, None, 3, 4], 'B': [5, 6, None, 8]}
df = pd.DataFrame(data)

# Drop rows with missing values in-place
df.dropna(inplace=True)

# The original DataFrame is modified
print(df)
