import pandas as pd

# Path to your file
file_path = r"/mnt/c/Users/gopur/OneDrive/Documents/Agentic_ai/Agentic_Course/data.csv"

# Read CSV
df = pd.read_csv(file_path)

# Show first 5 rows
print(df.head())
print ("\nData read successfully!\n")

#print(df.info())
print("\nData read info successfully!\n",df.info())

print("\nSummary Statistics:\n", df.describe())
    