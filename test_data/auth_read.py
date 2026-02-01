import pandas as pd

repo = "Narenderreddygopu/Agentic_AI_projects"
branch = "main"
folder = "test_data"
file = "AuthContractors.csv"

file_path = f"https://raw.githubusercontent.com/{repo}/{branch}/{folder}/{file}"

df = pd.read_csv(file_path)

print(f"File loaded from: {file_path}")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print(df.head())
