import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = r"/mnt/c/Users/gopur/OneDrive/Documents/Agentic_ai/test_data/data.csv"

# Read CSV
df = pd.read_csv(file_path)
print(df)

df_extr = df.copy()
print(df_extr)

print(len(df_extr))

# Filter age == 29
print(df_extr.loc[df_extr["age"] == 29])

# Access row with index label 1
print(df_extr.loc[1])

# Add a new row
df_extr.loc[len(df_extr)] = [7, 44, "India", 50, 5000]

# ✅ Correct way to access the newly added row
print(df_extr.iloc[-1])   # last row (safe)

# Plot histogram
plt.hist(df_extr["spend"].dropna(), bins=10)
#plt.show()
# Save the plot as an image file
plt.savefig('spend_histogram.png')