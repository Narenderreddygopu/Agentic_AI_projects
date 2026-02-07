
import pandas as pd
import numpy as np
import json
"""
 #read data from a csv file
file_path = r"/mnt/c/Users/gopur/OneDrive/Documents/Agentic_ai/testing/data_out.csv"

# Read CSV
df = pd.read_csv(file_path)
print(df)

# write data to a csv file
df["high_value"] = df["spend"] > 2000
df.to_csv(/mnt/c/Users/gopur/OneDrive/Documents/Agentic_ai/testing/data_out.csv, index=False)
"""
import pandas as pd

file_path = "/mnt/c/Users/gopur/OneDrive/Documents/Agentic_ai/testing/data_out.csv"

df = pd.read_csv(file_path)
df["high_value"] = df["spend"] > 2000
df.to_csv(file_path, index=False)

print("CSV updated successfully")

# r - read mode , w - write mode , a - append mode, r+ == w+ , a+
content = {
    "file" : "path to file"
}

# writing in json file
#ith open("D:\\Agentic_Course\\testing\\test.json", "a") as f:
#   json.dump(content, f)

#f = r"/mnt/c/Users/gopur/OneDrive/Documents/Agentic_ai/testing/data_out.csv"
# reading from json file
with open(r"/mnt/c/Users/gopur/OneDrive/Documents/Agentic_ai/testing/test.json") as f:
    data = json.load(f)
    print(data)




try : 
    with open(r"/mnt/c/Users/gopur/OneDrive/Documents/Agentic_ai/testing/test.json") as f:
        history = json.load(f)
        print(history)
except FileNotFoundError :
    history = []
