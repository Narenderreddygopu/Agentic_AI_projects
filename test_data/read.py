import pandas as pd

file_path = r"/mnt/c/Users/gopur/OneDrive/Documents/Agentic_ai/test_data/data.csv"

# Read CSV
df = pd.read_csv(file_path)
print(df)
def fun(a,b): #definition of a function
    print("calling from the function")
    print(a+b)

#fun(5,7) # calling a function


print("\n using info function \n")
print(df.info())
print("\n")

print("\n using describe function \n")
print(df.describe())
print("\n")
print("\n missing values per column \n")
print(df.isnull())

print("\n total per column \n")
print(df.isnull().sum())

print("\n filling the null values \n")
df["age"]=df["age"].fillna(df["age"].mean())
#df["spend"]=df["spend"].fillna(0)
print(df)
print("\n")
print("\n QUANTILE FUNC PER%TILE FOR BELOW PERCENTAGES :: \n")
print(df[["age", "signup_days"]].quantile([0.15,0.70]))
print("\n")
df["high_sign_up_days"] = df["signup_days"] > df["signup_days"].quantile(0.50)
print(df)
print("\n")
df["low_sign_up_days"] = df["signup_days"] < df["signup_days"].quantile(0.15)
print(df)
#
print("\n")
df["lowest_spend"] = df["spend"] < df["spend"].quantile(0.85) #
print("\n")
df["efficiency"]=(df["high_sign_up_days"])&(df["lowest_spend"])
print(df)

print("\n")
print(f"Mean of attributes \n: {df.mean(numeric_only=True)}")
print("\n")
print(df.median(numeric_only=True))
print("\n using mode f : \n")
print(df.mode(numeric_only=True))
print("\n using func std : \n")
print(df.std(numeric_only=True))
print("\n using func var : \n")
print(df.var(numeric_only=True))
#print(df.quantile(0.25, numeric_only=True))