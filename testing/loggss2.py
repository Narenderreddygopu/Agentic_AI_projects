import logging
import pandas as pd

logging.basicConfig(
    #filename = "test2.log",
    level = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s"
)

file_path = r"/mnt/c/Users/gopur/OneDrive/Documents/Agentic_ai/testing/data_out.csv"

# Read CSV
df = pd.read_csv(file_path)
#df = pd.read_csv(r"/mnt/c/Users/gopur/OneDrive/Documents/Agentic_ai/testing/data_out.csv")
logging.info("the code has begun")
for _, row in df.iterrows():
    if row["spend"] > 2000 :
        logging.warning(f"Customer {row['user_id']} is a high value customer")
    else : 
        logging.info(f"Customer {row['user_id']} is a low value customer")
logging.info("the code has ended")