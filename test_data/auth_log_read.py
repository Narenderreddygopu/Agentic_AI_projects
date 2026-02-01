import pandas as pd
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
start_time = datetime.now()
file_path = "https://raw.githubusercontent.com/Narenderreddygopu/Agentic_AI_projects/main/test_data/AuthContractors.csv"
logging.info(f"Process started at: {start_time}")
try:
    logging.info(f"Starting file load: {file_path}")

    # Read CSV
    df = pd.read_csv(file_path)

    rows_loaded = len(df)
    cols_loaded = len(df.columns)

    logging.info(f"File loaded successfully")
    logging.info(f"Rows loaded: {rows_loaded}")
    logging.info(f"Columns loaded: {cols_loaded}")

    # -------- Data Cleaning Example --------
    rows_before = len(df)

    # Example: remove duplicate rows
    df_cleaned = df.drop_duplicates()

    rows_after = len(df_cleaned)

    diff = rows_before - rows_after

    logging.info(f"Rows before cleaning: {rows_before}")
    logging.info(f"Rows after cleaning: {rows_after}")
    logging.info(f"Duplicate rows removed: {diff}")

except Exception as e:
    logging.error(f"File load failed: {e}", exc_info=True)

finally:
    end_time = datetime.now()
    duration = end_time - start_time
    logging.info("Job finished")
    logging.info(f"End time: {end_time}")
    logging.info(f"Total duration: {duration}")