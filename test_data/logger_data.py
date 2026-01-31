import pandas as pd
import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# File path
file_path = r"/mnt/c/Users/gopur/OneDrive/Documents/Agentic_ai/test_data/data.csv"

try:
    logger.info("Starting CSV read process...")

    # Read CSV
    df = pd.read_csv(file_path)
    logger.info("CSV file read successfully")

    # Show first rows
    logger.info("First 5 rows:")
    logger.info("\n%s", df.head())

    # Info
    logger.info("Data Info:")
    buffer = []
    df.info(buf=buffer)
    logger.info("\n".join(buffer))

    # Statistics
    logger.info("Summary Statistics:")
    logger.info("\n%s", df.describe())

except FileNotFoundError:
    logger.error("File not found: %s", file_path)

except Exception as e:
    logger.exception("Unexpected error occurred: %s", e)
