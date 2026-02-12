"""
==================================
DATA CLEANING WITH PROPER LOGGING
==================================
"""

import logging
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("=" * 50)
logger.info("Setup logging is done (using logger)")
logger.info("=" * 50)

logger.info("Starting data testing...")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
logger.info("Loading data...")
file_path = "/mnt/c/Users/gopur/OneDrive/Documents/Agentic_ai/testing/employee_data.csv"

try:
    df = pd.read_csv(file_path)
    logger.info(f"Data loaded successfully:- {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    logger.error(f"❌ File not found: {file_path}")
    exit()

# Show initial data info
logger.info(f"Columns: {df.columns.tolist()}")
logger.info(f"Missing values before cleaning:\n{df.isnull().sum()}")

# ============================================================================
# STEP 2: HANDLE MISSING VALUES
# ============================================================================
logger.info("Checking for missing values...")

# ISSUE FIX #1: You need to assign the result back to the DataFrame!
# Strategy 1: Fill with constant (MUST ASSIGN!)
logger.info("Filling missing names with 'Unknown'...")
df['Name'] = df['Name'].fillna('Unknown')  # ← Added assignment
logger.info(f"✓ Names filled: {(df['Name'] == 'Unknown').sum()} values")

# Strategy 2: Fill with median (MUST ASSIGN!)
logger.info("Filling missing ages with median...")
age_median = df['Age'].median()
df['Age'] = df['Age'].fillna(age_median)  # ← Added assignment
logger.info(f"✓ Ages filled with median: {age_median}")

# Strategy 3: Fill by group (MUST ASSIGN!)
logger.info("Filling missing salaries by department median...")
df['Salary'] = df.groupby('Department')['Salary'].transform(
    lambda x: x.fillna(x.median())
)  # ← Added assignment
logger.info(f"✓ Salaries filled by department")

# Fill any other missing values
logger.info("Handling remaining missing values...")
for col in df.columns:
    missing_count = df[col].isnull().sum()
    if missing_count > 0:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
            logger.info(f"  Filled {col}: {missing_count} missing values with median")
        else:
            df[col] = df[col].fillna('Unknown')
            logger.info(f"  Filled {col}: {missing_count} missing values with 'Unknown'")

# ============================================================================
# STEP 3: HANDLE OUTLIERS
# ============================================================================
logger.info("Handling outliers...")

# Using percentiles
Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1
outlier_mask = (df['Salary'] < Q1 - 1.5*IQR) | (df['Salary'] > Q3 + 1.5*IQR)
outlier_count = outlier_mask.sum()

logger.info(f"Outliers detected: {outlier_count}")
logger.info(f"  Q1: ${Q1:,.2f}")
logger.info(f"  Q3: ${Q3:,.2f}")
logger.info(f"  IQR: ${IQR:,.2f}")
logger.info(f"  Lower bound: ${Q1 - 1.5*IQR:,.2f}")
logger.info(f"  Upper bound: ${Q3 + 1.5*IQR:,.2f}")

# Cap at 95th percentile
salary_95th = df['Salary'].quantile(0.95)
df['Salary'] = df['Salary'].clip(upper=salary_95th)
logger.info(f"✓ Salaries capped at 95th percentile: ${salary_95th:,.2f}")

# ============================================================================
# STEP 4: DATA VALIDATION
# ============================================================================
logger.info("Running data validation checks...")

# ISSUE FIX #2: Age might still be float after fillna, need to convert first
# Convert Age to integer BEFORE checking
df['Age'] = df['Age'].astype(int)
logger.info("✓ Converted Age to integer")

# Check for negative salaries
try:
    assert (df['Salary'] >= 0).all(), "Found negative salaries!"
    logger.info("✓ No negative salaries found")
except AssertionError as e:
    logger.error(f"❌ Validation failed: {e}")
    # Show the problematic rows
    negative_salaries = df[df['Salary'] < 0]
    logger.error(f"Rows with negative salaries:\n{negative_salaries}")

# Check for duplicates
duplicate_count = df.duplicated().sum()
if duplicate_count > 0:
    logger.warning(f"⚠ Found {duplicate_count} duplicate rows - removing them...")
    df = df.drop_duplicates()
    logger.info(f"✓ Removed {duplicate_count} duplicates")
else:
    logger.info("✓ No duplicate rows found")

# ISSUE FIX #3: Use try-except for assertions to handle failures gracefully
try:
    assert df['Age'].dtype == 'int64', f"Age is {df['Age'].dtype}, not int64"
    logger.info("✓ Age data type is correct (int64)")
except AssertionError as e:
    logger.error(f"❌ Data type check failed: {e}")

# Additional validation checks
logger.info("Running additional validation checks...")

# Check for reasonable age range
if (df['Age'] < 18).any() or (df['Age'] > 100).any():
    logger.warning("⚠ Found ages outside reasonable range (18-100)")
    unreasonable_ages = df[(df['Age'] < 18) | (df['Age'] > 100)]
    logger.warning(f"  {len(unreasonable_ages)} rows affected")
else:
    logger.info("✓ All ages are within reasonable range")

# Check for reasonable salary range
if (df['Salary'] < 0).any() or (df['Salary'] > 1000000).any():
    logger.warning("⚠ Found salaries outside reasonable range ($0-$1M)")
else:
    logger.info("✓ All salaries are within reasonable range")

# ============================================================================
# STEP 5: FINAL SUMMARY
# ============================================================================
logger.info("=" * 50)
logger.info("FINAL DATA SUMMARY")
logger.info("=" * 50)

logger.info(f"Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
logger.info(f"Missing values: {df.isnull().sum().sum()}")
logger.info(f"Duplicate rows: {df.duplicated().sum()}")

logger.info("\nData types:")
for col, dtype in df.dtypes.items():
    logger.info(f"  {col}: {dtype}")

logger.info("\nBasic statistics:")
logger.info(f"  Average Age: {df['Age'].mean():.1f}")
logger.info(f"  Average Salary: ${df['Salary'].mean():,.2f}")
logger.info(f"  Min Salary: ${df['Salary'].min():,.2f}")
logger.info(f"  Max Salary: ${df['Salary'].max():,.2f}")

# ============================================================================
# STEP 6: SAVE CLEANED DATA
# ============================================================================
output_path = "/mnt/c/Users/gopur/OneDrive/Documents/Agentic_ai/testing/cleaned_data_with_logging.csv"
df.to_csv(output_path, index=False)
logger.info(f"✓ Cleaned data saved to: {output_path}")

logger.info("=" * 50)
logger.info(" Data testing completed successfully!")
logger.info("=" * 50)