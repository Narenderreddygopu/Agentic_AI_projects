import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime , timedelta

# ---------------- Streamlit Layout ----------------
st.set_page_config(page_title="CSV Loader & Cleaner", layout="wide")
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>📊 CSV Loader & Dashboard</h1>",
    unsafe_allow_html=True
)

# ---------------- File Input ----------------
st.sidebar.header("File Options")
default_file_url = "https://raw.githubusercontent.com/Narenderreddygopu/Agentic_AI_projects/main/test_data/AuthContractors.csv"
st.sidebar.markdown(f"**Default CSV URL:** {default_file_url}")

uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
file_url_input = st.sidebar.text_input("Or enter CSV URL here", default_file_url)

# ---------------- Track start time ----------------
start_time_total = datetime.now()
st.write(f"⏱ Process started at: {start_time_total.strftime('%Y-%m-%d %H:%M:%S')}")

# Stage timers
timing = {}

# ---------------- Load Data ----------------
stage_start = datetime.now()
df = None
try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif file_url_input:
        df = pd.read_csv(file_url_input)
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
stage_end = datetime.now()
timing['Load CSV'] = (stage_end - stage_start).total_seconds()

# ---------------- Data Cleaning ----------------
if df is not None:
    stage_start = datetime.now()
    remove_duplicates = st.sidebar.checkbox("Remove duplicate rows")
    drop_na = st.sidebar.checkbox("Drop rows with missing values")
    lowercase_cols = st.sidebar.checkbox("Lowercase column names")
    
    df_cleaned = df.copy()
    rows_before = len(df_cleaned)
    
    if remove_duplicates:
        df_cleaned = df_cleaned.drop_duplicates()
    if drop_na:
        df_cleaned = df_cleaned.dropna()
    if lowercase_cols:
        df_cleaned.columns = df_cleaned.columns.str.lower()
    
    rows_after = len(df_cleaned)
    duplicates_removed = rows_before - rows_after
    stage_end = datetime.now()
    timing['Clean Data'] = (stage_end - stage_start).total_seconds()

    # ---------------- Display cleaned data & metrics ----------------
    st.subheader("Cleaned Data Preview")
    st.dataframe(df_cleaned.head())
    st.write(f"Rows after cleaning: {rows_after}, Duplicates removed: {duplicates_removed}")
    
    st.subheader("Dataset Summary")
    st.metric("Rows before cleaning", rows_before)
    st.metric("Rows after cleaning", rows_after)
    st.metric("Columns", len(df_cleaned.columns))
    st.metric("Duplicates removed", duplicates_removed)

    # ---------------- Column Type Analysis ----------------
    st.subheader("Column Type Analysis")
    col_types = pd.DataFrame(df_cleaned.dtypes, columns=['Data Type'])
    st.dataframe(col_types)

    # ---------------- Missing Values Heatmap ----------------
    st.subheader("Missing Values Heatmap")
    if df_cleaned.isnull().sum().sum() > 0:
        plt.figure(figsize=(8,4))
        sns.heatmap(df_cleaned.isnull(), cbar=False, cmap='viridis')
        st.pyplot(plt)
    else:
        st.write("No missing values detected ✅")

    # ---------------- Numeric Column Distributions ----------------
    numeric_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if numeric_cols:
        st.subheader("Numeric Columns Distribution")
        selected_col = st.selectbox("Select numeric column", numeric_cols)
        plt.figure(figsize=(8,4))
        sns.histplot(df_cleaned[selected_col], kde=True)
        st.pyplot(plt)

    # ---------------- Download Cleaned CSV ----------------
    st.subheader("Download Cleaned Data")
    csv = df_cleaned.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned CSV",
        data=csv,
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

# ---------------- Track end time ----------------
end_time_total = datetime.now()
timing['Total'] = (end_time_total - start_time_total).total_seconds()

st.write(f"⏱ Process finished at: {end_time_total.strftime('%Y-%m-%d %H:%M:%S')}")
st.write(f"⏱ Total duration: {timedelta(seconds=int(timing['Total']))}")

# ---------------- Pie Chart for Time Distribution ----------------
st.subheader("⏱ Time Spent per Stage")
if len(timing) > 1:
    # Remove 'Total' from pie
    pie_data = {k: v for k, v in timing.items() if k != 'Total'}
    plt.figure(figsize=(6,6))
    plt.pie(pie_data.values(), labels=pie_data.keys(), autopct='%1.1f%%', startangle=90, colors=['#4CAF50','#FF5733'])
    plt.title("Time distribution per stage (seconds)")
    st.pyplot(plt)
