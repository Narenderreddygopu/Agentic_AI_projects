import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- Streamlit Layout ----------------
st.set_page_config(page_title="CSV Loader & Cleaner", layout="wide")
st.title("📊 CSV Loader & Dashboard")

# ---------------- File Input ----------------
st.sidebar.header("File Options")

#  GitHub file URL Auth
default_file_url = "https://raw.githubusercontent.com/Narenderreddygopu/Agentic_AI_projects/main/test_data/AuthContractors.csv"
st.sidebar.markdown(f"**Default CSV URL:** {default_file_url}")

uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
file_url_input = st.sidebar.text_input("Or enter CSV URL here", default_file_url)

# ---------------- Load Data ----------------
df = None
try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("File loaded successfully from upload!")
    elif file_url_input:
        df = pd.read_csv(file_url_input)
        st.success("File loaded successfully from URL!")
except Exception as e:
    st.error(f"Failed to load CSV: {e}")

# ---------------- Show Raw Data ----------------
if df is not None:
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())
    st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")

    # ---------------- Data Cleaning Options ----------------
    st.sidebar.header("Data Cleaning Options")
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

    # ---------------- Show Cleaned Data ----------------
    st.subheader("Cleaned Data Preview")
    st.dataframe(df_cleaned.head())
    st.write(f"Rows after cleaning: {rows_after}")
    st.write(f"Duplicate rows removed: {duplicates_removed}")

    # ---------------- Dataset Summary ----------------
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
