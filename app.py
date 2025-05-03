import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page layout
st.set_page_config(page_title="Oorja EDA Dashboard", layout="wide")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")  # Replace with your actual CSV exported from the notebook
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")

# Categorical dropdown filters
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
for col in categorical_cols:
    options = ["All"] + sorted(df[col].dropna().unique().tolist())
    choice = st.sidebar.selectbox(f"Select {col}", options, key=col)
    if choice != "All":
        df = df[df[col] == choice]

# Title
st.title("Oorja EDA Dashboard")

# Show filtered data
st.subheader("Filtered Dataset")
st.dataframe(df, use_container_width=True)

# Plot categorical distributions
st.subheader("Count Plots for Categorical Columns")
for col in categorical_cols:
    st.markdown(f"**{col}**")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x=col, order=df[col].value_counts().index, ax=ax)
    ax.set_title(f"Count Plot for {col}")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Plot numerical distributions
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
if numeric_cols:
    st.subheader("Distribution of Numeric Columns")
    for col in numeric_cols:
        st.markdown(f"**{col}**")
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Histogram and KDE for {col}")
        st.pyplot(fig)

# Optional: Correlation Heatmap
if len(numeric_cols) >= 2:
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)
