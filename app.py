import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="Oorja EDA Dashboard", layout="wide")

# Load the dataset
@st.cache_data

def load_data():
    df = pd.read_csv("train0.csv")  # Replace with your actual CSV file
    return df

df = load_data()

# Sidebar - Filters
st.sidebar.header("Filters")

# Dynamically create dropdowns for categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
filters = {}
for col in categorical_cols:
    options = df[col].dropna().unique().tolist()
    selected = st.sidebar.selectbox(f"Select {col}", ["All"] + sorted(options), key=col)
    if selected != "All":
        df = df[df[col] == selected]
        filters[col] = selected

# Title
st.title("Oorja EDA Dashboard")

# Show filtered data
st.subheader("Filtered Data")
st.dataframe(df, use_container_width=True)

# Show count plots for categorical columns
st.subheader("Categorical Column Count Plots")
for col in categorical_cols:
    fig, ax = plt.subplots()
    sns.countplot(data=df, x=col, order=df[col].value_counts().index, ax=ax)
    ax.set_title(f"Count Plot for {col}")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Show distribution plots for numeric columns
st.subheader("Numeric Column Distributions")
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
for col in numeric_cols:
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f"Distribution for {col}")
    st.pyplot(fig)

if numeric_column in df.columns:
    st.subheader(f"Distribution of {numeric_column}")
    fig, ax = plt.subplots()
    sns.histplot(df[numeric_column], kde=True, ax=ax)
    st.pyplot(fig)
