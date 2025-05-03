import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Oorja EDA Dashboard", layout="wide")

# Sidebar filters
st.sidebar.header("Filters")

# Load your data (replace with your actual file)
data_file = "data.csv"
@st.cache_data
def load_data():
    df = pd.read_csv(train0)
    return df

df = load_data()

# Example dropdown (adjust based on your dataset)
category_column = "Category"  # Replace with actual column
if category_column in df.columns:
    selected_category = st.sidebar.selectbox("Select Category", df[category_column].dropna().unique())
    df = df[df[category_column] == selected_category]

# Main layout
st.title("Oorja EDA Dashboard")

# Example: Show DataFrame
st.subheader("Filtered Data Preview")
st.dataframe(df)

# Example: Count plot
plot_column = "SubCategory"  # Replace with actual column
if plot_column in df.columns:
    st.subheader(f"Count Plot of {plot_column}")
    fig, ax = plt.subplots()
    sns.countplot(data=df, y=plot_column, ax=ax)
    st.pyplot(fig)

# Example: Numeric distribution
numeric_column = "Value"  # Replace with actual column
if numeric_column in df.columns:
    st.subheader(f"Distribution of {numeric_column}")
    fig, ax = plt.subplots()
    sns.histplot(df[numeric_column], kde=True, ax=ax)
    st.pyplot(fig)
