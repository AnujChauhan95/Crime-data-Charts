import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set wide page layout
st.set_page_config(page_title="Oorja EDA Dashboard", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("train0.csv")
    return df

df = load_data()

st.title("Oorja EDA Dashboard")

# Show shape and preview
st.subheader("Dataset Overview")
st.write(f"Shape of the dataset: {df.shape}")
st.dataframe(df.head(), use_container_width=True)

# Show dtypes and missing values
st.subheader("Data Types and Missing Values")
col1, col2 = st.columns(2)
with col1:
    st.write("**Column Data Types**")
    st.write(df.dtypes)
with col2:
    st.write("**Missing Values**")
    st.write(df.isnull().sum())

# Sidebar filters
st.sidebar.header("Filter Options")
cat_cols = df.select_dtypes(include='object').columns.tolist()
filtered_df = df.copy()
for col in cat_cols:
    values = ["All"] + sorted(df[col].dropna().unique().tolist())
    selection = st.sidebar.selectbox(f"Filter by {col}", values, key=col)
    if selection != "All":
        filtered_df = filtered_df[filtered_df[col] == selection]

# Visualizations

# Categorical plots
if cat_cols:
    st.subheader("Categorical Column Distributions")
    for col in cat_cols:
        try:
            st.markdown(f"**{col}**")
            fig, ax = plt.subplots()
            sns.countplot(data=filtered_df, x=col, order=filtered_df[col].value_counts().index, ax=ax)
            plt.xticks(rotation=45)
            ax.set_title(f"{col} Count Plot")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not plot `{col}`: {e}")

# Numerical plots
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if num_cols:
    st.subheader("Numerical Column Distributions")
    for col in num_cols:
        try:
            st.markdown(f"**{col}**")
            fig, ax = plt.subplots()
            sns.histplot(filtered_df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"{col} Distribution")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not plot `{col}`: {e}")

st.markdown("---")
st.caption("All visualizations dynamically generated from train0.csv")
