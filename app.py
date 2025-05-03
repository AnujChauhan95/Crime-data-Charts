import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Oorja EDA Dashboard", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("train0.csv")  # Ensure this file is in the Render project directory
    return df

df = load_data()

# Title
st.title("Oorja Exploratory Data Analysis Dashboard")

# Basic info
st.subheader("Dataset Overview")
st.write("Shape of the dataset:", df.shape)
st.write("First few rows:")
st.dataframe(df.head(), use_container_width=True)

# Column types
st.subheader("Data Types")
st.write(df.dtypes)

# Missing values
st.subheader("Missing Value Summary")
st.write(df.isnull().sum())

# Sidebar filters (for interactivity)
st.sidebar.header("Filter Options")
cat_cols = df.select_dtypes(include='object').columns.tolist()

filtered_df = df.copy()
for col in cat_cols:
    values = ["All"] + sorted(df[col].dropna().unique().tolist())
    selection = st.sidebar.selectbox(f"Filter by {col}", values, key=col)
    if selection != "All":
        filtered_df = filtered_df[filtered_df[col] == selection]

st.subheader("Filtered Data Preview")
st.dataframe(filtered_df, use_container_width=True)

# Charts based on notebook

# 1. Value counts: Gender
st.subheader("Gender Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(data=filtered_df, x='Gender', ax=ax1)
st.pyplot(fig1)

# 2. Value counts: Education_Level
st.subheader("Education Level Distribution")
fig2, ax2 = plt.subplots()
sns.countplot(data=filtered_df, x='Education_Level', ax=ax2)
plt.xticks(rotation=45)
st.pyplot(fig2)

# 3. Value counts: Ever_Married
st.subheader("Marital Status")
fig3, ax3 = plt.subplots()
sns.countplot(data=filtered_df, x='Ever_Married', ax=ax3)
st.pyplot(fig3)

# 4. Value counts: Profession
st.subheader("Profession Distribution")
fig4, ax4 = plt.subplots(figsize=(10, 4))
sns.countplot(data=filtered_df, x='Profession', ax=ax4)
plt.xticks(rotation=45)
st.pyplot(fig4)

# 5. Count by Age Groups
st.subheader("Age Distribution")
fig5, ax5 = plt.subplots()
sns.histplot(filtered_df['Age'].dropna(), kde=True, ax=ax5)
st.pyplot(fig5)

# 6. Spending Score Distribution
st.subheader("Spending Score Distribution")
fig6, ax6 = plt.subplots()
sns.histplot(filtered_df['Spending_Score'].dropna(), kde=True, ax=ax6)
st.pyplot(fig6)

# 7. Work Experience Distribution
st.subheader("Work Experience Distribution")
fig7, ax7 = plt.subplots()
sns.histplot(filtered_df['Work_Experience'].dropna(), kde=True, ax=ax7)
st.pyplot(fig7)

# 8. Family Size Distribution
st.subheader("Family Size Distribution")
fig8, ax8 = plt.subplots()
sns.histplot(filtered_df['Family_Size'].dropna(), kde=True, ax=ax8)
st.pyplot(fig8)

# Footer
st.markdown("---")
st.caption("This dashboard replicates all charts and logic from the original Oorja EDA notebook.")
