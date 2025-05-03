
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit page config
st.set_page_config(layout="wide")
st.title("Oorja Crime Data EDA Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("train0.csv")
    df['Date_Occurred'] = pd.to_datetime(df['Date_Occurred'], format="%m/%d/%Y")
    df['Date_Reported'] = pd.to_datetime(df['Date_Reported'], format="%m/%d/%Y")
    df['Day_occ'] = df['Date_Occurred'].dt.day
    df['Day_rep'] = df['Date_Reported'].dt.day
    df['Month_occ'] = df['Date_Occurred'].dt.month
    df['Month_rep'] = df['Date_Reported'].dt.month
    df['Weapon_Description'].fillna("UNKNOWN WEAPON/OTHER WEAPON", inplace=True)
    df['Victim_Descent'].fillna(df['Victim_Descent'].mode()[0], inplace=True)
    df['Victim_Sex'].fillna(df['Victim_Sex'].mode()[0], inplace=True)
    df = df.drop_duplicates()
    df = df[df['Victim_Age'] >= 0]
    return df

data = load_data()

# Sidebar filters
st.sidebar.header("Filters")
crime_categories = data['Crime_Category'].dropna().unique()
selected_crime = st.sidebar.multiselect("Select Crime Category", crime_categories, default=crime_categories)

filtered_data = data[data['Crime_Category'].isin(selected_crime)]

# Visualization 1: Crime Category Count
st.subheader("Crime Category Distribution")
crime_counts = filtered_data['Crime_Category'].value_counts()
fig1, ax1 = plt.subplots(figsize=(20, 6))
sns.barplot(x=crime_counts.index, y=crime_counts.values, ax=ax1)
plt.xticks(rotation=45)
st.pyplot(fig1)

# Visualization 2: Victim Age Distribution
st.subheader("Victim Age Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(filtered_data['Victim_Age'], bins=30, kde=True, ax=ax2)
st.pyplot(fig2)

# Visualization 3: Crime by Month Occurred
st.subheader("Crime Count by Month Occurred")
monthly_crime = filtered_data['Month_occ'].value_counts().sort_index()
fig3, ax3 = plt.subplots()
sns.lineplot(x=monthly_crime.index, y=monthly_crime.values, ax=ax3, marker="o")
ax3.set_xlabel("Month")
ax3.set_ylabel("Crime Count")
st.pyplot(fig3)

# Visualization 4: Weapon Description
st.subheader("Weapon Usage Distribution")
weapon_counts = filtered_data['Weapon_Description'].value_counts().head(10)
fig4, ax4 = plt.subplots()
sns.barplot(x=weapon_counts.values, y=weapon_counts.index, ax=ax4)
st.pyplot(fig4)
