import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Oorja Crime Data EDA Dashboard")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("train0.csv")
    df['Date_Occurred'] = pd.to_datetime(df['Date_Occurred'], errors='coerce')
    df['Date_Reported'] = pd.to_datetime(df['Date_Reported'], errors='coerce')
    df['Month_Occurred'] = df['Date_Occurred'].dt.to_period("M").astype(str)
    df['Year'] = df['Date_Occurred'].dt.year
    df['Month'] = df['Date_Occurred'].dt.month
    df['Day'] = df['Date_Occurred'].dt.day
    df = df[df['Victim_Age'] >= 0]
    df = df.drop_duplicates()
    df['Weapon_Description'].fillna("UNKNOWN", inplace=True)
    df['Victim_Sex'].fillna("X", inplace=True)
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("Filter Options")
categories = st.sidebar.multiselect("Crime Categories", df['Crime_Category'].dropna().unique(), default=df['Crime_Category'].dropna().unique())
weapons = st.sidebar.multiselect("Weapon Used", df['Weapon_Description'].dropna().unique(), default=df['Weapon_Description'].dropna().unique())
sexes = st.sidebar.multiselect("Victim Sex", df['Victim_Sex'].dropna().unique(), default=df['Victim_Sex'].dropna().unique())

min_date = df['Date_Occurred'].min()
max_date = df['Date_Occurred'].max()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date])

# Apply filters
filtered_df = df[
    (df['Crime_Category'].isin(categories)) &
    (df['Weapon_Description'].isin(weapons)) &
    (df['Victim_Sex'].isin(sexes)) &
    (df['Date_Occurred'] >= pd.to_datetime(date_range[0])) &
    (df['Date_Occurred'] <= pd.to_datetime(date_range[1]))
]

# Crime Category Distribution
st.subheader("Crime Category Distribution")
fig1, ax1 = plt.subplots(figsize=(10, 4))
sns.countplot(data=filtered_df, x='Crime_Category', order=filtered_df['Crime_Category'].value_counts().index, ax=ax1)
ax1.tick_params(axis='x', rotation=45)
st.pyplot(fig1)

# Victim Age Histogram
st.subheader("Victim Age Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(filtered_df['Victim_Age'], bins=30, kde=True, ax=ax2)
st.pyplot(fig2)

# Monthly Trend
st.subheader("Monthly Crime Trend")
monthly = filtered_df.groupby('Month_Occurred').size().reset_index(name='Crimes')
fig3, ax3 = plt.subplots()
sns.lineplot(data=monthly, x='Month_Occurred', y='Crimes', marker="o", ax=ax3)
ax3.tick_params(axis='x', rotation=45)
st.pyplot(fig3)

# Weapon Usage
st.subheader("Top 10 Weapons Used")
top_weapons = filtered_df['Weapon_Description'].value_counts().head(10)
fig4, ax4 = plt.subplots()
sns.barplot(y=top_weapons.index, x=top_weapons.values, ax=ax4)
st.pyplot(fig4)

# Victim Sex Distribution
st.subheader("Victim Sex Distribution")
fig5, ax5 = plt.subplots()
sns.countplot(data=filtered_df, x='Victim_Sex', ax=ax5)
st.pyplot(fig5)

st.success("Dashboard loaded with applied filters.")
