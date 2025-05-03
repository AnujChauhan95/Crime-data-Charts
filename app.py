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
    df['Date_Occurred'] = pd.to_datetime(df['Date_Occurred'], errors='coerce', infer_datetime_format=True)
    df['Date_Reported'] = pd.to_datetime(df['Date_Reported'], errors='coerce', infer_datetime_format=True)
    df['Day_occ'] = df['Date_Occurred'].dt.day
    df['Day_rep'] = df['Date_Reported'].dt.day
    df['Month_occ'] = df['Date_Occurred'].dt.month
    df['Month_rep'] = df['Date_Reported'].dt.month
    df['Year_occ'] = df['Date_Occurred'].dt.year
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
victim_sex_options = data['Victim_Sex'].dropna().unique()
weapon_options = data['Weapon_Description'].dropna().unique()
min_date, max_date = data['Date_Occurred'].min(), data['Date_Occurred'].max()

selected_crimes = st.sidebar.multiselect("Crime Categories", crime_categories, default=crime_categories)
selected_sex = st.sidebar.multiselect("Victim Sex", victim_sex_options, default=victim_sex_options)
selected_weapon = st.sidebar.multiselect("Weapon Used", weapon_options, default=weapon_options)
date_range = st.sidebar.date_input("Date Range", [min_date, max_date])

# Filter data
filtered_data = data[
    (data['Crime_Category'].isin(selected_crimes)) &
    (data['Victim_Sex'].isin(selected_sex)) &
    (data['Weapon_Description'].isin(selected_weapon)) &
    (data['Date_Occurred'] >= pd.to_datetime(date_range[0])) &
    (data['Date_Occurred'] <= pd.to_datetime(date_range[1]))
]

# Layout container
col1, col2 = st.columns(2)

# Crime Category Count
with col1:
    st.subheader("Crime Category Distribution")
    crime_counts = filtered_data['Crime_Category'].value_counts()
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    sns.barplot(x=crime_counts.index, y=crime_counts.values, ax=ax1)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

# Victim Age Distribution
with col2:
    st.subheader("Victim Age Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(filtered_data['Victim_Age'], bins=30, kde=True, ax=ax2)
    st.pyplot(fig2)

# Crime by Month and Year
st.subheader("Monthly Crime Trend")
monthly_trend = filtered_data.groupby(['Year_occ', 'Month_occ']).size().reset_index(name='Count')
monthly_trend['Month'] = pd.to_datetime(monthly_trend[['Year_occ', 'Month_occ']].assign(DAY=1))
fig3, ax3 = plt.subplots()
sns.lineplot(data=monthly_trend, x='Month', y='Count', marker="o", ax=ax3)
ax3.set_title("Crimes Over Time")
st.pyplot(fig3)

# Weapon Description
st.subheader("Top 10 Weapon Usage")
weapon_counts = filtered_data['Weapon_Description'].value_counts().head(10)
fig4, ax4 = plt.subplots()
sns.barplot(x=weapon_counts.values, y=weapon_counts.index, ax=ax4)
st.pyplot(fig4)

# Victim Sex Count
st.subheader("Victim Sex Distribution")
sex_counts = filtered_data['Victim_Sex'].value_counts()
fig5, ax5 = plt.subplots()
sns.barplot(x=sex_counts.index, y=sex_counts.values, ax=ax5)
st.pyplot(fig5)

# Correlation Heatmap
st.subheader("Correlation Heatmap (Numerical Columns)")
numeric_cols = filtered_data.select_dtypes(include=[np.number])
if not numeric_cols.empty:
    fig6, ax6 = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", ax=ax6)
    st.pyplot(fig6)
else:
    st.write("No numerical data to display heatmap.")
