import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

st.set_page_config(layout="wide")

# Load data from SQL Server
@st.cache_data

def load_data():
    engine = create_engine("mssql+pyodbc://DESKTOP-ALEGQ73\\SQLEXPRESS/ML?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes")
    df = pd.read_sql("SELECT * FROM train", engine)
    df['Date_Occurred'] = pd.to_datetime(df['Date_Occurred'])
    df['Date_Reported'] = pd.to_datetime(df['Date_Reported'])
    df['Day_occ'] = df['Date_Occurred'].dt.day
    df['Day_rep'] = df['Date_Reported'].dt.day
    df['Month_occ'] = df['Date_Occurred'].dt.month
    df['Month_rep'] = df['Date_Reported'].dt.month
    df['Weapon_Description'].fillna("UNKNOWN WEAPON/OTHER WEAPON", inplace=True)
    df['Victim_Descent'].fillna(df['Victim_Descent'].mode()[0], inplace=True)
    df['Victim_Sex'].fillna(df['Victim_Sex'].mode()[0], inplace=True)
    df.drop_duplicates(inplace=True)
    df = df[df['Victim_Age'] >= 0]
    df['diff'] = (df['Date_Reported'] - df['Date_Occurred']).dt.days
    df['Age_Group'] = df['Victim_Age'].apply(classify_age_group)
    df['Time_Segments'] = df['Time_Occurred'].apply(classify_time_slot)
    return df

def classify_age_group(age):
    if 0 <= age <= 5:
        return 'Infants'
    elif 6 <= age <= 17:
        return 'Children'
    elif 18 <= age <= 30:
        return 'Young Adults'
    elif 31 <= age <= 60:
        return 'Middle Aged'
    elif 61 <= age <= 99:
        return 'Elderly'
    else:
        return 'Unknown'

def classify_time_slot(time):
    if 400 <= time <= 759:
        return 'Early Morning'
    elif 800 <= time <= 1159:
        return 'Morning'
    elif 1200 <= time <= 1559:
        return 'Afternoon'
    elif 1600 <= time <= 1959:
        return 'Evening'
    elif 2000 <= time <= 2359:
        return 'Night'
    elif 1 <= time <= 359:
        return 'Late Night'
    else:
        return 'Unknown'

# Load data
data = load_data()

# Sidebar filters
st.sidebar.header("Filter Options")
crime_filter = st.sidebar.multiselect("Select Crime Category", data['Crime_Category'].unique(), default=data['Crime_Category'].unique())
sex_filter = st.sidebar.multiselect("Select Victim Sex", data['Victim_Sex'].unique(), default=data['Victim_Sex'].unique())
descent_filter = st.sidebar.multiselect("Select Victim Descent", data['Victim_Descent'].unique(), default=data['Victim_Descent'].unique())
weapon_filter = st.sidebar.multiselect("Select Weapon Description", data['Weapon_Description'].unique(), default=data['Weapon_Description'].unique())

# Apply filters
data = data[data['Crime_Category'].isin(crime_filter)]
data = data[data['Victim_Sex'].isin(sex_filter)]
data = data[data['Victim_Descent'].isin(descent_filter)]
data = data[data['Weapon_Description'].isin(weapon_filter)]

# Main content
st.title("Oorja EDA Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Crime Category Distribution")
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    data['Crime_Category'].value_counts().plot.pie(autopct="%.2f", ax=ax1)
    ax1.axis("equal")
    st.pyplot(fig1)

with col2:
    st.subheader("Victim Sex Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(data['Victim_Sex'], ax=ax2)
    ax2.set_ylabel("No. of reportings")
    st.pyplot(fig2)

st.subheader("Age Group Distribution")
fig3, ax3 = plt.subplots()
sns.histplot(data['Age_Group'], ax=ax3)
ax3.set_ylabel("No. of reportings")
st.pyplot(fig3)

st.subheader("Victim Descent Distribution")
fig4, ax4 = plt.subplots()
sns.histplot(data['Victim_Descent'], ax=ax4)
ax4.set_ylabel("No. of reportings")
st.pyplot(fig4)

st.subheader("Crime Status Distribution")
fig5, ax5 = plt.subplots()
sns.histplot(data['Status_Description'], ax=ax5)
ax5.set_ylabel("No. of reportings")
st.pyplot(fig5)

st.subheader("Time Segment Distribution")
fig6, ax6 = plt.subplots()
sns.histplot(data['Time_Segments'], ax=ax6)
ax6.set_ylabel("No. of reportings")
st.pyplot(fig6)

st.subheader("Monthly Reporting Trend")
monthly_count = data.groupby(['Month_rep'])['Crime_Category'].count()
fig7, ax7 = plt.subplots()
sns.lineplot(x=monthly_count.index, y=monthly_count.values, ax=ax7)
ax7.set_xlabel("Month of Reporting")
ax7.set_ylabel("No. of reportings")
st.pyplot(fig7)

st.subheader("Top 5 Weapons Used")
fig8, ax8 = plt.subplots(figsize=(10, 4))
data['Weapon_Description'].value_counts().nlargest(5).plot(kind='bar', ax=ax8)
ax8.set_ylabel("No. of reportings")
st.pyplot(fig8)
