import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Nassau Candy Optimization Dashboard",
    layout="wide"
)

# -----------------------------
# TITLE
# -----------------------------

st.title("Factory Reallocation & Shipping Optimization System")

st.write(
    """
    This dashboard predicts shipping performance and
    recommends optimal factory assignments for
    Nassau Candy Distributor.
    """
)

# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv("data/Nassau Candy Distributor.csv")

# -----------------------------
# DATE CONVERSION
# -----------------------------

df['Order Date'] = pd.to_datetime(
    df['Order Date'],
    dayfirst=True
)

df['Ship Date'] = pd.to_datetime(
    df['Ship Date'],
    dayfirst=True
)

# -----------------------------
# CREATE LEAD TIME
# -----------------------------

df['Lead_Time'] = (
    df['Ship Date'] - df['Order Date']
).dt.days

# -----------------------------
# FACTORY MAPPING
# -----------------------------

factory_map = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Kazookles": "The Other Factory"
}

df['Factory'] = df['Product Name'].map(factory_map)

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------

st.sidebar.title("Filters")

product = st.sidebar.selectbox(
    "Select Product",
    df['Product Name'].unique()
)

region = st.sidebar.selectbox(
    "Select Region",
    df['Region'].unique()
)

ship_mode = st.sidebar.selectbox(
    "Select Ship Mode",
    df['Ship Mode'].unique()
)

# -----------------------------
# FILTER DATA
# -----------------------------

filtered_df = df[
    (df['Product Name'] == product) &
    (df['Region'] == region) &
    (df['Ship Mode'] == ship_mode)
]

# -----------------------------
# KPI CARDS
# -----------------------------

st.subheader("Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Sales",
    f"${df['Sales'].sum():,.0f}"
)

col2.metric(
    "Total Profit",
    f"${df['Gross Profit'].sum():,.0f}"
)

col3.metric(
    "Average Lead Time",
    round(df['Lead_Time'].mean(), 2)
)

col4.metric(
    "Total Orders",
    df.shape[0]
)

# -----------------------------
# DATASET PREVIEW
# -----------------------------

st.subheader("Dataset Preview")

st.dataframe(df.head())

# -----------------------------
# FILTERED DATA
# -----------------------------

st.subheader("Filtered Data")

st.dataframe(filtered_df)

# -----------------------------
# SALES CHART
# -----------------------------

st.subheader("Sales by Product")

sales_data = df.groupby(
    'Product Name'
)['Sales'].sum().reset_index()

fig1, ax1 = plt.subplots(figsize=(12,5))

sns.barplot(
    x='Product Name',
    y='Sales',
    data=sales_data,
    ax=ax1
)

plt.xticks(rotation=90)

st.pyplot(fig1)

# -----------------------------
# LEAD TIME CHART
# -----------------------------

st.subheader("Average Lead Time by Ship Mode")

fig2, ax2 = plt.subplots(figsize=(8,5))

sns.barplot(
    x='Ship Mode',
    y='Lead_Time',
    data=df,
    ax=ax2
)

plt.xticks(rotation=20)

st.pyplot(fig2)

# -----------------------------
# FACTORY DISTRIBUTION
# -----------------------------

st.subheader("Factory Order Distribution")

fig3, ax3 = plt.subplots(figsize=(8,5))

sns.countplot(
    x='Factory',
    data=df,
    ax=ax3
)

plt.xticks(rotation=20)

st.pyplot(fig3)

# -----------------------------
# RECOMMENDATION SECTION
# -----------------------------

st.subheader("Factory Recommendation")

if not filtered_df.empty:

    best_factory = filtered_df['Factory'].mode()[0]

    avg_time = round(
        filtered_df['Lead_Time'].mean(),
        2
    )

    st.success(
        f"""
        Recommended Factory: {best_factory}

        Expected Average Lead Time:
        {avg_time} days
        """
    )

else:

    st.warning(
        "No data available for selected filters."
    )

# -----------------------------
# PROJECT SUMMARY
# -----------------------------

st.subheader("Project Summary")

st.info(
    """
    This AI-powered system helps Nassau Candy Distributor
    optimize factory allocation, reduce shipping lead time,
    and improve operational efficiency using
    machine learning and data analytics.
    """
)