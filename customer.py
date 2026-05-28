# Customer Segmentation Project
# Machine Learning using K-Means Clustering

# Install Required Libraries
# pip install pandas matplotlib seaborn scikit-learn streamlit plotly openpyxl

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Streamlit Page Settings
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

# Title
st.title("🧑‍🤝‍🧑 Customer Segmentation Dashboard")

# Upload File
uploaded_file = st.file_uploader(
    "Upload Customer Dataset (CSV or Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    # Read Dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df)

    # Selecting Features for Clustering
    features = ['Age', 'Annual Income', 'Spending Score']

    # Check Columns
    if all(col in df.columns for col in features):

        # Feature Data
        X = df[features]

        # Standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # KMeans Clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        df['Customer Segment'] = kmeans.fit_predict(X_scaled)

        # KPI Metrics
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Customers", len(df))
        col2.metric("Segments Created", df['Customer Segment'].nunique())
        col3.metric("Average Income", f"{df['Annual Income'].mean():,.2f}")

        # Segment Distribution
        st.subheader("📊 Customer Segment Distribution")

        segment_count = df['Customer Segment'].value_counts().reset_index()
        segment_count.columns = ['Segment', 'Customers']

        fig1 = px.pie(
            segment_count,
            names='Segment',
            values='Customers',
            title='Customer Segments'
        )

        st.plotly_chart(fig1, use_container_width=True)

        # Scatter Plot
        st.subheader("📈 Customer Segmentation Visualization")

        fig2 = px.scatter(
            df,
            x='Annual Income',
            y='Spending Score',
            color='Customer Segment',
            size='Age',
            hover_data=['Age'],
            title='Customer Segments based on Income & Spending'
        )

        st.plotly_chart(fig2, use_container_width=True)

        # Segment Analysis
        st.subheader("📋 Segment Analysis")

        analysis = df.groupby('Customer Segment')[features].mean()

        st.dataframe(analysis)

        # Download Segmented Data
        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="⬇ Download Segmented Dataset",
            data=csv,
            file_name='customer_segments.csv',
            mime='text/csv'
        )

    else:
        st.error("Dataset must contain: Age, Annual Income, Spending Score columns")

else:
    st.info("Please upload a CSV or Excel file.")
