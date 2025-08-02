
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(layout="wide", page_title="Bluesky Analytics Dashboard")

st.title("📘 Bluesky Analytics Dashboard")

# Upload CSV files
followers_file = st.sidebar.file_uploader("Upload Follower CSV", type="csv")
engagement_file = st.sidebar.file_uploader("Upload Engagement CSV", type="csv")
unfollowers_file = st.sidebar.file_uploader("Upload Unfollower Log CSV", type="csv")

# Load dataframes
followers_df = pd.read_csv(followers_file) if followers_file else pd.DataFrame()
engagement_df = pd.read_csv(engagement_file) if engagement_file else pd.DataFrame()
unfollowers_df = pd.read_csv(unfollowers_file) if unfollowers_file else pd.DataFrame()

# Convert 'date' columns to datetime with error coercion
for df in [followers_df, engagement_df, unfollowers_df]:
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Account creation date picker
if not followers_df.empty and 'date' in followers_df.columns:
    account_creation_default = followers_df['date'].min()
    account_creation_date = st.sidebar.date_input(
        "Account creation date",
        value=account_creation_default,
        min_value=account_creation_default,
        max_value=followers_df['date'].max()
    )
else:
    account_creation_date = None

# Placeholder logic for engagement heatmap and charts
if not engagement_df.empty:
    engagement_df['date'] = pd.to_datetime(engagement_df['date'], errors='coerce')
    daily_engagement = engagement_df.groupby(engagement_df['date'].dt.date).sum(numeric_only=True)
    st.subheader("📈 Post Engagement Over Time")
    st.area_chart(daily_engagement)
else:
    st.info("Upload an Engagement CSV to view post engagement charts.")
