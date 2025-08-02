
import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
import os

st.set_page_config(page_title="Bluesky Analytics Dashboard", layout="wide")

st.title("📊 Bluesky Analytics Dashboard")

# Sidebar file upload
st.sidebar.header("📂 Upload Your Data")
follower_file = st.sidebar.file_uploader("Upload Follower CSV", type=["csv"])
engagement_file = st.sidebar.file_uploader("Upload Engagement CSV", type=["csv"])
unfollower_file = st.sidebar.file_uploader("Upload Unfollower Log CSV", type=["csv"])

# Load CSVs
if follower_file:
    followers_df = pd.read_csv(follower_file)
    followers_df['date'] = pd.to_datetime(followers_df['date'], errors='coerce')

    # Ensure valid dates
    valid_dates = followers_df['date'].dropna()
    if not valid_dates.empty:
        min_date_followers = valid_dates.min().date()
        max_date_followers = valid_dates.max().date()
        account_creation_default = min_date_followers
    else:
        min_date_followers = date(2024, 1, 1)
        max_date_followers = date.today()
        account_creation_default = min_date_followers

    # Sidebar date filter
    st.sidebar.subheader("📅 Date Range Filter")
    account_creation_date = st.sidebar.date_input(
        "Account creation date",
        value=account_creation_default,
        min_value=min_date_followers,
        max_value=max_date_followers
    )

    st.success("Follower CSV loaded successfully.")
else:
    st.info("Please upload a Follower CSV to begin analysis.")

# Additional dashboard features can go here
