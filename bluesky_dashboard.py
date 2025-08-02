import streamlit as st
import pandas as pd
import numpy as np
import holidays
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

# NOTE: matplotlib is intentionally not imported here to avoid requiring
# additional dependencies on Streamlit Cloud. All charts are rendered
# using Streamlit's built‑in charting functions.

st.set_page_config(page_title="Bluesky Analytics Dashboard", layout="wide")

st.title("📊 Bluesky Analytics Dashboard")

# --- Upload CSV Files ---
st.sidebar.header("📤 Upload Your Data")
follower_csv = st.sidebar.file_uploader("Upload Follower CSV", type="csv")
engagement_csv = st.sidebar.file_uploader("Upload Engagement CSV", type="csv")
unfollower_csv = st.sidebar.file_uploader("Upload Unfollower Log CSV", type="csv")

# --- Load Data ---
@st.cache_data
def load_data(file):
    return pd.read_csv(file, parse_dates=True)

if follower_csv and engagement_csv:
    followers_df = load_data(follower_csv)
    engagement_df = load_data(engagement_csv)
    if unfollower_csv:
        unfollower_df = load_data(unfollower_csv)

    # --- Global Date Filter ---
    st.sidebar.header("📆 Date Range Filter")
    min_date = min(followers_df['date'].min(), engagement_df['date'].min())
    max_date = max(followers_df['date'].max(), engagement_df['date'].max())
    date_range = st.sidebar.date_input(
        "Select date range", [min_date, max_date], min_value=min_date, max_value=max_date
    )
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

    followers_df = followers_df[(followers_df['date'] >= start_date) & (followers_df['date'] <= end_date)]
    engagement_df = engagement_df[(engagement_df['date'] >= start_date) & (engagement_df['date'] <= end_date)]
    if unfollower_csv:
        unfollower_df = unfollower_df[(unfollower_df['date'] >= start_date) & (unfollower_df['date'] <= end_date)]

    # --- Toggle Options ---
    st.sidebar.header("⚙️ Chart Options")
    show_outliers = st.sidebar.checkbox("Exclude Outliers (Viral Posts)", value=True)
    engagement_type = st.sidebar.radio("Engagement Metric", ['Combined', 'Likes Only', 'Reposts Only', 'Replies Only'])
    time_zone_adjustment = st.sidebar.slider("Adjust Timezone (Hours)", -12, 12, 0)

    # --- Follower Growth and Churn Chart ---
    st.subheader("📈 Follower Growth vs. Churn")
    daily_stats = followers_df.groupby('date').agg({'gained': 'sum', 'lost': 'sum'})
    daily_stats['net'] = daily_stats['gained'] - daily_stats['lost']
    st.line_chart(daily_stats)

    # --- Engagement Over Time ---
    st.subheader("💬 Post Engagement Over Time")
    engagement_summary = engagement_df.groupby('date').agg({'likes': 'sum', 'reposts': 'sum', 'replies': 'sum'})
    if show_outliers:
        engagement_summary = engagement_summary[(engagement_summary < engagement_summary.quantile(0.95))]

    if engagement_type == 'Combined':
        st.area_chart(engagement_summary)
    else:
        metric = engagement_type.split()[0].lower()
        st.line_chart(engagement_summary[[metric]])

    # --- Hashtag Effectiveness ---
    st.subheader("🏷️ Top Hashtag Performance")
    hashtag_summary = engagement_df.copy()
    hashtag_summary['hashtags'] = hashtag_summary['hashtags'].fillna('').apply(lambda x: x.split(';'))
    hashtag_summary = hashtag_summary.explode('hashtags')
    hashtag_stats = (
        hashtag_summary.groupby('hashtags').agg({
            'likes': 'mean',
            'reposts': 'mean',
            'replies': 'mean',
            'hashtags': 'count'
        }).rename(columns={'hashtags': 'count'}).sort_values(by='likes', ascending=False).head(10)
    )
    st.dataframe(hashtag_stats)

    # --- Predictive Hashtag Suggestions ---
    st.subheader("🔍 Predictive Hashtag Suggestions")
    top_hashtags = hashtag_stats.index.tolist()
    st.markdown("**Suggested High-ROI Hashtags for New Posts:**")
    st.write(", ".join([f"#{tag}" for tag in top_hashtags]))

    # --- Automated Tag Clustering ---
    st.subheader("🧪 Automated Hashtag Clustering")
    if len(engagement_df['hashtags'].dropna()) >= 3:
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split(';'))
        matrix = vectorizer.fit_transform(engagement_df['hashtags'].dropna())
        n_components = min(3, matrix.shape[1])
        svd = TruncatedSVD(n_components=n_components)
        reduced = svd.fit_transform(matrix)
        cluster_df = pd.DataFrame(reduced, columns=[f'Cluster {i+1}' for i in range(n_components)])
        st.dataframe(cluster_df.head(10))
    else:
        st.info("Not enough hashtag data to perform clustering.")

    # --- Feed / Starter Pack Performance ---
    st.subheader("🧭 Feed and Starter Pack Comparison")
    if 'feed' in engagement_df.columns:
        feed_summary = engagement_df.groupby('feed').agg({
            'likes': 'mean',
            'reposts': 'mean',
            'replies': 'mean'
        })
        st.bar_chart(feed_summary)

    # --- Starter Pack Generator ---
    st.subheader("🧠 Starter Pack Generator")
    suggested_pack = top_hashtags[:5]
    st.markdown("**Starter Pack (Top 5 Tags):**")
    st.code(" ".join([f"#{tag}" for tag in suggested_pack]))

    # --- Engagement Timing Heatmap ---
    st.subheader("🕒 Engagement Timing Heatmap")
    engagement_df['datetime'] = pd.to_datetime(engagement_df['date']) + pd.to_timedelta(time_zone_adjustment, unit='h')
    engagement_df['hour'] = engagement_df['datetime'].dt.hour
    engagement_df['day_of_week'] = engagement_df['datetime'].dt.day_name()
    if engagement_type == 'Combined':
        metric_col = 'likes'
    else:
        metric_col = engagement_type.split()[0].lower()
    heatmap_data = engagement_df.groupby(['day_of_week', 'hour'])[[metric_col]].sum().reset_index()
    pivot_table = heatmap_data.pivot(index='day_of_week', columns='hour', values=metric_col)
    st.dataframe(pivot_table.fillna(0))

    # --- Holiday Overlay Indicator ---
    st.subheader("🎉 Holiday Awareness")
    country = st.selectbox("Select Country for Holiday Overlay", ['US', 'CN', 'GB', 'IN', 'CA'])
    holiday_dates = holidays.country_holidays(country)
    engagement_df['is_holiday'] = engagement_df['datetime'].dt.date.astype(str).isin(holiday_dates)
    st.write("Total posts on holidays:", int(engagement_df['is_holiday'].sum()))

    # --- Unfollower Log ---
    if unfollower_csv:
        st.subheader("📉 Unfollower Log")
        st.dataframe(unfollower_df.sort_values(by='date', ascending=False))

    # --- CSV Export Section ---
    st.subheader("📤 Export Filtered Data")
    export_type = st.radio("Select export type", ['Follower Data', 'Engagement Data', 'Unfollower Log'])
    if export_type == 'Follower Data':
        st.download_button(
            "Download Follower Data",
            followers_df.to_csv(index=False),
            "followers_filtered.csv",
            "text/csv"
        )
    elif export_type == 'Engagement Data':
        st.download_button(
            "Download Engagement Data",
            engagement_df.to_csv(index=False),
            "engagement_filtered.csv",
            "text/csv"
        )
    elif export_type == 'Unfollower Log' and unfollower_csv:
        st.download_button(
            "Download Unfollower Log",
            unfollower_df.to_csv(index=False),
            "unfollower_log_filtered.csv",
            "text/csv"
        )
else:
    st.info("Please upload both a Follower CSV and an Engagement CSV to begin.")

# --- Notes on CSV Format ---
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Expected CSV Formats")
st.sidebar.markdown(
    """
**Follower CSV**

- `date`: YYYY-MM-DD
- `gained`: Number of followers gained
- `lost`: Number of followers lost

**Engagement CSV**

- `date`: YYYY-MM-DD
- `likes`, `reposts`, `replies`: engagement counts
- `hashtags`: semicolon-separated list (e.g., `#ai;#streamlit`)
- `feed`: optional, used for tracking feed/starter pack

**Unfollower Log CSV**

- `date`: YYYY-MM-DD
- `username`: Handle of the user who unfollowed
    """
)