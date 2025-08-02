import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import holidays
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

st.set_page_config(page_title="Bluesky + X Analytics Dashboard", layout="wide")

st.title("📊 Bluesky + X Analytics Dashboard")

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
    date_range = st.sidebar.date_input("Select date range", [min_date, max_date], min_value=min_date, max_value=max_date)

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
    # Aggregate follower gains and losses per day
    daily_stats = followers_df.groupby('date').agg({'gained': 'sum', 'lost': 'sum'})
    daily_stats['net'] = daily_stats['gained'] - daily_stats['lost']

    # Option to exclude outliers in follower gains/losses
    # Uses 95th percentile to cap extreme values when checkbox is selected
    if show_outliers:
        # Replace values beyond the 95th percentile with the percentile value
        for col in ['gained', 'lost', 'net']:
            threshold = daily_stats[col].quantile(0.95)
            daily_stats[col] = np.where(daily_stats[col] > threshold, threshold, daily_stats[col])

    # Toggle between Net vs Total metrics
    follower_view = st.radio(
        "Select follower metric to display",
        options=["Net Growth", "Gained vs Lost"],
        horizontal=True,
        key="follower_metric"
    )
    if follower_view == "Net Growth":
        st.line_chart(daily_stats[['net']])
    else:
        st.line_chart(daily_stats[['gained', 'lost']])

    # --- Engagement Over Time ---
    st.subheader("💬 Post Engagement Over Time")
    engagement_summary = engagement_df.groupby('date').agg({'likes': 'sum', 'reposts': 'sum', 'replies': 'sum'})
    # Optionally exclude outliers (viral posts) by capping values at the 95th percentile
    if show_outliers:
        for col in engagement_summary.columns:
            threshold = engagement_summary[col].quantile(0.95)
            engagement_summary[col] = np.where(engagement_summary[col] > threshold, threshold, engagement_summary[col])
    # Choose which metric(s) to display
    if engagement_type == 'Combined':
        st.area_chart(engagement_summary)
    else:
        metric = engagement_type.split()[0].lower()
        st.line_chart(engagement_summary[[metric]])

    # --- Hashtag Effectiveness ---
    st.subheader("🏷️ Top Hashtag Performance")
    # Split semicolon-separated hashtags into individual entries
    hashtag_summary = engagement_df.copy()
    hashtag_summary['hashtags'] = hashtag_summary['hashtags'].fillna('').apply(lambda x: x.split(';'))
    hashtag_summary = hashtag_summary.explode('hashtags')
    # Remove empty hashtags that result from trailing semicolons
    hashtag_summary = hashtag_summary[hashtag_summary['hashtags'] != '']
    # Compute total engagement for each hashtag using a weighted sum
    # Assign weights to reposts and replies relative to likes
    repost_weight = st.sidebar.slider("Repost weight", 1.0, 5.0, 2.0, step=0.5)
    reply_weight = st.sidebar.slider("Reply weight", 1.0, 5.0, 1.5, step=0.5)
    hashtag_summary['total_engagement'] = (
        hashtag_summary['likes'] + hashtag_summary['reposts'] * repost_weight + hashtag_summary['replies'] * reply_weight
    )
    hashtag_stats = hashtag_summary.groupby('hashtags').agg({
        'likes': 'sum',
        'reposts': 'sum',
        'replies': 'sum',
        'total_engagement': 'sum',
        'hashtags': 'count'
    }).rename(columns={'hashtags': 'count'})
    # Compute engagement per use (ROI metric)
    hashtag_stats['engagement_per_use'] = hashtag_stats['total_engagement'] / hashtag_stats['count']
    # Show top hashtags based on ROI
    top_hashtag_count = st.sidebar.slider("Number of top hashtags", 5, 20, 10)
    hashtag_stats = hashtag_stats.sort_values(by='engagement_per_use', ascending=False).head(top_hashtag_count)
    st.dataframe(hashtag_stats[['likes', 'reposts', 'replies', 'count', 'engagement_per_use']])

    # --- Predictive Hashtag Suggestions ---
    st.subheader("🔍 Predictive Hashtag Suggestions")
    top_hashtags = hashtag_stats.index.tolist()
    st.markdown("**Suggested High-ROI Hashtags for New Posts:**")
    if top_hashtags:
        st.write(", ".join([f"#{tag}" for tag in top_hashtags]))
    else:
        st.write("No hashtags found in the current date range.")

    # --- Automated Tag Clustering ---
    st.subheader("🧪 Automated Hashtag Clustering")
    # Fit a matrix on the full set of hashtags (drop NA and empty strings)
    nonempty_hashtags = engagement_df['hashtags'].dropna().loc[engagement_df['hashtags'].dropna() != '']
    if len(nonempty_hashtags) > 0:
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split(';'))
        matrix = vectorizer.fit_transform(nonempty_hashtags)
        # Limit number of components to the minimum of 3 and number of features to avoid errors
        n_components = min(3, matrix.shape[1] - 1) if matrix.shape[1] > 1 else 1
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
    else:
        st.info("No feed column found in the engagement data.")

    # --- Starter Pack Generator ---
    st.subheader("🧠 Starter Pack Generator")
    # Suggest a starter pack using the top 5 ROI hashtags (if available)
    suggested_pack = top_hashtags[:5]
    if suggested_pack:
        st.markdown("**Starter Pack (Top 5 Tags):**")
        st.code(" ".join([f"#{tag}" for tag in suggested_pack]))
    else:
        st.info("Not enough hashtags to generate a starter pack.")

    # --- Engagement Timing Heatmap ---
    st.subheader("🕒 Engagement Timing Heatmap")
    # Adjust timestamps by timezone offset and derive hour and weekday
    engagement_df['datetime'] = pd.to_datetime(engagement_df['date']) + pd.to_timedelta(time_zone_adjustment, unit='h')
    engagement_df['hour'] = engagement_df['datetime'].dt.hour
    engagement_df['day_of_week'] = engagement_df['datetime'].dt.day_name()
    # Summarize engagement by day of week and hour
    heatmap_data = engagement_df.groupby(['day_of_week', 'hour'])[['likes', 'reposts', 'replies']].sum().reset_index()
    # Choose which metric to display on the heatmap
    heatmap_metric = st.radio(
        "Select metric for heatmap",
        options=['likes', 'reposts', 'replies', 'combined'],
        horizontal=True,
        key="heatmap_metric"
    )
    if heatmap_metric == 'combined':
        heatmap_data['combined'] = heatmap_data['likes'] + heatmap_data['reposts'] + heatmap_data['replies']
        metric_to_plot = 'combined'
    else:
        metric_to_plot = heatmap_metric
    pivot_table = heatmap_data.pivot(index='day_of_week', columns='hour', values=metric_to_plot)
    st.dataframe(pivot_table.fillna(0))

    # --- Holiday Overlay Indicator ---
    st.subheader("🎉 Holiday Awareness")
    # Allow users to select the country whose public holidays will be highlighted
    country = st.selectbox("Select Country for Holiday Overlay", ['US', 'CN', 'GB', 'IN', 'CA'])
    # Fetch holiday dates for the selected country
    try:
        holiday_dates = holidays.country_holidays(country)
    except Exception:
        holiday_dates = []
    # Mark rows where the post date (converted to date string) coincides with a holiday
    engagement_df['is_holiday'] = engagement_df['datetime'].dt.date.astype(str).isin(holiday_dates)
    st.write("Total posts on holidays:", engagement_df['is_holiday'].sum())
    # If there are posts on holidays, display them for further analysis
    if engagement_df['is_holiday'].sum() > 0:
        with st.expander("Show posts on holidays"):
            st.dataframe(engagement_df[engagement_df['is_holiday']])

    # --- Unfollower Log ---
    if unfollower_csv:
        st.subheader("📉 Unfollower Log")
        st.dataframe(unfollower_df.sort_values(by='date', ascending=False))

    # --- CSV Export Section ---
    st.subheader("📤 Export Filtered Data")
    export_type = st.radio("Select export type", ['Follower Data', 'Engagement Data', 'Unfollower Log'])
    if export_type == 'Follower Data':
        st.download_button("Download Follower Data", followers_df.to_csv(index=False), "followers_filtered.csv", "text/csv")
    elif export_type == 'Engagement Data':
        st.download_button("Download Engagement Data", engagement_df.to_csv(index=False), "engagement_filtered.csv", "text/csv")
    elif export_type == 'Unfollower Log' and unfollower_csv:
        st.download_button("Download Unfollower Log", unfollower_df.to_csv(index=False), "unfollower_log_filtered.csv", "text/csv")

else:
    st.info("Please upload both a Follower CSV and an Engagement CSV to begin.")

# --- Notes on CSV Format ---
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Expected CSV Formats")
st.sidebar.markdown("""
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
""")
