import streamlit as st
import time
import pandas as pd
import numpy as np
import requests
import holidays

import altair as alt

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

# Determine data source: CSV uploads or API fetch stored in session state
if (follower_csv and engagement_csv) or (
    'followers_df' in st.session_state and 'engagement_df' in st.session_state
):
    if follower_csv and engagement_csv:
        followers_df = load_data(follower_csv)
        engagement_df = load_data(engagement_csv)
        # Persist uploaded data to session state so it can be reused
        st.session_state['followers_df'] = followers_df
        st.session_state['engagement_df'] = engagement_df
        if unfollower_csv:
            unfollower_df = load_data(unfollower_csv)
            st.session_state['unfollower_df'] = unfollower_df
    else:
        # Load data from session state (populated by API fetch)
        followers_df = st.session_state['followers_df']
        engagement_df = st.session_state['engagement_df']
        unfollower_df = st.session_state.get('unfollower_df')

    # --- Account Creation & Global Date Filter ---
    # Allow the user to specify their account creation date.  This prevents
    # early data (e.g. from before the user joined Bluesky) from skewing
    # analytics.  The default is set to July 26 2025 (the date provided
    # by the user in our conversation).  All data prior to this date is
    # filtered out automatically.
    st.sidebar.header("📆 Date Range Filter")
    # Convert pandas Timestamps to Python dates for the date_input widget.
    account_creation_default = pd.to_datetime("2025-07-26").date()
    min_date_followers = pd.to_datetime(followers_df['date']).min().date()
    max_date_followers = pd.to_datetime(followers_df['date']).max().date()
    account_creation_date = st.sidebar.date_input(
        "Account creation date", value=account_creation_default, min_value=min_date_followers, max_value=max_date_followers
    )
    # Determine the available date range and let the user choose a subset
    min_date = max(pd.to_datetime(account_creation_date), min(pd.to_datetime(followers_df['date']).min(), pd.to_datetime(engagement_df['date']).min()))
    max_date = max(pd.to_datetime(followers_df['date']).max(), pd.to_datetime(engagement_df['date']).max())
    # Convert min/max to Python dates for the date range widget
    date_range = st.sidebar.date_input(
        "Select date range", [min_date.date(), max_date.date()], min_value=min_date.date(), max_value=max_date.date()
    )
    # Convert back to pandas timestamps for filtering
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    # Apply both the account creation date and the selected date range to the
    # followers, engagement and unfollower dataframes.
    start_filter = max(pd.to_datetime(account_creation_date), start_date)
    followers_df = followers_df[(followers_df['date'] >= start_filter) & (followers_df['date'] <= end_date)]
    engagement_df = engagement_df[(engagement_df['date'] >= start_filter) & (engagement_df['date'] <= end_date)]
    if unfollower_csv:
        unfollower_df = unfollower_df[(unfollower_df['date'] >= start_filter) & (unfollower_df['date'] <= end_date)]

    # --- Toggle Options ---
    st.sidebar.header("⚙️ Chart Options")
    show_outliers = st.sidebar.checkbox("Exclude Outliers (Viral Posts)", value=True)
    engagement_type = st.sidebar.radio("Engagement Metric", ['Combined', 'Likes Only', 'Reposts Only', 'Replies Only'])
    time_zone_adjustment = st.sidebar.slider("Adjust Timezone (Hours)", -12, 12, 0)

    # --- Follower Growth, Churn & Churn Rate ---
    st.subheader("📈 Follower Growth vs. Churn")
    daily_stats = followers_df.groupby('date').agg({'gained': 'sum', 'lost': 'sum'})
    daily_stats['net'] = daily_stats['gained'] - daily_stats['lost']
    # Compute overall churn rate for the filtered period: lost / (gained + lost)
    total_gained = daily_stats['gained'].sum()
    total_lost = daily_stats['lost'].sum()
    churn_rate = (total_lost / (total_gained + total_lost) * 100) if (total_gained + total_lost) > 0 else 0
    st.metric("Churn Rate (%)", f"{churn_rate:.2f}%", help="Percentage of followers lost relative to total follower change in the selected date range.")
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

    # Note: Automated Tag Clustering has been removed from the Bluesky dashboard.  The
    # clustering functionality is specific to the X dashboard and not relevant here.

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
    # Compute local datetime based on timezone adjustment
    engagement_df['datetime'] = pd.to_datetime(engagement_df['date']) + pd.to_timedelta(time_zone_adjustment, unit='h')
    engagement_df['hour'] = engagement_df['datetime'].dt.hour
    engagement_df['day_of_week'] = engagement_df['datetime'].dt.day_name()
    # Let the user assign weights for reposts and replies when using the
    # combined metric.  Likes always have a weight of 1.  These controls
    # appear in the sidebar for clarity.
    st.sidebar.markdown("### 📌 Combined Metric Weights")
    repost_weight = st.sidebar.slider("Repost weight", 0.0, 3.0, 1.0, 0.1)
    reply_weight = st.sidebar.slider("Reply weight", 0.0, 3.0, 1.0, 0.1)
    # Compute the metric column based on user selection.  For combined
    # metrics, apply the weights; for single metrics, pick the relevant
    # column directly.  We calculate the 'combined_metric' column on
    # demand to simplify downstream aggregation.
    if engagement_type == 'Combined':
        engagement_df['combined_metric'] = (
            engagement_df['likes'] + engagement_df['reposts'] * repost_weight + engagement_df['replies'] * reply_weight
        )
        metric_col = 'combined_metric'
    elif engagement_type == 'Likes Only':
        metric_col = 'likes'
    elif engagement_type == 'Reposts Only':
        metric_col = 'reposts'
    else:
        metric_col = 'replies'
    # Aggregate engagement by day of week and hour
    heatmap_data = engagement_df.groupby(['day_of_week', 'hour'])[metric_col].sum().reset_index()
    pivot_table = heatmap_data.pivot(index='day_of_week', columns='hour', values=metric_col).fillna(0)
    # Optionally cap values at the 95th percentile to prevent viral posts from skewing
    if show_outliers:
        cap_value = pivot_table.values.flatten()
        cap_threshold = np.nanpercentile(cap_value, 95) if len(cap_value) > 0 else 0
        pivot_table = pivot_table.applymap(lambda x: min(x, cap_threshold))
    # Ensure days are ordered Monday→Sunday for readability
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_table = pivot_table.reindex(day_order)
    # Build a heatmap using Altair.  Convert the pivot_table into a long
    # dataframe with columns day_of_week, hour and value.  Altair will
    # automatically apply a sequential colour scale (YlOrRd) for
    # quantitative values.  Tooltips show the engagement value.
    heatmap_long = pivot_table.reset_index().melt(id_vars='day_of_week', var_name='hour', value_name='value')
    heatmap_long['hour'] = heatmap_long['hour'].astype(str)
    heatmap_chart = (
        alt.Chart(heatmap_long)
        .mark_rect()
        .encode(
            x=alt.X('hour:O', title='Hour of Day'),
            y=alt.Y('day_of_week:O', title='Day of Week', sort=day_order),
            color=alt.Color('value:Q', title='Engagement', scale=alt.Scale(scheme='redyellowblue')),
            tooltip=[alt.Tooltip('day_of_week:O', title='Day'), alt.Tooltip('hour:O', title='Hour'), alt.Tooltip('value:Q', title='Engagement')]
        )
        .properties(height=400)
    )
    st.altair_chart(heatmap_chart, use_container_width=True)
    # Show a summary table of total engagement by day of week beneath the heatmap
    day_totals = heatmap_data.groupby('day_of_week')[metric_col].sum().reindex(day_order).reset_index()
    day_totals = day_totals.sort_values(by=metric_col, ascending=False).rename(columns={'day_of_week': 'Day', metric_col: 'Total Engagement'})
    st.dataframe(day_totals)
    # --- Holiday Awareness ---
    st.subheader("🎉 Holiday Awareness")
    country = st.selectbox("Select Country for Holiday Overlay", ['US', 'CN', 'GB', 'IN', 'CA'])
    holiday_dates = holidays.country_holidays(country)
    engagement_df['is_holiday'] = engagement_df['datetime'].dt.date.astype(str).isin(holiday_dates)
    holiday_post_count = int(engagement_df['is_holiday'].sum())
    st.write("Total posts on holidays:", holiday_post_count)
    # Provide an expandable table listing posts that occurred on holidays
    if holiday_post_count > 0:
        with st.expander("Show posts on holidays"):
            st.dataframe(engagement_df[engagement_df['is_holiday']][['date', 'likes', 'reposts', 'replies', 'hashtags']])

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
    st.info(
        "Please upload both a Follower CSV and an Engagement CSV, or use the API fetch above to load data."
    )

# ------------------------ API INTEGRATION ------------------------

# Helper functions for Bluesky API
def _send_request(method: str, url: str, *, params: dict | None = None, headers: dict | None = None, json: dict | None = None) -> requests.Response:
    """
    Internal helper to issue a HTTP request and automatically respect Bluesky's
    rate limiting. If a request returns a 429 status (Too Many Requests), this
    function will pause until the reset time indicated by the `RateLimit-Reset`
    header before retrying. A small delay is also inserted between successful
    requests to avoid hammering the API unnecessarily.

    Args:
        method: HTTP method, e.g. 'get' or 'post'.
        url: Endpoint URL to call.
        params: Optional query parameters.
        headers: Optional HTTP headers.
        json: Optional JSON body for POST requests.

    Returns:
        A requests.Response object for a successful (non-429) request.
    """
    while True:
        resp = requests.request(method, url, params=params, headers=headers, json=json, timeout=30)
        if resp.status_code == 429:
            # Rate limited. Determine how long to wait. Bluesky typically
            # returns a RateLimit-Reset header with a UNIX epoch seconds
            # specifying when new requests are permitted. If not present,
            # default to a 60‑second backoff.
            reset = resp.headers.get("RateLimit-Reset")
            try:
                wait_seconds = int(reset) - int(time.time()) if reset else 60
            except Exception:
                wait_seconds = 60
            # Ensure we wait at least 1 second
            if wait_seconds < 1:
                wait_seconds = 1
            # Inform the caller via the Streamlit status area when possible
            try:
                st.info(f"Rate limited. Waiting {wait_seconds} seconds before retrying…")
            except Exception:
                pass
            time.sleep(wait_seconds)
            continue
        # Insert a brief delay to avoid hitting per‑IP limits (3000/5min ~ 10/s)
        time.sleep(0.2)
        return resp
def resolve_handle_api(handle: str) -> str:
    """Resolve a handle to its DID using the ATProto identity API.

    This helper respects rate limits by delegating to `_send_request`. Handles
    should be cached by callers to avoid unnecessary API traffic.
    """
    url = "https://bsky.social/xrpc/com.atproto.identity.resolveHandle"
    resp = _send_request("get", url, params={"handle": handle})
    data = resp.json()
    return data.get("did")


def create_session_api(identifier: str, password: str) -> tuple[str, str]:
    """Create a session and return both the access JWT and the DID.

    A session is created using the provided handle and app password. If the
    response indicates rate limiting, this function will wait and retry.
    """
    url = "https://bsky.social/xrpc/com.atproto.server.createSession"
    resp = _send_request("post", url, json={"identifier": identifier, "password": password})
    data = resp.json()
    return data.get("accessJwt"), data.get("did")


def fetch_followers_api(did: str, headers: dict) -> list:
    """Fetch all followers for the given DID using paginated requests.

    The Bluesky API caps the maximum `limit` at 100, so we request followers
    in batches of 100 and follow the pagination cursor until no more pages
    remain.

    Args:
        did: The decentralized identifier (DID) of the actor to fetch followers for.
        headers: A dictionary containing the Authorization header with a valid JWT.

    Returns:
        A list of follower DIDs.
    """
    followers = []
    cursor = None
    while True:
        params = {"actor": did, "limit": 100}
        if cursor:
            params["cursor"] = cursor
        url = "https://bsky.social/xrpc/app.bsky.graph.getFollowers"
        resp = _send_request("get", url, params=params, headers=headers)
        data = resp.json()
        # Each entry in the "followers" list contains the DID of the follower
        followers.extend([entry.get("did") for entry in data.get("followers", [])])
        cursor = data.get("cursor")
        if not cursor:
            break
    return followers


def fetch_posts_api(did: str, headers: dict) -> pd.DataFrame:
    """Fetch posts for the given actor and return a DataFrame."""
    rows = []
    cursor = None
    while True:
        params = {"actor": did, "limit": 100}
        if cursor:
            params["cursor"] = cursor
        url = "https://bsky.social/xrpc/app.bsky.feed.getAuthorFeed"
        resp = _send_request("get", url, params=params, headers=headers)
        feed_data = resp.json()
        for item in feed_data.get("feed", []):
            post = item.get("post", {})
            record = post.get("record", {})
            created_at = record.get("createdAt") or record.get("timestamp")
            if not created_at:
                continue
            date = created_at[:10]
            likes = post.get("likeCount", 0)
            reposts = post.get("repostCount", 0)
            replies = post.get("replyCount", 0)
            text = record.get("text", "")
            hashtags = [t for t in text.split() if t.startswith("#")]
            hashtag_str = ";".join(hashtags)
            rows.append((date, likes, reposts, replies, hashtag_str))
        cursor = feed_data.get("cursor")
        if not cursor:
            break
    df = pd.DataFrame(rows, columns=["date", "likes", "reposts", "replies", "hashtags"])
    return df


st.sidebar.markdown("---")
st.sidebar.header("🔗 Fetch Live Data from Bluesky")
bs_handle = st.sidebar.text_input("Bluesky handle (e.g. yourname.bsky.social)")
bs_password = st.sidebar.text_input("App password", type="password")
if st.sidebar.button("Fetch Latest Data"):
    if bs_handle and bs_password:
        with st.spinner("Connecting to Bluesky API…"):
            try:
                # Cache handle→DID lookups to avoid unnecessary resolveHandle
                handle_cache = st.session_state.setdefault('handle_cache', {})
                did = handle_cache.get(bs_handle)
                if not did:
                    did = resolve_handle_api(bs_handle)
                    handle_cache[bs_handle] = did
                jwt, session_did = create_session_api(bs_handle, bs_password)
                # Use the DID returned by the session if available.  Some handles
                # resolve to a different DID once authenticated.
                actor_did = session_did or did
                headers = {"Authorization": f"Bearer {jwt}"}
                followers_list = fetch_followers_api(actor_did, headers)
                posts_df = fetch_posts_api(actor_did, headers)
                # Convert date column on posts to datetime for proper filtering/aggregation
                posts_df['date'] = pd.to_datetime(posts_df['date'])
                # Compute follower gains/losses relative to previous fetch stored in session state
                prev_followers = st.session_state.get("_prev_followers", [])
                gained = len(set(followers_list) - set(prev_followers))
                lost = len(set(prev_followers) - set(followers_list))
                today = pd.Timestamp.today().normalize()
                # Save for next time
                st.session_state["_prev_followers"] = followers_list
                # Create follower DataFrame and store in session state
                followers_df = pd.DataFrame({
                    "date": [today],
                    "gained": [gained],
                    "lost": [lost]
                })
                # Ensure followers_df date column is datetime
                followers_df['date'] = pd.to_datetime(followers_df['date'])
                st.session_state['followers_df'] = followers_df
                # Store engagement DataFrame in session state
                st.session_state['engagement_df'] = posts_df
                # Clear any existing unfollower log
                if 'unfollower_df' in st.session_state:
                    del st.session_state['unfollower_df']
                # Display success message
                st.success("Data fetched successfully. Scroll up to see updated charts.")
                # Trigger a rerun to refresh the app if the Streamlit version supports it
                if hasattr(st, 'rerun'):
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to fetch data: {e}")
    else:
        st.warning("Please provide both a handle and an app password.")

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