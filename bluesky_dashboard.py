import streamlit as st
import time
import pandas as pd
import numpy as np
import requests
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