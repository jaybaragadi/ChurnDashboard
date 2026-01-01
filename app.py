import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from io import BytesIO

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Customer Churn EDA Dashboard",
    page_icon="",
    layout="wide"
)

# ---------------------------
# Helpers
# ---------------------------
def pick_first_existing(df, candidates):
    """Return the first column name that exists in df from a list of candidates, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def safe_numeric(series):
    """Convert to numeric safely."""
    return pd.to_numeric(series, errors="coerce")

def churn_to_binary(s):
    """
    Convert churn column to binary 0/1.
    Supports numeric 0/1 and common string labels (Yes/No, Churned/Active, etc.).
    Returns nullable Int64 to preserve missing values.
    """
    if s is None:
        return None

    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float).astype("Int64")

    s2 = s.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1, "y": 1, "true": 1, "1": 1, "churn": 1, "churned": 1,
        "no": 0, "n": 0, "false": 0, "0": 0, "not churn": 0, "active": 0, "stayed": 0
    }

    result = s2.map(mapping)
    unmapped = result.isna()
    if unmapped.any():
        numeric_vals = pd.to_numeric(s2[unmapped], errors="coerce")
        result[unmapped] = numeric_vals

    return result.astype("Int64")

def pct(x):
    return f"{x*100:.1f}%"

@st.cache_data(show_spinner=False)
def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(file_bytes))

# ---------------------------
# Header
# ---------------------------
st.title("Customer Churn EDA Dashboard")
st.caption(
    "Single-file-first: upload dataset.csv to run the dashboard. "
    "Optional: upload an incoming/unlabeled file to compare distributions. "
    "You can also create an ML-style train/test split from a single dataset."
)

# ---------------------------
# Upload (single-file-first)
# ---------------------------
with st.expander("Upload data files", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        dataset_upload = st.file_uploader("Upload dataset.csv (required)", type=["csv"], key="dataset")
    with c2:
        incoming_upload = st.file_uploader(
            "Upload incoming.csv (optional, unlabeled/incoming customers)",
            type=["csv"],
            key="incoming"
        )

# ---------------------------
# Local mode support (optional)
# ---------------------------
def local_mode_available():
    if os.path.exists("data/dataset.csv"):
        return "single"
    if os.path.exists("data/train.csv") and os.path.exists("data/test.csv"):
        return "train_test"
    return None

def load_single_from_path():
    df_main = pd.read_csv("data/dataset.csv")
    df_main["__split__"] = "main"
    df_all = df_main.copy()
    df_incoming = pd.DataFrame()
    return df_all, df_main, df_incoming

def load_train_test_from_path():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    train["__split__"] = "train"
    test["__split__"] = "test"
    df_all = pd.concat([train, test], ignore_index=True)
    return df_all, train, test

def load_single_or_incoming_from_upload(dataset_upload, incoming_upload=None):
    df_main = read_csv_bytes(dataset_upload.getvalue())
    df_main["__split__"] = "main"

    if incoming_upload is not None:
        df_incoming = read_csv_bytes(incoming_upload.getvalue())
        df_incoming["__split__"] = "incoming"
        df_all = pd.concat([df_main, df_incoming], ignore_index=True)
    else:
        df_incoming = pd.DataFrame()
        df_all = df_main.copy()

    return df_all, df_main, df_incoming

# ---------------------------
# Load data (no tracebacks)
# ---------------------------
df_all = df_main = df_incoming = None

try:
    if dataset_upload is not None:
        df_all, df_main, df_incoming = load_single_or_incoming_from_upload(dataset_upload, incoming_upload)
        st.success("Dataset loaded successfully.")
    else:
        mode = local_mode_available()
        if mode == "single":
            df_all, df_main, df_incoming = load_single_from_path()
            st.info("Loaded local data from data/dataset.csv.")
        elif mode == "train_test":
            df_all, df_train, df_test = load_train_test_from_path()
            df_main = df_train
            df_incoming = df_test
            st.info("Loaded local data from data/train.csv and data/test.csv (legacy mode).")
        else:
            st.info("No dataset found. Please upload dataset.csv to continue.")
            st.stop()
except Exception:
    st.warning("Unable to read the CSV file(s). Please upload valid CSV files.")
    st.stop()

# ---------------------------
# Optional ML-ready split (single dataset)
# ---------------------------
# Only show if we currently have a single split (typically "main") and no incoming file.
if "__split__" in df_all.columns and df_all["__split__"].nunique() == 1:
    st.subheader("Optional ML split (train/test)")
    ml_split = st.checkbox("Create ML-style train/test split from dataset.csv", value=False)

    if ml_split:
        split_ratio = st.slider("Train size (%)", min_value=50, max_value=95, value=80, step=5)
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)

        df_tmp = df_all.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)
        n_train = int(len(df_tmp) * (split_ratio / 100.0))

        df_tmp["__split__"] = "test"
        df_tmp.loc[: n_train - 1, "__split__"] = "train"

        df_all = df_tmp
        st.info("Split created. Use the 'Dataset split' filter in the sidebar.")

# ---------------------------
# Column detection (robust)
# ---------------------------
churn_col = pick_first_existing(df_all, ["Churn", "churn", "Exited", "exit", "Target", "target", "label", "Label"])

subscription_col = pick_first_existing(df_all, ["SubscriptionType", "Plan", "Contract", "subscription_type", "contract_type"])
payment_col = pick_first_existing(df_all, ["PaymentMethod", "payment_method", "Payment", "payment"])

tenure_col = pick_first_existing(df_all, ["tenure", "Tenure", "AccountAge", "account_age", "MonthsAsCustomer", "months_as_customer"])

usage_cols_candidates = [
    "ViewingHoursPerWeek", "viewing_hours_per_week",
    "AverageViewingDuration", "avg_viewing_duration",
    "ContentDownloadsPerMonth", "downloads_per_month",
    "WatchlistSize", "watchlist_size",
    "MonthlyCharges", "monthly_charges",
    "TotalCharges", "total_charges",
    "SupportTicketsPerMonth", "support_tickets_per_month"
]
usage_cols = [c for c in usage_cols_candidates if c in df_all.columns]

# ---------------------------
# Prepare analysis columns
# ---------------------------
df = df_all.copy()

if churn_col is not None:
    df["__churn__"] = churn_to_binary(df[churn_col])
else:
    df["__churn__"] = pd.NA

if tenure_col is not None:
    df[tenure_col] = safe_numeric(df[tenure_col])

for c in usage_cols:
    df[c] = safe_numeric(df[c])

# ---------------------------
# Banner: churn charts need labeled rows
# ---------------------------
labeled_rows = df["__churn__"].notna().sum()
total_rows = len(df)

if churn_col is None or labeled_rows == 0:
    st.warning(
        "Churn label column was not found (or contains no usable labels). "
        "Churn-based charts will be unavailable. Upload a dataset with a churn/target column "
        "(for example: Churn, Target, Label, Exited)."
    )
else:
    if labeled_rows < total_rows:
        st.info(
            f"Churn-based charts use labeled rows only: {labeled_rows:,} / {total_rows:,} rows have churn labels. "
            "If you uploaded an 'incoming' file, it is usually unlabeled and will be excluded from churn charts."
        )

# ---------------------------
# Sidebar filters
# ---------------------------
st.sidebar.header("Filters")

unique_splits = sorted(df["__split__"].unique().tolist())
default_splits = ["train"] if "train" in unique_splits else [unique_splits[0]]

split_filter = st.sidebar.multiselect(
    "Dataset split",
    options=unique_splits,
    default=default_splits
)

filtered = df[df["__split__"].isin(split_filter)].copy()
filtered_with_churn = filtered.dropna(subset=["__churn__"]).copy()

if subscription_col:
    subs_options = sorted(filtered[subscription_col].dropna().astype(str).unique().tolist())
    if subs_options:
        subs_default = subs_options[: min(8, len(subs_options))]
        subs_selected = st.sidebar.multiselect("Subscription / Contract", subs_options, default=subs_default)
        filtered = filtered[filtered[subscription_col].astype(str).isin(subs_selected)]
        filtered_with_churn = filtered_with_churn[filtered_with_churn[subscription_col].astype(str).isin(subs_selected)]

if payment_col:
    pay_options = sorted(filtered[payment_col].dropna().astype(str).unique().tolist())
    if pay_options:
        pay_default = pay_options[: min(8, len(pay_options))]
        pay_selected = st.sidebar.multiselect("Payment method", pay_options, default=pay_default)
        filtered = filtered[filtered[payment_col].astype(str).isin(pay_selected)]
        filtered_with_churn = filtered_with_churn[filtered_with_churn[payment_col].astype(str).isin(pay_selected)]

if tenure_col:
    if filtered[tenure_col].notna().any():
        tmin = float(np.nanmin(filtered[tenure_col].values))
        tmax = float(np.nanmax(filtered[tenure_col].values))
        if tmax > tmin:
            tenure_range = st.sidebar.slider(
                "Tenure range",
                min_value=tmin,
                max_value=tmax,
                value=(tmin, tmax)
            )
            filtered = filtered[(filtered[tenure_col] >= tenure_range[0]) & (filtered[tenure_col] <= tenure_range[1])]
            filtered_with_churn = filtered_with_churn[
                (filtered_with_churn[tenure_col] >= tenure_range[0]) & (filtered_with_churn[tenure_col] <= tenure_range[1])
            ]

show_raw = st.sidebar.checkbox("Show raw data preview", value=False)

# ---------------------------
# KPIs
# ---------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

total_rows_f = len(filtered)
split_counts = filtered["__split__"].value_counts().to_dict()

kpi1.metric("Rows (filtered)", f"{total_rows_f:,}")
kpi2.metric("Main / Train rows", f"{split_counts.get('main', 0) + split_counts.get('train', 0):,}")
kpi3.metric("Incoming / Test rows", f"{split_counts.get('incoming', 0) + split_counts.get('test', 0):,}")

if churn_col and len(filtered_with_churn) > 0:
    churn_rate = float(filtered_with_churn["__churn__"].mean())
    kpi4.metric("Churn rate (labeled rows)", pct(churn_rate))
else:
    kpi4.metric("Churn rate", "N/A")

st.divider()

# ---------------------------
# Dataset overview
# ---------------------------
st.subheader("Dataset overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total columns", len(filtered.columns))
    st.metric("Numeric features", len(filtered.select_dtypes(include=[np.number]).columns))
    st.metric("Categorical features", len(filtered.select_dtypes(include=["object"]).columns))

with col2:
    st.write("Key columns detected")
    detected = []
    if churn_col:
        detected.append(f"Churn: {churn_col}")
    if subscription_col:
        detected.append(f"Subscription: {subscription_col}")
    if payment_col:
        detected.append(f"Payment: {payment_col}")
    if tenure_col:
        detected.append(f"Tenure: {tenure_col}")
    if detected:
        for item in detected:
            st.write(f"- {item}")
    else:
        st.write("- None")

with col3:
    st.write("Usage metrics detected")
    if usage_cols:
        for c in usage_cols[:8]:
            st.write(f"- {c}")
        if len(usage_cols) > 8:
            st.caption(f"Additional metrics: {len(usage_cols) - 8}")
    else:
        st.write("- None")

# ---------------------------
# Missing values
# ---------------------------
st.subheader("Missing values (top 15)")
missing = (filtered.isna().mean().sort_values(ascending=False) * 100).head(15)
miss_df = missing.reset_index()
miss_df.columns = ["column", "missing_pct"]

if len(miss_df) > 0 and miss_df["missing_pct"].max() > 0:
    fig_miss = px.bar(
        miss_df,
        x="missing_pct",
        y="column",
        orientation="h",
        labels={"missing_pct": "Missing (%)", "column": "Column"}
    )
    fig_miss.update_layout(height=420)
    st.plotly_chart(fig_miss, use_container_width=True)

    high_missing = miss_df[miss_df["missing_pct"] > 20]
    if len(high_missing) > 0:
        st.warning(f"{len(high_missing)} columns have more than 20% missing values.")
else:
    st.info("No meaningful missingness detected in the filtered data.")

st.divider()

# ---------------------------
# Core insights
# ---------------------------
st.subheader("Churn drivers")

c1, c2 = st.columns(2)

with c1:
    st.write("Churn by subscription / contract")
    if churn_col is None or subscription_col is None or len(filtered_with_churn) == 0:
        st.info("Churn and/or subscription column not available for labeled rows.")
    else:
        grp = (
            filtered_with_churn
            .groupby(subscription_col)["__churn__"]
            .agg(churn_rate="mean", customers="count")
            .reset_index()
            .sort_values("churn_rate", ascending=False)
        )
        fig = px.bar(grp, x=subscription_col, y="churn_rate", hover_data=["customers"])
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

with c2:
    st.write("Churn by payment method")
    if churn_col is None or payment_col is None or len(filtered_with_churn) == 0:
        st.info("Churn and/or payment column not available for labeled rows.")
    else:
        grp = (
            filtered_with_churn
            .groupby(payment_col)["__churn__"]
            .agg(churn_rate="mean", customers="count")
            .reset_index()
            .sort_values("churn_rate", ascending=False)
        )
        fig = px.bar(grp, x=payment_col, y="churn_rate", hover_data=["customers"])
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

c3, c4 = st.columns(2)

with c3:
    st.write("Tenure vs churn")
    if churn_col is None or tenure_col is None or len(filtered_with_churn) == 0:
        st.info("Churn and/or tenure column not available for labeled rows.")
    else:
        tmp = filtered_with_churn.dropna(subset=[tenure_col]).copy()
        if len(tmp) == 0:
            st.info("No valid tenure values after filtering.")
        else:
            bins = 10
            tmp["tenure_bin"] = pd.cut(tmp[tenure_col], bins=bins, duplicates="drop")
            grp = tmp.groupby("tenure_bin")["__churn__"].mean().reset_index()
            grp["tenure_mid"] = grp["tenure_bin"].apply(lambda x: (x.left + x.right) / 2)

            fig = px.line(grp, x="tenure_mid", y="__churn__", markers=True)
            fig.update_yaxes(tickformat=".0%")
            fig.update_xaxes(title="Tenure (binned midpoint)")
            fig.update_yaxes(title="Churn rate")
            st.plotly_chart(fig, use_container_width=True)

with c4:
    st.write("Usage vs churn")
    if churn_col is None or len(filtered_with_churn) == 0 or len(usage_cols) == 0:
        st.info("Churn and/or usage columns not available for labeled rows.")
    else:
        metric = st.selectbox("Usage metric", usage_cols)
        tmp = filtered_with_churn.dropna(subset=[metric]).copy()
        if len(tmp) == 0:
            st.info("No valid values for the selected metric after filtering.")
        else:
            tmp["ChurnLabel"] = np.where(tmp["__churn__"] == 1, "Churned", "Not churned")
            fig = px.box(tmp, x="ChurnLabel", y=metric, points="outliers")
            st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------
# Additional EDA
# ---------------------------
st.subheader("Additional EDA")

num_cols = filtered_with_churn.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in ["__churn__"] and not c.startswith("__")]

a, b = st.columns(2)

with a:
    st.write("Numeric distributions")
    if len(num_cols) == 0:
        st.info("No numeric columns available for distribution plots (labeled rows).")
    else:
        col = st.selectbox("Numeric column", num_cols, key="dist_col")
        tmp = filtered_with_churn.dropna(subset=[col]).copy()
        tmp["ChurnLabel"] = np.where(tmp["__churn__"] == 1, "Churned", "Not churned")
        fig = px.histogram(tmp, x=col, color="ChurnLabel", barmode="overlay", nbins=40)
        st.plotly_chart(fig, use_container_width=True)

with b:
    st.write("Correlation with churn (numeric)")
    if len(num_cols) == 0 or len(filtered_with_churn) == 0:
        st.info("Not enough labeled numeric data to compute correlation.")
    else:
        corr_df = filtered_with_churn[["__churn__"] + num_cols].corr(numeric_only=True)
        churn_corr = (
            corr_df["__churn__"]
            .drop("__churn__", errors="ignore")
            .sort_values(key=lambda s: s.abs(), ascending=False)
            .head(15)
        )
        cc = churn_corr.reset_index()
        cc.columns = ["feature", "corr_with_churn"]
        fig = px.bar(cc, x="corr_with_churn", y="feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Recommendations
# ---------------------------
st.subheader("Recommendations")

if churn_col and len(filtered_with_churn) > 0:
    insights = []
    recommendations = []

    if subscription_col:
        grp = filtered_with_churn.groupby(subscription_col)["__churn__"].mean().sort_values(ascending=False)
        if len(grp) > 0:
            highest = grp.index[0]
            rate = float(grp.iloc[0])
            insights.append(f"Highest churn segment by subscription/contract: {highest} ({pct(rate)})")
            if rate > 0.30:
                recommendations.append(f"Target retention interventions for {highest} customers (offers, outreach, support).")

    if payment_col:
        grp = filtered_with_churn.groupby(payment_col)["__churn__"].mean().sort_values(ascending=False)
        if len(grp) > 0:
            highest = grp.index[0]
            rate = float(grp.iloc[0])
            insights.append(f"Highest churn segment by payment method: {highest} ({pct(rate)})")

            best_payment = filtered_with_churn.groupby(payment_col)["__churn__"].mean().idxmin()
            recommendations.append(f"Encourage adoption of lower-risk payment method: {best_payment} (incentives/reminders).")

    if tenure_col:
        tmp = filtered_with_churn.dropna(subset=[tenure_col])
        if len(tmp) > 0:
            early = float(tmp[tmp[tenure_col] <= np.nanpercentile(tmp[tenure_col], 25)]["__churn__"].mean())
            late = float(tmp[tmp[tenure_col] >= np.nanpercentile(tmp[tenure_col], 75)]["__churn__"].mean())
            insights.append(f"Tenure effect (lower vs higher tenure): {pct(early)} vs {pct(late)}")
            if early > 0.25:
                recommendations.append("Improve first-90-days onboarding (education, check-ins, friction reduction).")

    if usage_cols:
        recommendations.append("Monitor engagement metrics weekly; trigger re-engagement when activity drops below baseline.")

    if insights:
        st.write("Key observations")
        for i, x in enumerate(insights, 1):
            st.write(f"{i}. {x}")

    if recommendations:
        st.write("Actions")
        for i, r in enumerate(recommendations[:4], 1):
            st.write(f"{i}. {r}")
else:
    st.info("Churn label not available in the selected rows. Upload a labeled dataset to enable churn-based recommendations.")

# ---------------------------
# Raw data preview
# ---------------------------
if show_raw:
    st.subheader("Raw data preview")
    st.dataframe(filtered.head(200), use_container_width=True)
