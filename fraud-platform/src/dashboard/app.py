"""
Streamlit dashboard for fraud case review.
"""
import logging
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = Path("data/alerts.db")

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("Real-Time Fraud Detection Dashboard")

st.sidebar.header("Filters")
status_filter = st.sidebar.selectbox("Status", ["All", "Open", "Under Review", "Confirmed Fraud", "False Positive"])

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Alerts", "0")
with col2:
    st.metric("High Risk", "0")
with col3:
    st.metric("Today's Cases", "0")

st.header("Recent Alerts")

# Sample data (would query DB in production)
sample_data = {
    "transaction_id": ["TX001", "TX002", "TX003"],
    "account_id": ["C1234567", "C2345678", "C3456789"],
    "fraud_score": [0.85, 0.92, 0.67],
    "amount": [15000.0, 25000.0, 5000.0],
    "status": ["Open", "Under Review", "Confirmed Fraud"],
    "timestamp": ["2026-10-21 10:30", "2026-10-21 11:15", "2026-10-21 09:45"],
}

df = pd.DataFrame(sample_data)

st.dataframe(df, use_container_width=True)

st.header("Risk Score Distribution")
st.bar_chart(df[["fraud_score"]])

st.header("Case Details")
selected_tx = st.selectbox("Select Transaction", df["transaction_id"])

if selected_tx:
    st.subheader(f"Transaction {selected_tx}")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Account: {df[df['transaction_id'] == selected_tx]['account_id'].values[0]}")
        st.write(f"Amount: ${df[df['transaction_id'] == selected_tx]['amount'].values[0]:,.2f}")
    with col2:
        st.write(f"Fraud Score: {df[df['transaction_id'] == selected_tx]['fraud_score'].values[0]:.2f}")
        st.write(f"Status: {df[df['transaction_id'] == selected_tx]['status'].values[0]}")

    st.subheader("Analyst Feedback")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Confirm Fraud"):
            st.success("Marked as confirmed fraud")
    with col2:
        if st.button("False Positive"):
            st.info("Marked as false positive")
    with col3:
        if st.button("Escalate"):
            st.warning("Case escalated")
