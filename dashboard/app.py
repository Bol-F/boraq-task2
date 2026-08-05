"""Streamlit entry point for telecom customer churn predictions."""

from __future__ import annotations

import streamlit as st

from dashboard.components import DashboardValidationError, render_customer_form

st.set_page_config(
    page_title="Telecom Churn Risk Dashboard",
    page_icon="📡",
    layout="wide",
)

st.title("Telecom Customer Churn Risk")
st.write(
    "Customer churn means a customer stops using a company's service. "
    "Enter one customer's account and service details to estimate their risk."
)
st.info(
    "This prediction is a model estimate, not a guaranteed customer outcome. "
    "Use it as one input to a broader review."
)

try:
    customer_payload = render_customer_form()
except DashboardValidationError as error:
    st.error(str(error))
else:
    if customer_payload is not None:
        st.success("Customer details are complete and ready for prediction.")
