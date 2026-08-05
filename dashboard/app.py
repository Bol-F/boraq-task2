"""Streamlit entry point for telecom customer churn predictions."""

from __future__ import annotations

from pathlib import Path

import environ
import streamlit as st

if __package__:
    from dashboard.api_client import (
        ApiResult,
        HealthData,
        PredictionData,
        get_api_base_url,
        get_health_status,
    )
    from dashboard.api_client import predict_customer as request_prediction
    from dashboard.components import (
        DashboardValidationError,
        render_api_error,
        render_customer_form,
        render_health_status,
        render_prediction_result,
    )
else:
    from api_client import (
        ApiResult,
        HealthData,
        PredictionData,
        get_api_base_url,
        get_health_status,
    )
    from api_client import predict_customer as request_prediction
    from components import (
        DashboardValidationError,
        render_api_error,
        render_customer_form,
        render_health_status,
        render_prediction_result,
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTION_RESULT_KEY = "latest_prediction_result"


def load_local_environment() -> None:
    """Load the ignored local .env file without replacing shell variables."""
    environment_path = PROJECT_ROOT / ".env"
    if environment_path.is_file():
        environ.Env.read_env(environment_path)


@st.cache_data(ttl=30, show_spinner=False, max_entries=8)
def load_health_status(api_base_url: str) -> ApiResult[HealthData]:
    """Cache short-lived readiness results across Streamlit widget reruns."""
    return get_health_status(base_url=api_base_url)


def main() -> None:
    """Render the dashboard and request predictions only after form submission."""
    load_local_environment()
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

    api_base_url = get_api_base_url()
    with st.sidebar:
        st.header("API status")
        if st.button("Refresh API status", use_container_width=True):
            load_health_status.clear()
        health_result = load_health_status(api_base_url)
        render_health_status(health_result, api_base_url)

    try:
        customer_payload = render_customer_form()
    except DashboardValidationError as error:
        st.error(str(error))
    else:
        if customer_payload is not None:
            with st.spinner("Calculating churn risk..."):
                result = request_prediction(
                    customer_payload,
                    base_url=api_base_url,
                )
            st.session_state[PREDICTION_RESULT_KEY] = result

    latest_result: ApiResult[PredictionData] | None = st.session_state.get(
        PREDICTION_RESULT_KEY
    )
    if latest_result is None:
        return
    if latest_result.is_success and latest_result.data is not None:
        render_prediction_result(latest_result.data)
    elif latest_result.error is not None:
        render_api_error(latest_result.error)


if __name__ == "__main__":
    main()
