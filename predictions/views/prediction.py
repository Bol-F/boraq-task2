from __future__ import annotations

from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiExample, OpenApiResponse, extend_schema
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.request import Request
from rest_framework.response import Response

from predictions.serializers import (
    ModelUnavailableResponseSerializer,
    PredictionRequestSerializer,
    PredictionResponseSerializer,
)
from predictions.services.model_loader import (
    MODEL_UNAVAILABLE_DETAIL,
    ModelUnavailableError,
)
from predictions.services.prediction import predict_customer_churn


@extend_schema(
    operation_id="predict_customer_churn",
    summary="Predict churn for one telecom customer",
    description=(
        "Validates 19 customer features and sends them directly through the "
        "cached, complete scikit-learn pipeline."
    ),
    request=PredictionRequestSerializer,
    responses={
        200: PredictionResponseSerializer,
        400: OpenApiResponse(
            response=OpenApiTypes.OBJECT,
            description="The JSON body or one or more customer fields are invalid.",
        ),
        503: ModelUnavailableResponseSerializer,
    },
    examples=[
        OpenApiExample(
            "High-risk customer",
            request_only=True,
            value={
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 5,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 89.9,
                "TotalCharges": 450.5,
            },
        ),
        OpenApiExample(
            "Successful high-risk prediction",
            response_only=True,
            status_codes=["200"],
            value={
                "churn_probability": 0.899,
                "will_churn": True,
                "risk": "high",
                "model_version": "1.0.0",
            },
        ),
        OpenApiExample(
            "Validation error",
            response_only=True,
            status_codes=["400"],
            value={"tenure": ["Ensure this value is greater than or equal to 0."]},
        ),
        OpenApiExample(
            "Prediction model unavailable",
            response_only=True,
            status_codes=["503"],
            value={"detail": MODEL_UNAVAILABLE_DETAIL},
        ),
    ],
    tags=["Predictions"],
)
@api_view(["POST"])
def predict_churn(request: Request) -> Response:
    """Validate one customer and return churn probability and risk."""
    serializer = PredictionRequestSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)

    try:
        prediction = predict_customer_churn(serializer.validated_data)
    except ModelUnavailableError:
        return Response(
            {"detail": MODEL_UNAVAILABLE_DETAIL},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    return Response(prediction.as_dict(), status=status.HTTP_200_OK)
