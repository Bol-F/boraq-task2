from __future__ import annotations

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.request import Request
from rest_framework.response import Response

from predictions.serializers import PredictionRequestSerializer
from predictions.services.model_loader import (
    MODEL_UNAVAILABLE_DETAIL,
    ModelUnavailableError,
)
from predictions.services.prediction import predict_customer_churn


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
