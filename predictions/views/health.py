from drf_spectacular.utils import OpenApiExample, extend_schema
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.request import Request
from rest_framework.response import Response

from predictions.serializers import HealthResponseSerializer
from predictions.services.model_loader import (
    ModelUnavailableError,
    get_model_bundle,
)


@extend_schema(
    operation_id="health_check",
    summary="Check API and prediction model readiness",
    responses={
        200: HealthResponseSerializer,
        503: HealthResponseSerializer,
    },
    examples=[
        OpenApiExample(
            "Ready",
            response_only=True,
            status_codes=["200"],
            value={
                "status": "ok",
                "service": "churn-prediction-api",
                "model_loaded": True,
                "model_version": "1.0.0",
            },
        ),
        OpenApiExample(
            "Degraded",
            response_only=True,
            status_codes=["503"],
            value={
                "status": "degraded",
                "service": "churn-prediction-api",
                "model_loaded": False,
                "model_version": None,
            },
        ),
    ],
    tags=["Health"],
)
@api_view(["GET"])
def health_check(_request: Request) -> Response:
    """Report API availability and whether this process can serve predictions."""
    try:
        bundle = get_model_bundle()
    except ModelUnavailableError:
        return Response(
            {
                "status": "degraded",
                "service": "churn-prediction-api",
                "model_loaded": False,
                "model_version": None,
            },
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    return Response(
        {
            "status": "ok",
            "service": "churn-prediction-api",
            "model_loaded": True,
            "model_version": bundle.model_version,
        },
        status=status.HTTP_200_OK,
    )
