from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.request import Request
from rest_framework.response import Response

from predictions.services.model_loader import (
    ModelUnavailableError,
    get_model_bundle,
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
