from rest_framework.decorators import api_view
from rest_framework.request import Request
from rest_framework.response import Response


@api_view(["GET"])
def health_check(_request: Request) -> Response:
    """Return a small response confirming that the API is available."""
    return Response(
        {
            "status": "ok",
            "service": "churn-prediction-api",
        }
    )
