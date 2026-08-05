from django.urls import path

from predictions.views.health import health_check
from predictions.views.prediction import predict_churn

app_name = "predictions"

urlpatterns = [
    path("health/", health_check, name="health"),
    path("predict/", predict_churn, name="predict"),
]
