from django.urls import path

from predictions.views.health import health_check

app_name = "predictions"

urlpatterns = [
    path("health/", health_check, name="health"),
]
