from __future__ import annotations

import math
from collections.abc import Mapping

from rest_framework import serializers

YES_NO_CHOICES = ("No", "Yes")
MULTIPLE_LINES_CHOICES = ("No", "No phone service", "Yes")
INTERNET_SERVICE_CHOICES = ("DSL", "Fiber optic", "No")
INTERNET_ADDON_CHOICES = ("No", "No internet service", "Yes")
INTERNET_ADDON_FIELDS = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)
CONTRACT_CHOICES = ("Month-to-month", "One year", "Two year")
PAYMENT_METHOD_CHOICES = (
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check",
)


def validate_finite_number(value: float) -> None:
    """Reject NaN and infinite values before they reach scikit-learn."""
    if not math.isfinite(value):
        raise serializers.ValidationError("A finite number is required.")


class PredictionRequestSerializer(serializers.Serializer):
    """Validate the 19 raw customer features expected by the saved pipeline."""

    gender = serializers.ChoiceField(choices=("Female", "Male"))
    SeniorCitizen = serializers.ChoiceField(choices=(0, 1))
    Partner = serializers.ChoiceField(choices=YES_NO_CHOICES)
    Dependents = serializers.ChoiceField(choices=YES_NO_CHOICES)
    tenure = serializers.IntegerField(min_value=0, max_value=72)
    PhoneService = serializers.ChoiceField(choices=YES_NO_CHOICES)
    MultipleLines = serializers.ChoiceField(choices=MULTIPLE_LINES_CHOICES)
    InternetService = serializers.ChoiceField(choices=INTERNET_SERVICE_CHOICES)
    OnlineSecurity = serializers.ChoiceField(choices=INTERNET_ADDON_CHOICES)
    OnlineBackup = serializers.ChoiceField(choices=INTERNET_ADDON_CHOICES)
    DeviceProtection = serializers.ChoiceField(choices=INTERNET_ADDON_CHOICES)
    TechSupport = serializers.ChoiceField(choices=INTERNET_ADDON_CHOICES)
    StreamingTV = serializers.ChoiceField(choices=INTERNET_ADDON_CHOICES)
    StreamingMovies = serializers.ChoiceField(choices=INTERNET_ADDON_CHOICES)
    Contract = serializers.ChoiceField(choices=CONTRACT_CHOICES)
    PaperlessBilling = serializers.ChoiceField(choices=YES_NO_CHOICES)
    PaymentMethod = serializers.ChoiceField(choices=PAYMENT_METHOD_CHOICES)
    MonthlyCharges = serializers.FloatField(
        min_value=0.0,
        validators=[validate_finite_number],
    )
    TotalCharges = serializers.FloatField(
        min_value=0.0,
        validators=[validate_finite_number],
    )

    def to_internal_value(self, data: object) -> dict[str, object]:
        """Reject unexpected keys instead of silently ignoring them."""
        if isinstance(data, Mapping):
            unknown_fields = sorted(set(data) - set(self.fields))
            if unknown_fields:
                raise serializers.ValidationError(
                    {field_name: ["Unknown field."] for field_name in unknown_fields}
                )

        return super().to_internal_value(data)

    def validate(self, attrs: dict[str, object]) -> dict[str, object]:
        """Keep phone and internet special values internally consistent."""
        errors: dict[str, list[str]] = {}

        has_phone_service = attrs["PhoneService"] == "Yes"
        no_phone_value = attrs["MultipleLines"] == "No phone service"
        if has_phone_service and no_phone_value:
            errors["MultipleLines"] = ["Use No or Yes when PhoneService is Yes."]
        elif not has_phone_service and not no_phone_value:
            errors["MultipleLines"] = ["Use No phone service when PhoneService is No."]

        has_internet_service = attrs["InternetService"] != "No"
        for field_name in INTERNET_ADDON_FIELDS:
            no_internet_value = attrs[field_name] == "No internet service"
            if has_internet_service and no_internet_value:
                errors[field_name] = ["Use No or Yes when internet service is active."]
            elif not has_internet_service and not no_internet_value:
                errors[field_name] = [
                    "Use No internet service when InternetService is No."
                ]

        if errors:
            raise serializers.ValidationError(errors)
        return attrs


class PredictionResponseSerializer(serializers.Serializer):
    """Document the successful prediction response."""

    churn_probability = serializers.FloatField(min_value=0.0, max_value=1.0)
    will_churn = serializers.BooleanField()
    risk = serializers.ChoiceField(choices=("low", "medium", "high"))
    model_version = serializers.CharField()


class ModelUnavailableResponseSerializer(serializers.Serializer):
    """Document the stable response returned when prediction is unavailable."""

    detail = serializers.CharField()


class HealthResponseSerializer(serializers.Serializer):
    """Document ready and degraded health response fields."""

    status = serializers.ChoiceField(choices=("ok", "degraded"))
    service = serializers.CharField()
    model_loaded = serializers.BooleanField()
    model_version = serializers.CharField(allow_null=True)
