from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from django.conf import settings

pytestmark = pytest.mark.unit


def test_static_files_use_whitenoise_manifest_storage() -> None:
    assert settings.MIDDLEWARE[1] == "whitenoise.middleware.WhiteNoiseMiddleware"
    assert settings.STATIC_URL == "/static/"
    assert settings.STATIC_ROOT == settings.BASE_DIR / "staticfiles"
    assert settings.STORAGES["staticfiles"]["BACKEND"] == (
        "whitenoise.storage.CompressedManifestStaticFilesStorage"
    )


def test_render_environment_enables_production_security_settings() -> None:
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.update(
        {
            "DJANGO_SECRET_KEY": "deployment-test-secret-with-more-than-fifty-characters-123",
            "DJANGO_DEBUG": "false",
            "DJANGO_ALLOWED_HOSTS": "api.example.com, localhost,,api.example.com",
            "RENDER_EXTERNAL_HOSTNAME": "telecom-churn-api.onrender.com",
        }
    )
    command = """
import json
from config import settings

print(json.dumps({
    "allowed_hosts": settings.ALLOWED_HOSTS,
    "csrf_cookie_secure": settings.CSRF_COOKIE_SECURE,
    "hsts_include_subdomains": settings.SECURE_HSTS_INCLUDE_SUBDOMAINS,
    "hsts_preload": settings.SECURE_HSTS_PRELOAD,
    "hsts_seconds": settings.SECURE_HSTS_SECONDS,
    "secure_proxy_ssl_header": settings.SECURE_PROXY_SSL_HEADER,
    "secure_ssl_redirect": settings.SECURE_SSL_REDIRECT,
    "session_cookie_secure": settings.SESSION_COOKIE_SECURE,
}))
"""

    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    production_settings = json.loads(result.stdout)

    assert production_settings["allowed_hosts"] == [
        "api.example.com",
        "localhost",
        "telecom-churn-api.onrender.com",
    ]
    assert "*" not in production_settings["allowed_hosts"]
    assert production_settings["secure_proxy_ssl_header"] == [
        "HTTP_X_FORWARDED_PROTO",
        "https",
    ]
    assert production_settings["session_cookie_secure"] is True
    assert production_settings["csrf_cookie_secure"] is True
    assert production_settings["secure_ssl_redirect"] is True
    assert production_settings["hsts_seconds"] == 31_536_000
    assert production_settings["hsts_include_subdomains"] is True
    assert production_settings["hsts_preload"] is True


def test_production_environment_requires_an_explicit_secret_key() -> None:
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.pop("DJANGO_SECRET_KEY", None)
    environment.update(
        {
            "DJANGO_DEBUG": "false",
            "DJANGO_ALLOWED_HOSTS": "localhost",
            "RENDER_EXTERNAL_HOSTNAME": "",
        }
    )

    result = subprocess.run(
        [sys.executable, "-c", "import config.settings"],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "DJANGO_SECRET_KEY must be set" in result.stderr
