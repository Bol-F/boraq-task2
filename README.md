# Telecom Customer Churn MLOps

This repository is the foundation for a Telecom Customer Churn MLOps project.
It currently provides a Django REST Framework API, environment-based settings,
quality tooling, and a tested health endpoint. Model training, prediction logic,
MLflow experiments, the Streamlit dashboard, Docker images, and CI/CD workflows
belong to later project phases.

## Technology stack

- Python 3.11 and uv for the runtime and dependency management
- Django and Django REST Framework for the API
- pandas, scikit-learn, and joblib for future data and model work
- MLflow for future experiment tracking
- Streamlit for the future dashboard
- requests for HTTP clients
- Psycopg for future PostgreSQL connectivity
- Gunicorn for production serving on supported platforms
- pytest and pytest-django for tests
- Ruff for linting and formatting
- Docker and GitHub Actions for future container and CI/CD work

## Installation with uv

Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) if it is not
already available. From the project directory, let uv install the pinned Python
version and all locked dependencies:

```bash
uv python install 3.11
uv sync
```

Create a local environment file from the safe example and replace the example
secret before using production-like settings:

```powershell
Copy-Item .env.example .env
```

On macOS or Linux, use `cp .env.example .env` instead. The real `.env` file is
ignored by Git and must never be committed.

## Dependency synchronization

Run this command after switching branches or whenever `pyproject.toml` or
`uv.lock` changes:

```bash
uv sync
```

Use `uv add <package>` for a production dependency and
`uv add --dev <package>` for a development dependency. Commit both
`pyproject.toml` and `uv.lock` when dependencies change.

## Database migrations

Apply the built-in Django migrations to the local SQLite database:

```bash
uv run python manage.py migrate
```

When a future model change needs a migration, create and apply it with:

```bash
uv run python manage.py makemigrations
uv run python manage.py migrate
```

## Run Django locally

Check the configuration, apply migrations, and start the development server:

```bash
uv run python manage.py check
uv run python manage.py migrate
uv run python manage.py runserver
```

The API is then available at `http://127.0.0.1:8000/api/`.

## Run tests

```bash
uv run pytest -v
```

## Run Ruff

Check for lint problems and confirm that files are formatted:

```bash
uv run ruff check .
uv run ruff format --check .
```

To format Python files automatically, run:

```bash
uv run ruff format .
```

## Health endpoint

With the development server running, request the health endpoint:

```bash
curl http://127.0.0.1:8000/api/health/
```

PowerShell users can run `curl.exe` explicitly. A successful response is:

```json
{
  "status": "ok",
  "service": "churn-prediction-api"
}
```

The endpoint accepts `GET` requests. Other methods, including `POST`, return
HTTP 405 Method Not Allowed.
