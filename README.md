# Telecom Customer Churn MLOps

This repository provides the foundation and model-training workflow for a
Telecom Customer Churn MLOps project. It includes a Django REST Framework API,
a safe IBM dataset downloader, reproducible scikit-learn training, local MLflow
experiment tracking, model selection, artifact persistence, a documented churn
prediction endpoint, a Streamlit dashboard, and automated tests. Docker images
are provided for the Django prediction API. CI/CD workflows belong to a later
project phase.

## Technology stack

- Python 3.11 and uv for the runtime and dependency management
- Django and Django REST Framework for the API
- drf-spectacular for OpenAPI schema generation and Swagger UI
- pandas, scikit-learn, and joblib for preprocessing, training, and persistence
- MLflow for local experiment tracking and model artifacts
- Streamlit for the customer churn dashboard
- requests for HTTP clients
- Psycopg for future PostgreSQL connectivity
- Gunicorn for production serving on supported platforms
- pytest, pytest-django, and pytest-cov for tests and coverage
- Ruff for linting and formatting
- Docker for the production-oriented API image and local Compose workflow
- GitHub Actions for future CI/CD work

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

## Dataset

The project uses the public
[IBM Telco Customer Churn dataset](https://github.com/IBM/telco-customer-churn-on-icp4d).
It contains 7,043 fictional telecom customers and 21 columns. `Churn` is the
Yes/No target. After removing `customerID` and the target, 19 model features
remain.

Download the pinned and checksum-verified CSV to `data/telco_churn.csv`:

```bash
uv run python manage.py download_churn_data
```

The command validates all required columns and will not silently replace an
existing dataset. To intentionally download it again, use:

```bash
uv run python manage.py download_churn_data --overwrite
```

Downloaded data is local and ignored by Git. Fast tests use a tracked fixture
and mock HTTP requests, so they do not require network access. The full test
suite deliberately fails with a setup instruction if this real CSV is absent,
because the model-quality test is a required gate rather than an optional skip.

## Train churn models

Download the dataset first, then train and compare all configured models:

```bash
uv run python manage.py train_churn_model
```

Training performs one stratified 80/20 split with `random_state=42`. It compares
balanced logistic regression and balanced random forest classifiers. Each
candidate is one complete scikit-learn pipeline containing `TotalCharges`
cleaning, automatic numerical/categorical preprocessing, scaling, one-hot
encoding, and the classifier. Saving the whole pipeline keeps training and
future API inference transformations consistent.

The winning pipeline and its calculated metadata are written to:

```text
models/model.pkl
models/model_metadata.json
```

These generated files are ignored by Git. Only load joblib model files produced
by a trusted training workflow.

### Class imbalance and metrics

Only about 26.5% of the customers in this dataset churn. Both classifiers use
balanced class weights so mistakes on the smaller churn class receive more
importance during fitting.

Every model is evaluated with:

- **ROC-AUC**, which measures ranking quality across classification thresholds
  and is the model-selection metric;
- **PR-AUC** (average precision), which emphasizes precision and recall for the
  minority churn class and is especially informative for imbalanced data;
- **F1**, which summarizes the balance between precision and recall at the
  classifier's decision threshold.

Accuracy is not used as the main selection metric because an imbalanced model
can appear accurate simply by favoring the majority non-churn class.

## MLflow

Training creates or reuses a local experiment named `telecom-churn`. It creates
one clearly named run per candidate and logs parameters, ROC-AUC, PR-AUC, F1,
and the complete fitted pipeline. MLflow uses a project-local SQLite database
and local artifact directory; no external or paid tracking server is required.

Start the local MLflow interface from the project root:

```bash
uv run mlflow ui
```

Open `http://127.0.0.1:5000` to compare runs and inspect artifacts. Local MLflow
database and artifact files are ignored by Git.

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

The saved model and metadata must exist at `models/model.pkl` and
`models/model_metadata.json` for predictions. Train the model first if those
local artifacts are not present.

## Prediction endpoint

Send one customer's 19 raw features to `POST /api/predict/`. The API validates
the request, then passes it directly to the cached, complete scikit-learn
pipeline. It does not recreate the training preprocessing.

Example request body:

```json
{
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
  "TotalCharges": 450.5
}
```

With the current locally trained artifact, an example HTTP 200 response is:

```json
{
  "churn_probability": 0.899,
  "will_churn": true,
  "risk": "high",
  "model_version": "1.0.0"
}
```

Probabilities are rounded to four decimal places. `will_churn` uses a `0.5`
threshold. Risk is `low` below `0.35`, `medium` from `0.35` to below `0.65`,
and `high` from `0.65` upward.

Validation problems return HTTP 400 with standard field-based Django REST
Framework errors. For example:

```json
{
  "tenure": [
    "Ensure this value is greater than or equal to 0."
  ]
}
```

If the model or metadata cannot be loaded, prediction returns HTTP 503 without
exposing internal exception details or local paths:

```json
{
  "detail": "Prediction model is not available."
}
```

Call the endpoint with curl from macOS or Linux:

```bash
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{
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
    "TotalCharges": 450.5
  }'
```

PowerShell users can use `curl.exe` with equivalent quoting or
`Invoke-RestMethod`.

## API documentation

With Django running, interactive Swagger documentation is available at
`http://127.0.0.1:8000/api/docs/`. The generated OpenAPI schema is available at
`http://127.0.0.1:8000/api/schema/`.

## Streamlit dashboard

The dashboard at `dashboard/app.py` provides a guided form for all 19 customer
features, displays prediction results, and reports API/model readiness. It is a
presentation client: it sends HTTP requests to Django and never imports or
loads `models/model.pkl` itself. Django remains responsible for validation,
preprocessing, and inference.

The dashboard reads the Django base URL from `API_URL` and defaults to
`http://127.0.0.1:8000`. You can set it in the ignored local `.env` file:

```dotenv
API_URL=http://127.0.0.1:8000
```

Use only the base URL; the dashboard safely adds `/api/health/` and
`/api/predict/`. Do not include either endpoint path in `API_URL`.

Run Django in the first terminal:

```bash
uv run python manage.py runserver
```

Run Streamlit from the project root in a second terminal:

```bash
uv run streamlit run dashboard/app.py
```

The normal local URLs are:

- Django API: `http://127.0.0.1:8000/api/`
- Streamlit dashboard: `http://127.0.0.1:8501`

The expected workflow is:

1. The cached sidebar health check confirms that Django is reachable and the
   prediction model is loaded.
2. A user completes the customer form and selects **Predict churn risk**.
3. Streamlit sends one JSON request to `POST /api/predict/`.
4. The dashboard displays the probability, Boolean threshold result, risk
   level, neutral interpretation, and model version returned by Django.

If the dashboard reports a connection error, confirm that Django is running,
that `API_URL` uses the correct host and port, and that
`http://127.0.0.1:8000/api/health/` is reachable. An HTTP 503 health response
means Django is reachable but the local model or metadata is unavailable; run
the training command to recreate those ignored artifacts. The dashboard stays
available during these failures and shows a safe error instead of a Python
traceback.

## Run tests

Tests are separated by responsibility:

```text
tests/
├── fixtures/sample_churn.csv
├── conftest.py
├── test_data_pipeline.py
├── test_training.py
├── test_model_quality.py
├── test_prediction_api.py
├── test_health.py
├── test_model_loader.py
├── test_dashboard_api_client.py
├── test_dashboard_components.py
├── test_data_download.py
└── test_openapi.py
```

The deterministic sample CSV has the complete production schema, both churn
classes, blank and invalid `TotalCharges` values, all contract types, and the
phone/internet special values used by the real dataset. Temporary artifact
fixtures keep tests from overwriting `models/`.

Run the fast, network-independent suite without the real-data quality gate:

```bash
uv run pytest -m "not model_quality"
```

Download the real dataset and run every test, including the quality gate:

```bash
uv run python manage.py download_churn_data
uv run pytest -v
```

The `model_quality` test uses the shared preprocessing pipeline, a stratified
80/20 split, the configured Logistic Regression baseline, and
`random_state=42`. It requires ROC-AUC to be strictly greater than `0.78`; a
regression reports the measured score in the failure message. The normal
`uv run pytest` command includes this gate.

Run tests with missing-line coverage and the configured 75% starting floor:

```bash
uv run pytest --cov=. --cov-report=term-missing
```

Coverage omits migrations, generated server entry points, test code, and the
Streamlit page-composition entry point. Business logic in the data pipeline,
training services, prediction API, model loader, and dashboard client remains
included.

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

PowerShell users can run `curl.exe` explicitly. A ready HTTP 200 response is:

```json
{
  "status": "ok",
  "service": "churn-prediction-api",
  "model_loaded": true,
  "model_version": "1.0.0"
}
```

If the model or metadata is unavailable, the endpoint remains responsive and
returns HTTP 503 with readiness information:

```json
{
  "status": "degraded",
  "service": "churn-prediction-api",
  "model_loaded": false,
  "model_version": null
}
```

HTTP 503 is used consistently for the degraded response because this endpoint
acts as a readiness check: the web process is reachable, but it is not ready to
serve predictions. The endpoint accepts `GET` requests. Other methods,
including `POST`, return HTTP 405 Method Not Allowed.

## Run the Django API with Docker

The multi-stage image uses Python 3.11 slim and pinned `uv`, installs locked
production dependencies with `uv sync --frozen --no-dev`, copies only the
Django API code and required model artifacts, runs a Django check during the
build, and serves through Gunicorn as an unprivileged user. Its health check
uses Python's standard library, so no extra curl package is needed. The
Swagger UI uses its configured CDN assets and does not require collected local
static files.

The generated model and metadata are intentionally ignored by Git but are
required in the Docker build context. Create them before building if needed:

```bash
uv run python manage.py download_churn_data
uv run python manage.py train_churn_model
```

Build the image:

```bash
docker build -t churn-api .
```

Run it directly on port 8000:

```bash
docker run --rm -p 8000:8000 \
  -e DJANGO_SECRET_KEY=local-development-secret \
  -e DJANGO_DEBUG=false \
  -e DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1 \
  -e MODEL_PATH=/app/models/model.pkl \
  -e MODEL_METADATA_PATH=/app/models/model_metadata.json \
  -e PORT=8000 \
  churn-api
```

Or build and start the same API through Compose, with no host volumes that can
overwrite the bundled artifacts:

```bash
docker compose up --build
```

The API container reads these settings:

| Variable | Purpose | Local/container default |
| --- | --- | --- |
| `DJANGO_SECRET_KEY` | Django signing secret | Unsafe local fallback; always set for production |
| `DJANGO_DEBUG` | Enables Django debug mode | `True` locally, `false` in the image |
| `DJANGO_ALLOWED_HOSTS` | Comma-separated accepted hosts | `localhost,127.0.0.1` |
| `MODEL_PATH` | Saved scikit-learn pipeline | `models/model.pkl` |
| `MODEL_METADATA_PATH` | Model metadata JSON | `models/model_metadata.json` |
| `PORT` | Internal Gunicorn listening port | `8000` |

Once the container is ready, verify its three public surfaces:

```bash
curl http://127.0.0.1:8000/api/health/
curl http://127.0.0.1:8000/api/docs/
```

Use the complete prediction curl request in the Prediction endpoint section to
verify `POST /api/predict/` against the container.

Common Docker troubleshooting:

- If the build reports missing `models/model.pkl` or metadata, run the download
  and training commands above before rebuilding.
- If `/api/health/` returns HTTP 503, confirm both model paths are correct and
  that the two artifacts were produced by the current project dependencies.
- If Django reports an invalid host, add the hostname used by the request to
  the comma-separated `DJANGO_ALLOWED_HOSTS` value. Keep `127.0.0.1` for the
  built-in container health check.
- If port 8000 is already in use, publish a different host port, for example
  `-p 8080:8000`, and call the API on port 8080.
- If Docker cannot connect to its engine, start Docker Desktop or the local
  Docker daemon before running build or Compose commands.
