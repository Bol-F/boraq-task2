# Telecom Customer Churn MLOps

This repository provides the foundation and model-training workflow for a
Telecom Customer Churn MLOps project. It includes a Django REST Framework API,
a safe IBM dataset downloader, reproducible scikit-learn training, local MLflow
experiment tracking, model selection, artifact persistence, a documented churn
prediction endpoint, a Streamlit dashboard, and automated tests. Docker images
and CI/CD workflows belong to later project phases.

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

Downloaded data is local and ignored by Git. Tests mock HTTP requests and do
not require network access. The explicitly marked model-quality test uses the
local CSV when it is available.

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

```bash
uv run pytest -v
```

For the full model-quality assertion, download the dataset before running the
tests. Without the local CSV, that explicitly marked integration test is
skipped while the network-independent unit tests still run.

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
