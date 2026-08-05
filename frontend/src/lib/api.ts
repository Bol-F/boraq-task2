import { MODEL_FIELD_NAMES } from "@/lib/schema";
import type {
  CustomerFieldErrors,
  CustomerFieldName,
  CustomerPayload,
  HealthResponse,
  PredictionResponse,
  RiskLevel,
} from "@/lib/types";

const HEALTH_ROUTE = "/api/health";
const PREDICTION_ROUTE = "/api/predict";
const RISK_LEVELS: readonly RiskLevel[] = ["low", "medium", "high"];

export class ApiRequestError extends Error {
  readonly fieldErrors: CustomerFieldErrors;
  readonly status: number | null;

  constructor(
    message: string,
    options: { fieldErrors?: CustomerFieldErrors; status?: number | null } = {},
  ) {
    super(message);
    this.name = "ApiRequestError";
    this.fieldErrors = options.fieldErrors ?? {};
    this.status = options.status ?? null;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

async function readJsonSafely(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function isCustomerFieldName(value: string): value is CustomerFieldName {
  return MODEL_FIELD_NAMES.some((fieldName) => fieldName === value);
}

function validationErrorsFrom(body: unknown): CustomerFieldErrors {
  if (!isRecord(body)) {
    return {};
  }

  const errors: CustomerFieldErrors = {};
  for (const [fieldName, value] of Object.entries(body)) {
    if (!isCustomerFieldName(fieldName)) {
      continue;
    }

    if (typeof value === "string") {
      errors[fieldName] = value;
    } else if (Array.isArray(value) && typeof value[0] === "string") {
      errors[fieldName] = value[0];
    }
  }
  return errors;
}

function detailFrom(body: unknown): string | null {
  if (!isRecord(body)) {
    return null;
  }
  return typeof body.detail === "string" ? body.detail : null;
}

function responseError(response: Response, body: unknown): ApiRequestError {
  const fieldErrors = validationErrorsFrom(body);
  const hasFieldErrors = Object.keys(fieldErrors).length > 0;
  const fallback =
    response.status === 503
      ? "The prediction model is temporarily unavailable."
      : response.status >= 500
        ? "The prediction service could not complete the request."
        : hasFieldErrors
          ? "Review the highlighted fields and try again."
          : "The request could not be completed.";

  return new ApiRequestError(detailFrom(body) ?? fallback, {
    fieldErrors,
    status: response.status,
  });
}

function isHealthResponse(value: unknown): value is HealthResponse {
  if (!isRecord(value)) {
    return false;
  }

  return (
    (value.status === "ok" || value.status === "degraded") &&
    typeof value.service === "string" &&
    typeof value.model_loaded === "boolean" &&
    (typeof value.model_version === "string" || value.model_version === null)
  );
}

function isPredictionResponse(value: unknown): value is PredictionResponse {
  if (!isRecord(value)) {
    return false;
  }

  return (
    typeof value.churn_probability === "number" &&
    Number.isFinite(value.churn_probability) &&
    value.churn_probability >= 0 &&
    value.churn_probability <= 1 &&
    typeof value.will_churn === "boolean" &&
    RISK_LEVELS.some((risk) => risk === value.risk) &&
    typeof value.model_version === "string"
  );
}

export async function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  let response: Response;
  try {
    response = await fetch(HEALTH_ROUTE, {
      cache: "no-store",
      headers: { Accept: "application/json" },
      signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }
    throw new ApiRequestError("The API status could not be reached.");
  }

  const body = await readJsonSafely(response);
  if (isHealthResponse(body)) {
    return body;
  }
  throw responseError(response, body);
}

export async function predictCustomer(
  payload: CustomerPayload,
  signal?: AbortSignal,
): Promise<PredictionResponse> {
  let response: Response;
  try {
    response = await fetch(PREDICTION_ROUTE, {
      body: JSON.stringify(payload),
      cache: "no-store",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      method: "POST",
      signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiRequestError("The prediction request timed out.");
    }
    throw new ApiRequestError("The prediction service could not be reached.");
  }

  const body = await readJsonSafely(response);
  if (!response.ok) {
    throw responseError(response, body);
  }
  if (!isPredictionResponse(body)) {
    throw new ApiRequestError("The prediction service returned an invalid response.", {
      status: response.status,
    });
  }
  return body;
}
