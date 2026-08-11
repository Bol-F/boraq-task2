import {
  fetchBackendJson,
  jsonResponse,
  proxyFailureResponse,
} from "@/lib/backend";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const RISKS = new Set(["low", "medium", "high"]);

interface PredictionResponse {
  churn_probability: number;
  model_version: string;
  risk: "high" | "low" | "medium";
  will_churn: boolean;
}

export async function POST(request: Request): Promise<Response> {
  let customer: unknown;
  try {
    customer = await request.json();
  } catch {
    return jsonResponse({ detail: "Request body must be valid JSON." }, 400);
  }

  if (!isRecord(customer)) {
    return jsonResponse({ detail: "Request body must be a JSON object." }, 400);
  }

  try {
    const backendResponse = await fetchBackendJson("/api/predict/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(customer),
    });

    if (
      backendResponse.status === 200 &&
      isPredictionResponse(backendResponse.body)
    ) {
      return jsonResponse(backendResponse.body, 200);
    }

    if (backendResponse.status === 400) {
      const validationErrors = sanitizeValidationErrors(backendResponse.body);
      return jsonResponse(
        validationErrors ?? { detail: "Customer data is invalid." },
        400,
      );
    }

    if (backendResponse.status === 503) {
      return jsonResponse(
        { detail: "Prediction model is not available." },
        503,
      );
    }

    return jsonResponse(
      { detail: "Prediction API request failed." },
      meaningfulStatus(backendResponse.status),
    );
  } catch (error) {
    return proxyFailureResponse(error);
  }
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
    typeof value.risk === "string" &&
    RISKS.has(value.risk) &&
    typeof value.model_version === "string" &&
    value.model_version.length > 0
  );
}

function sanitizeValidationErrors(
  value: unknown,
): Record<string, string | string[]> | null {
  if (!isRecord(value)) {
    return null;
  }

  const errors: Record<string, string | string[]> = {};
  for (const [field, messages] of Object.entries(value)) {
    if (typeof messages === "string") {
      errors[field] = messages;
    } else if (
      Array.isArray(messages) &&
      messages.every((message) => typeof message === "string")
    ) {
      errors[field] = messages;
    }
  }

  return Object.keys(errors).length > 0 ? errors : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function meaningfulStatus(status: number): number {
  return status >= 400 && status <= 599 ? status : 502;
}
