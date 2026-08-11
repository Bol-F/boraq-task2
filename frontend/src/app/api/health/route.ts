import {
  fetchBackendJson,
  jsonResponse,
  proxyFailureResponse,
} from "@/lib/backend";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

interface HealthResponse {
  model_loaded: boolean;
  model_version: string | null;
  service: string;
  status: "degraded" | "ok";
}

export async function GET(): Promise<Response> {
  try {
    const backendResponse = await fetchBackendJson("/api/health/", {
      method: "GET",
    });

    if (
      [200, 503].includes(backendResponse.status) &&
      isHealthResponse(backendResponse.body)
    ) {
      return jsonResponse(backendResponse.body, backendResponse.status);
    }

    return jsonResponse(
      { detail: "Prediction API health check failed." },
      meaningfulStatus(backendResponse.status),
    );
  } catch (error) {
    return proxyFailureResponse(error);
  }
}

function isHealthResponse(value: unknown): value is HealthResponse {
  if (!isRecord(value)) {
    return false;
  }

  const hasValidVersion =
    value.model_version === null ||
    (typeof value.model_version === "string" && value.model_version.length > 0);

  return (
    (value.status === "ok" || value.status === "degraded") &&
    value.service === "churn-prediction-api" &&
    typeof value.model_loaded === "boolean" &&
    hasValidVersion
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function meaningfulStatus(status: number): number {
  return status >= 400 && status <= 599 ? status : 502;
}
