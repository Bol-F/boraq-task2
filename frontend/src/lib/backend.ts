import "server-only";

export const BACKEND_TIMEOUT_MS = 10_000;

export class BackendConfigurationError extends Error {
  constructor() {
    super("The backend API is not configured.");
    this.name = "BackendConfigurationError";
  }
}

export class InvalidBackendResponseError extends Error {
  constructor() {
    super("The backend API returned an unreadable response.");
    this.name = "InvalidBackendResponseError";
  }
}

export interface BackendJsonResponse {
  body: unknown;
  ok: boolean;
  status: number;
}

export function buildBackendUrl(
  endpoint: `/${string}`,
  configuredBaseUrl = process.env.RENDER_API_URL,
): string {
  const rawBaseUrl = configuredBaseUrl?.trim();
  if (!rawBaseUrl) {
    throw new BackendConfigurationError();
  }

  let baseUrl: URL;
  try {
    baseUrl = new URL(rawBaseUrl);
  } catch {
    throw new BackendConfigurationError();
  }

  if (
    !["http:", "https:"].includes(baseUrl.protocol) ||
    baseUrl.username ||
    baseUrl.password
  ) {
    throw new BackendConfigurationError();
  }

  baseUrl.pathname = baseUrl.pathname.replace(/\/+$/, "");
  baseUrl.search = "";
  baseUrl.hash = "";
  return `${baseUrl.toString().replace(/\/+$/, "")}${endpoint}`;
}

export async function fetchBackendJson(
  endpoint: `/${string}`,
  init: RequestInit,
): Promise<BackendJsonResponse> {
  const response = await fetch(buildBackendUrl(endpoint), {
    ...init,
    cache: "no-store",
    headers: {
      Accept: "application/json",
      ...init.headers,
    },
    signal: AbortSignal.timeout(BACKEND_TIMEOUT_MS),
  });

  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new InvalidBackendResponseError();
  }

  return {
    body,
    ok: response.ok,
    status: response.status,
  };
}

export function jsonResponse(body: unknown, status: number): Response {
  return Response.json(body, {
    status,
    headers: {
      "Cache-Control": "no-store",
    },
  });
}

export function proxyFailureResponse(error: unknown): Response {
  if (error instanceof BackendConfigurationError) {
    return jsonResponse(
      { detail: "Prediction API is not configured." },
      503,
    );
  }

  if (isTimeoutError(error)) {
    return jsonResponse(
      { detail: "Prediction API request timed out." },
      504,
    );
  }

  if (error instanceof InvalidBackendResponseError) {
    return jsonResponse(
      { detail: "Prediction API returned an unexpected response." },
      502,
    );
  }

  return jsonResponse(
    { detail: "Prediction API is temporarily unreachable." },
    502,
  );
}

function isTimeoutError(error: unknown): boolean {
  if (!error || typeof error !== "object" || !("name" in error)) {
    return false;
  }
  return error.name === "AbortError" || error.name === "TimeoutError";
}
