// @vitest-environment node

import { afterEach, describe, expect, it, vi } from "vitest";

import { GET } from "./route";

function backendJson(body: unknown, status: number): Response {
  return Response.json(body, { status });
}

describe("GET /api/health", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("preserves a ready backend health response", async () => {
    vi.stubEnv("RENDER_API_URL", "https://api.example.com/");
    const health = {
      status: "ok",
      service: "churn-prediction-api",
      model_loaded: true,
      model_version: "1.0.0",
    };
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(backendJson(health, 200));
    vi.stubGlobal("fetch", fetchMock);

    const response = await GET();

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual(health);
    expect(fetchMock).toHaveBeenCalledWith(
      "https://api.example.com/api/health/",
      expect.objectContaining({ method: "GET", cache: "no-store" }),
    );
  });

  it("preserves a degraded backend health response", async () => {
    vi.stubEnv("RENDER_API_URL", "https://api.example.com");
    const health = {
      status: "degraded",
      service: "churn-prediction-api",
      model_loaded: false,
      model_version: null,
    };
    vi.stubGlobal(
      "fetch",
      vi.fn<typeof fetch>().mockResolvedValue(backendJson(health, 503)),
    );

    const response = await GET();

    expect(response.status).toBe(503);
    await expect(response.json()).resolves.toEqual(health);
  });

  it("does not pass through malformed backend health data", async () => {
    vi.stubEnv("RENDER_API_URL", "https://api.example.com");
    vi.stubGlobal(
      "fetch",
      vi
        .fn<typeof fetch>()
        .mockResolvedValue(
          backendJson({ detail: "C:\\private\\model.pkl" }, 503),
        ),
    );

    const response = await GET();

    expect(response.status).toBe(503);
    await expect(response.json()).resolves.toEqual({
      detail: "Prediction API health check failed.",
    });
  });
});
