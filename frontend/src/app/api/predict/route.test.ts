// @vitest-environment node

import { afterEach, describe, expect, it, vi } from "vitest";

import { POST } from "./route";

function predictionRequest(body = "{}"): Request {
  return new Request("http://localhost/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
  });
}

function backendJson(body: unknown, status: number): Response {
  return Response.json(body, { status });
}

describe("POST /api/predict", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("returns 503 when RENDER_API_URL is missing", async () => {
    vi.stubEnv("RENDER_API_URL", "");
    const fetchMock = vi.fn<typeof fetch>();
    vi.stubGlobal("fetch", fetchMock);

    const response = await POST(predictionRequest());

    expect(response.status).toBe(503);
    await expect(response.json()).resolves.toEqual({
      detail: "Prediction API is not configured.",
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("forwards JSON and preserves backend validation errors", async () => {
    vi.stubEnv("RENDER_API_URL", "https://api.example.com///");
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(
        backendJson({ tenure: ["Ensure this value is less than or equal to 72."] }, 400),
      );
    vi.stubGlobal("fetch", fetchMock);

    const response = await POST(predictionRequest('{"tenure":73}'));

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual({
      tenure: ["Ensure this value is less than or equal to 72."],
    });
    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, requestInit] = fetchMock.mock.calls[0];
    expect(url).toBe("https://api.example.com/api/predict/");
    expect(requestInit).toMatchObject({
      method: "POST",
      body: '{"tenure":73}',
      cache: "no-store",
    });
  });

  it("returns a validated successful prediction", async () => {
    vi.stubEnv("RENDER_API_URL", "https://api.example.com");
    const prediction = {
      churn_probability: 0.899,
      will_churn: true,
      risk: "high",
      model_version: "1.0.0",
    };
    vi.stubGlobal(
      "fetch",
      vi.fn<typeof fetch>().mockResolvedValue(backendJson(prediction, 200)),
    );

    const response = await POST(predictionRequest());

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual(prediction);
    expect(response.headers.get("Cache-Control")).toBe("no-store");
  });

  it("returns a safe 503 when the backend model is unavailable", async () => {
    vi.stubEnv("RENDER_API_URL", "https://api.example.com");
    vi.stubGlobal(
      "fetch",
      vi
        .fn<typeof fetch>()
        .mockResolvedValue(
          backendJson({ detail: "internal/path/model.pkl failed" }, 503),
        ),
    );

    const response = await POST(predictionRequest());

    expect(response.status).toBe(503);
    await expect(response.json()).resolves.toEqual({
      detail: "Prediction model is not available.",
    });
  });

  it.each([
    ["TimeoutError", 504, "Prediction API request timed out."],
    ["TypeError", 502, "Prediction API is temporarily unreachable."],
  ])(
    "handles %s failures without exposing internal details",
    async (errorName, expectedStatus, expectedDetail) => {
      vi.stubEnv("RENDER_API_URL", "https://api.example.com");
      const error = Object.assign(new Error("private backend details"), {
        name: errorName,
      });
      vi.stubGlobal(
        "fetch",
        vi.fn<typeof fetch>().mockRejectedValue(error),
      );

      const response = await POST(predictionRequest());

      expect(response.status).toBe(expectedStatus);
      await expect(response.json()).resolves.toEqual({
        detail: expectedDetail,
      });
    },
  );

  it("rejects malformed browser JSON before calling the backend", async () => {
    vi.stubEnv("RENDER_API_URL", "https://api.example.com");
    const fetchMock = vi.fn<typeof fetch>();
    vi.stubGlobal("fetch", fetchMock);

    const response = await POST(predictionRequest("{"));

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual({
      detail: "Request body must be valid JSON.",
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects unreadable successful backend responses", async () => {
    vi.stubEnv("RENDER_API_URL", "https://api.example.com");
    vi.stubGlobal(
      "fetch",
      vi.fn<typeof fetch>().mockResolvedValue(
        new Response("not json", {
          status: 200,
          headers: { "Content-Type": "text/plain" },
        }),
      ),
    );

    const response = await POST(predictionRequest());

    expect(response.status).toBe(502);
    await expect(response.json()).resolves.toEqual({
      detail: "Prediction API returned an unexpected response.",
    });
  });
});
