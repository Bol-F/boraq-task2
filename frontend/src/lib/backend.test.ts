// @vitest-environment node

import { afterEach, describe, expect, it, vi } from "vitest";

import { BackendConfigurationError, buildBackendUrl } from "./backend";

describe("buildBackendUrl", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("normalizes whitespace and duplicate trailing slashes", () => {
    expect(
      buildBackendUrl(
        "/api/predict/",
        "  https://churn-api.example.com///  ",
      ),
    ).toBe("https://churn-api.example.com/api/predict/");
  });

  it("reads the server-only Render URL at request time", () => {
    vi.stubEnv("RENDER_API_URL", "http://127.0.0.1:8000/");

    expect(buildBackendUrl("/api/health/")).toBe(
      "http://127.0.0.1:8000/api/health/",
    );
  });

  it.each([undefined, "", "ftp://example.com", "not a URL"])(
    "rejects missing or unsafe backend configuration: %s",
    (configuredUrl) => {
      expect(() =>
        buildBackendUrl("/api/health/", configuredUrl),
      ).toThrow(BackendConfigurationError);
    },
  );
});
