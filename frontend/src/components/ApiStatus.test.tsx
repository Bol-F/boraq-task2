import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiStatus } from "@/components/ApiStatus";
import { fetchHealth } from "@/lib/api";

vi.mock("@/lib/api", () => ({
  fetchHealth: vi.fn(),
}));

const fetchHealthMock = vi.mocked(fetchHealth);

beforeEach(() => {
  fetchHealthMock.mockReset();
});

afterEach(cleanup);

describe("ApiStatus", () => {
  it("renders a ready model response", async () => {
    fetchHealthMock.mockResolvedValue({
      status: "ok",
      service: "churn-prediction-api",
      model_loaded: true,
      model_version: "1.0.0",
    });

    render(<ApiStatus />);

    expect(await screen.findByText("Model ready")).toBeVisible();
    expect(screen.getByText("Model 1.0.0")).toBeVisible();
  });

  it("renders a degraded model response", async () => {
    fetchHealthMock.mockResolvedValue({
      status: "degraded",
      service: "churn-prediction-api",
      model_loaded: false,
      model_version: null,
    });

    render(<ApiStatus />);

    expect(await screen.findByText("Model unavailable")).toBeVisible();
  });

  it("renders a safe unavailable state after a network error", async () => {
    fetchHealthMock.mockRejectedValue(new Error("private network details"));

    render(<ApiStatus />);

    expect(await screen.findByText("API unavailable")).toBeVisible();
    expect(screen.queryByText("private network details")).not.toBeInTheDocument();
  });
});
