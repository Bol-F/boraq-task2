import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { PredictionResult } from "@/components/PredictionResult";
import type { PredictionResponse, RiskLevel } from "@/lib/types";

afterEach(cleanup);

describe("PredictionResult", () => {
  it.each([
    ["low", 0.12],
    ["medium", 0.5],
    ["high", 0.81],
  ] satisfies [RiskLevel, number][])(
    "renders a %s-risk prediction",
    (risk, churnProbability) => {
      const prediction: PredictionResponse = {
        churn_probability: churnProbability,
        will_churn: risk === "high",
        risk,
        model_version: "1.0.0",
      };

      render(<PredictionResult prediction={prediction} />);

      expect(screen.getByText(`${risk[0].toUpperCase()}${risk.slice(1)} risk`)).toBeVisible();
      expect(
        screen.getByText(`${(churnProbability * 100).toFixed(1)}%`),
      ).toBeVisible();
      expect(screen.getByText("1.0.0")).toBeVisible();
      expect(
        screen.getByRole("heading", {
          name: risk === "high" ? "Likely to churn" : "Unlikely to churn",
        }),
      ).toBeVisible();
    },
  );
});
