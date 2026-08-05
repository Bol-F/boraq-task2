import type { PredictionResponse } from "@/lib/types";

const RISK_COPY = {
  low: "This profile shows relatively stable retention signals.",
  medium: "This profile may benefit from a proactive account review.",
  high: "This profile shows elevated churn signals and may need attention.",
} as const;

interface PredictionResultProps {
  prediction: PredictionResponse;
}

export function PredictionResult({ prediction }: PredictionResultProps) {
  const percentage = (prediction.churn_probability * 100).toFixed(1);
  const riskLabel = `${prediction.risk[0].toUpperCase()}${prediction.risk.slice(1)}`;

  return (
    <article
      className={`prediction-result prediction-result--${prediction.risk}`}
      aria-labelledby="prediction-result-heading"
      aria-live="polite"
    >
      <div className="prediction-result__topline">
        <p className="eyebrow">Assessment complete</p>
        <span className="risk-badge">{riskLabel} risk</span>
      </div>

      <div className="probability-gauge" aria-hidden="true">
        <span style={{ width: `${percentage}%` }} />
      </div>

      <p className="prediction-result__probability">
        <strong>{percentage}%</strong>
        <span>churn probability</span>
      </p>

      <h2 id="prediction-result-heading">
        {prediction.will_churn ? "Likely to churn" : "Unlikely to churn"}
      </h2>
      <p className="prediction-result__summary">
        {RISK_COPY[prediction.risk]}
      </p>

      <dl className="result-details">
        <div>
          <dt>Boolean outcome</dt>
          <dd>{String(prediction.will_churn)}</dd>
        </div>
        <div>
          <dt>Risk category</dt>
          <dd>{prediction.risk}</dd>
        </div>
        <div>
          <dt>Model version</dt>
          <dd>{prediction.model_version}</dd>
        </div>
      </dl>

      <p className="prediction-result__note">
        Use this estimate as a decision-support signal, not as the sole basis
        for a customer action.
      </p>
    </article>
  );
}
