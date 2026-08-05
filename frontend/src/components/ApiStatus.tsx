import { useEffect, useState } from "react";

import { fetchHealth } from "@/lib/api";
import type { HealthResponse } from "@/lib/types";

type StatusState =
  | { kind: "checking" }
  | { health: HealthResponse; kind: "ready" | "degraded" }
  | { kind: "unavailable" };

export function ApiStatus() {
  const [state, setState] = useState<StatusState>({ kind: "checking" });

  useEffect(() => {
    const controller = new AbortController();

    async function checkHealth() {
      try {
        const health = await fetchHealth(controller.signal);
        if (!controller.signal.aborted) {
          setState({
            health,
            kind:
              health.status === "ok" && health.model_loaded
                ? "ready"
                : "degraded",
          });
        }
      } catch {
        if (!controller.signal.aborted) {
          setState({ kind: "unavailable" });
        }
      }
    }

    void checkHealth();
    return () => controller.abort();
  }, []);

  const copy = {
    checking: {
      label: "Checking API",
      message: "Confirming model readiness…",
    },
    degraded: {
      label: "Model unavailable",
      message: "The API is online, but predictions are temporarily paused.",
    },
    ready: {
      label: "Model ready",
      message: "The prediction API is online and accepting requests.",
    },
    unavailable: {
      label: "API unavailable",
      message: "The service could not be reached. Try again shortly.",
    },
  }[state.kind];

  const modelVersion =
    "health" in state && state.health.model_version
      ? `Model ${state.health.model_version}`
      : null;

  return (
    <div
      className={`api-status api-status--${state.kind}`}
      role="status"
      aria-live="polite"
      aria-atomic="true"
    >
      <span className="api-status__indicator" aria-hidden="true" />
      <span className="api-status__copy">
        <strong>{copy.label}</strong>
        <span>{copy.message}</span>
      </span>
      {modelVersion ? (
        <span className="api-status__version">{modelVersion}</span>
      ) : null}
    </div>
  );
}
