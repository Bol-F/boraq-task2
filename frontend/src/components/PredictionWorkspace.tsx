"use client";

import { useState, type FormEvent } from "react";

import { ApiStatus } from "@/components/ApiStatus";
import { CustomerForm } from "@/components/CustomerForm";
import { PredictionResult } from "@/components/PredictionResult";
import { ApiRequestError, predictCustomer } from "@/lib/api";
import {
  buildPredictionPayload,
  DEFAULT_CUSTOMER_FORM,
  updateCustomerFormValue,
  validateCustomerForm,
} from "@/lib/schema";
import type {
  CustomerFieldErrors,
  CustomerFieldName,
  CustomerFormValues,
  PredictionResponse,
} from "@/lib/types";

export function PredictionWorkspace() {
  const [values, setValues] = useState<CustomerFormValues>({
    ...DEFAULT_CUSTOMER_FORM,
  });
  const [errors, setErrors] = useState<CustomerFieldErrors>({});
  const [requestError, setRequestError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  function handleChange(name: CustomerFieldName, value: string) {
    setValues((current) => updateCustomerFormValue(current, name, value));
    setErrors((current) => ({ ...current, [name]: undefined }));
    setRequestError(null);
  }

  function handleReset() {
    setValues({ ...DEFAULT_CUSTOMER_FORM });
    setErrors({});
    setRequestError(null);
    setPrediction(null);
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const validationErrors = validateCustomerForm(values);

    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
      setRequestError("Review the highlighted fields and try again.");
      setPrediction(null);
      return;
    }

    setErrors({});
    setRequestError(null);
    setPrediction(null);
    setIsSubmitting(true);

    try {
      const response = await predictCustomer(buildPredictionPayload(values));
      setPrediction(response);
    } catch (error) {
      if (error instanceof ApiRequestError) {
        setErrors(error.fieldErrors);
        setRequestError(error.message);
      } else {
        setRequestError("The prediction could not be completed. Try again.");
      }
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <section className="workspace" aria-label="Churn prediction workspace">
      <CustomerForm
        values={values}
        errors={errors}
        isSubmitting={isSubmitting}
        onChange={handleChange}
        onReset={handleReset}
        onSubmit={handleSubmit}
      />

      <aside className="results-panel" aria-label="Prediction results">
        <ApiStatus />

        {requestError ? (
          <div className="request-error" role="alert">
            <span aria-hidden="true">!</span>
            <div>
              <strong>Prediction not available</strong>
              <p>{requestError}</p>
            </div>
          </div>
        ) : null}

        {isSubmitting ? (
          <div className="result-placeholder result-placeholder--loading" role="status">
            <span className="result-placeholder__pulse" aria-hidden="true" />
            <p className="eyebrow">Running model</p>
            <h2>Analyzing 19 customer signals…</h2>
            <p>The saved pipeline is calculating churn probability.</p>
          </div>
        ) : prediction ? (
          <PredictionResult prediction={prediction} />
        ) : (
          <div className="result-placeholder">
            <div className="result-placeholder__mark" aria-hidden="true">
              <span />
              <span />
              <span />
            </div>
            <p className="eyebrow">Ready when you are</p>
            <h2>Your assessment will appear here</h2>
            <p>
              Review the example profile, adjust the customer details, then
              calculate the churn risk.
            </p>
          </div>
        )}

        <div className="model-context">
          <p className="eyebrow">How to read the score</p>
          <ul>
            <li>
              <span className="legend-dot legend-dot--low" />
              <span>
                <strong>Low</strong> below 35%
              </span>
            </li>
            <li>
              <span className="legend-dot legend-dot--medium" />
              <span>
                <strong>Medium</strong> 35% to 64.9%
              </span>
            </li>
            <li>
              <span className="legend-dot legend-dot--high" />
              <span>
                <strong>High</strong> 65% or above
              </span>
            </li>
          </ul>
        </div>
      </aside>
    </section>
  );
}
