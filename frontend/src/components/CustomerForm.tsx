import type { FormEvent } from "react";

import {
  FORM_SECTIONS,
  getAvailableOptions,
  isDependentFieldLocked,
  type FormFieldDefinition,
} from "@/lib/schema";
import type {
  CustomerFieldErrors,
  CustomerFieldName,
  CustomerFormValues,
} from "@/lib/types";

interface CustomerFormProps {
  errors: CustomerFieldErrors;
  isSubmitting: boolean;
  onChange: (name: CustomerFieldName, value: string) => void;
  onReset: () => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  values: CustomerFormValues;
}

interface FieldProps {
  error?: string;
  field: FormFieldDefinition;
  onChange: (name: CustomerFieldName, value: string) => void;
  values: CustomerFormValues;
}

function CustomerField({ error, field, onChange, values }: FieldProps) {
  const inputId = `customer-${field.name}`;
  const hintId = `${inputId}-hint`;
  const errorId = `${inputId}-error`;
  const describedBy = error ? `${hintId} ${errorId}` : hintId;
  const className = error ? "form-control form-control--error" : "form-control";
  const isLocked = isDependentFieldLocked(field.name, values);

  return (
    <div className={`form-field form-field--${field.name}`}>
      <label htmlFor={inputId}>{field.label}</label>
      {field.kind === "select" ? (
        <div className="select-wrapper">
          <select
            id={inputId}
            name={field.name}
            className={className}
            value={values[field.name]}
            onChange={(event) => onChange(field.name, event.target.value)}
            aria-describedby={describedBy}
            aria-invalid={error ? "true" : undefined}
            disabled={isLocked}
            required
          >
            {getAvailableOptions(field, values).map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          <span className="select-wrapper__icon" aria-hidden="true">
            ↓
          </span>
        </div>
      ) : (
        <div className="number-wrapper">
          {field.name === "MonthlyCharges" || field.name === "TotalCharges" ? (
            <span className="number-wrapper__prefix" aria-hidden="true">
              $
            </span>
          ) : null}
          <input
            id={inputId}
            name={field.name}
            type="number"
            className={className}
            value={values[field.name]}
            onChange={(event) => onChange(field.name, event.target.value)}
            aria-describedby={describedBy}
            aria-invalid={error ? "true" : undefined}
            inputMode={field.inputMode}
            min={field.min}
            max={field.max}
            step={field.step}
            required
          />
        </div>
      )}
      <p className="field-hint" id={hintId}>
        {field.hint}
      </p>
      {error ? (
        <p className="field-error" id={errorId}>
          {error}
        </p>
      ) : null}
    </div>
  );
}

export function CustomerForm({
  errors,
  isSubmitting,
  onChange,
  onReset,
  onSubmit,
  values,
}: CustomerFormProps) {
  return (
    <form
      className="customer-form"
      onSubmit={onSubmit}
      aria-busy={isSubmitting}
      noValidate
    >
      <div className="form-intro">
        <div>
          <p className="eyebrow">Customer assessment</p>
          <h2>Build a churn profile</h2>
        </div>
        <p>
          Complete all 19 model inputs. No customer data is stored by this
          interface.
        </p>
      </div>

      {FORM_SECTIONS.map((section, index) => (
        <fieldset className="form-section" key={section.id}>
          <legend className="form-section__heading">
            <span className="form-section__number" aria-hidden="true">
              {String(index + 1).padStart(2, "0")}
            </span>
            <span>
              <span className="form-section__title">{section.title}</span>
              <span className="form-section__description">
                {section.description}
              </span>
            </span>
          </legend>
          <div className="form-grid">
            {section.fields.map((field) => (
              <CustomerField
                key={field.name}
                field={field}
                values={values}
                error={errors[field.name]}
                onChange={onChange}
              />
            ))}
          </div>
        </fieldset>
      ))}

      <div className="form-actions">
        <button
          className="button button--secondary"
          type="button"
          onClick={onReset}
          disabled={isSubmitting}
        >
          Reset example
        </button>
        <button
          className="button button--primary"
          type="submit"
          disabled={isSubmitting}
        >
          {isSubmitting ? (
            <>
              <span className="spinner" aria-hidden="true" />
              Calculating risk…
            </>
          ) : (
            <>
              Calculate churn risk
              <span aria-hidden="true">→</span>
            </>
          )}
        </button>
      </div>
    </form>
  );
}
