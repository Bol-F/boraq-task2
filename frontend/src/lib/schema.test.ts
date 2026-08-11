import { describe, expect, it } from "vitest";

import {
  buildPredictionPayload,
  DEFAULT_CUSTOMER_FORM,
  INTERNET_ADDON_FIELDS,
  MODEL_FIELD_NAMES,
  updateCustomerFormValue,
  validateCustomerForm,
} from "@/lib/schema";

describe("customer payload schema", () => {
  it("builds the exact 19-field Django payload with numeric values", () => {
    const payload = buildPredictionPayload({ ...DEFAULT_CUSTOMER_FORM });

    expect(Object.keys(payload)).toEqual(MODEL_FIELD_NAMES);
    expect(payload.SeniorCitizen).toBe(0);
    expect(payload.tenure).toBe(5);
    expect(payload.MonthlyCharges).toBe(89.9);
    expect(payload.TotalCharges).toBe(450.5);
  });

  it("reports a missing required field before submission", () => {
    const errors = validateCustomerForm({
      ...DEFAULT_CUSTOMER_FORM,
      MonthlyCharges: "",
    });

    expect(errors.MonthlyCharges).toBe("This field is required.");
  });

  it("keeps service-dependent special values consistent", () => {
    const withoutPhone = updateCustomerFormValue(
      DEFAULT_CUSTOMER_FORM,
      "PhoneService",
      "No",
    );
    const withoutInternet = updateCustomerFormValue(
      withoutPhone,
      "InternetService",
      "No",
    );

    expect(withoutInternet.MultipleLines).toBe("No phone service");
    for (const fieldName of INTERNET_ADDON_FIELDS) {
      expect(withoutInternet[fieldName]).toBe("No internet service");
    }
    expect(validateCustomerForm(withoutInternet)).toEqual({});
  });
});
