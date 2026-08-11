import type {
  CustomerFieldErrors,
  CustomerFieldName,
  CustomerFormValues,
  CustomerPayload,
} from "@/lib/types";

export interface FieldOption {
  label: string;
  value: string;
}

interface BaseFieldDefinition {
  hint: string;
  label: string;
  name: CustomerFieldName;
}

export interface SelectFieldDefinition extends BaseFieldDefinition {
  kind: "select";
  options: readonly FieldOption[];
}

export interface NumberFieldDefinition extends BaseFieldDefinition {
  inputMode: "decimal" | "numeric";
  kind: "number";
  max?: number;
  min: number;
  step: number;
}

export type FormFieldDefinition =
  | NumberFieldDefinition
  | SelectFieldDefinition;

export interface FormSectionDefinition {
  description: string;
  fields: readonly FormFieldDefinition[];
  id: string;
  title: string;
}

function choices(values: readonly string[]): readonly FieldOption[] {
  return values.map((value) => ({ label: value, value }));
}

export const GENDER_CHOICES = ["Female", "Male"] as const;
export const YES_NO_CHOICES = ["No", "Yes"] as const;
export const MULTIPLE_LINES_CHOICES = [
  "No",
  "No phone service",
  "Yes",
] as const;
export const INTERNET_SERVICE_CHOICES = ["DSL", "Fiber optic", "No"] as const;
export const INTERNET_ADDON_CHOICES = [
  "No",
  "No internet service",
  "Yes",
] as const;
export const CONTRACT_CHOICES = [
  "Month-to-month",
  "One year",
  "Two year",
] as const;
export const PAYMENT_METHOD_CHOICES = [
  "Bank transfer (automatic)",
  "Credit card (automatic)",
  "Electronic check",
  "Mailed check",
] as const;

export const INTERNET_ADDON_FIELDS = [
  "OnlineSecurity",
  "OnlineBackup",
  "DeviceProtection",
  "TechSupport",
  "StreamingTV",
  "StreamingMovies",
] as const satisfies readonly CustomerFieldName[];

export const MODEL_FIELD_NAMES = [
  "gender",
  "SeniorCitizen",
  "Partner",
  "Dependents",
  "tenure",
  "PhoneService",
  "MultipleLines",
  "InternetService",
  "OnlineSecurity",
  "OnlineBackup",
  "DeviceProtection",
  "TechSupport",
  "StreamingTV",
  "StreamingMovies",
  "Contract",
  "PaperlessBilling",
  "PaymentMethod",
  "MonthlyCharges",
  "TotalCharges",
] as const satisfies readonly CustomerFieldName[];

const seniorCitizenOptions: readonly FieldOption[] = [
  { label: "No", value: "0" },
  { label: "Yes", value: "1" },
];

export const FORM_SECTIONS: readonly FormSectionDefinition[] = [
  {
    id: "customer-information",
    title: "Customer information",
    description: "Basic household details recorded on the account.",
    fields: [
      {
        kind: "select",
        name: "gender",
        label: "Gender",
        hint: "Gender recorded on the account.",
        options: choices(GENDER_CHOICES),
      },
      {
        kind: "select",
        name: "SeniorCitizen",
        label: "Senior citizen",
        hint: "Whether the customer is 65 or older.",
        options: seniorCitizenOptions,
      },
      {
        kind: "select",
        name: "Partner",
        label: "Partner",
        hint: "Whether the customer has a partner.",
        options: choices(YES_NO_CHOICES),
      },
      {
        kind: "select",
        name: "Dependents",
        label: "Dependents",
        hint: "Whether the customer has dependents.",
        options: choices(YES_NO_CHOICES),
      },
    ],
  },
  {
    id: "account-information",
    title: "Account information",
    description: "Customer history and current contract preferences.",
    fields: [
      {
        kind: "number",
        name: "tenure",
        label: "Tenure",
        hint: "Number of months with the company (0–72).",
        inputMode: "numeric",
        min: 0,
        max: 72,
        step: 1,
      },
      {
        kind: "select",
        name: "Contract",
        label: "Contract",
        hint: "The customer's current contract term.",
        options: choices(CONTRACT_CHOICES),
      },
      {
        kind: "select",
        name: "PaperlessBilling",
        label: "Paperless billing",
        hint: "Whether billing statements are paperless.",
        options: choices(YES_NO_CHOICES),
      },
    ],
  },
  {
    id: "phone-and-internet-services",
    title: "Phone & internet services",
    description: "Connectivity, security, support, and entertainment services.",
    fields: [
      {
        kind: "select",
        name: "PhoneService",
        label: "Phone service",
        hint: "Whether phone service is active.",
        options: choices(YES_NO_CHOICES),
      },
      {
        kind: "select",
        name: "MultipleLines",
        label: "Multiple lines",
        hint: "Line configuration, adjusted to match phone service.",
        options: choices(MULTIPLE_LINES_CHOICES),
      },
      {
        kind: "select",
        name: "InternetService",
        label: "Internet service",
        hint: "The account's internet connection type.",
        options: choices(INTERNET_SERVICE_CHOICES),
      },
      {
        kind: "select",
        name: "OnlineSecurity",
        label: "Online security",
        hint: "Online security add-on status.",
        options: choices(INTERNET_ADDON_CHOICES),
      },
      {
        kind: "select",
        name: "OnlineBackup",
        label: "Online backup",
        hint: "Online backup add-on status.",
        options: choices(INTERNET_ADDON_CHOICES),
      },
      {
        kind: "select",
        name: "DeviceProtection",
        label: "Device protection",
        hint: "Device protection add-on status.",
        options: choices(INTERNET_ADDON_CHOICES),
      },
      {
        kind: "select",
        name: "TechSupport",
        label: "Tech support",
        hint: "Technical support add-on status.",
        options: choices(INTERNET_ADDON_CHOICES),
      },
      {
        kind: "select",
        name: "StreamingTV",
        label: "Streaming TV",
        hint: "Streaming television add-on status.",
        options: choices(INTERNET_ADDON_CHOICES),
      },
      {
        kind: "select",
        name: "StreamingMovies",
        label: "Streaming movies",
        hint: "Streaming movie add-on status.",
        options: choices(INTERNET_ADDON_CHOICES),
      },
    ],
  },
  {
    id: "billing-information",
    title: "Billing information",
    description: "Payment method and current account charges.",
    fields: [
      {
        kind: "select",
        name: "PaymentMethod",
        label: "Payment method",
        hint: "How the customer pays each bill.",
        options: choices(PAYMENT_METHOD_CHOICES),
      },
      {
        kind: "number",
        name: "MonthlyCharges",
        label: "Monthly charges",
        hint: "Current monthly bill in dollars.",
        inputMode: "decimal",
        min: 0,
        step: 0.01,
      },
      {
        kind: "number",
        name: "TotalCharges",
        label: "Total charges",
        hint: "Total amount billed to date in dollars.",
        inputMode: "decimal",
        min: 0,
        step: 0.01,
      },
    ],
  },
] as const;

export const DEFAULT_CUSTOMER_FORM: CustomerFormValues = {
  gender: "Female",
  SeniorCitizen: "0",
  Partner: "Yes",
  Dependents: "No",
  tenure: "5",
  PhoneService: "Yes",
  MultipleLines: "No",
  InternetService: "Fiber optic",
  OnlineSecurity: "No",
  OnlineBackup: "No",
  DeviceProtection: "No",
  TechSupport: "No",
  StreamingTV: "Yes",
  StreamingMovies: "Yes",
  Contract: "Month-to-month",
  PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check",
  MonthlyCharges: "89.90",
  TotalCharges: "450.50",
};

function isInternetAddonField(
  name: CustomerFieldName,
): name is (typeof INTERNET_ADDON_FIELDS)[number] {
  return INTERNET_ADDON_FIELDS.some((fieldName) => fieldName === name);
}

export function getAvailableOptions(
  field: SelectFieldDefinition,
  values: CustomerFormValues,
): readonly FieldOption[] {
  if (field.name === "MultipleLines") {
    return values.PhoneService === "No"
      ? field.options.filter(({ value }) => value === "No phone service")
      : field.options.filter(({ value }) => value !== "No phone service");
  }

  if (isInternetAddonField(field.name)) {
    return values.InternetService === "No"
      ? field.options.filter(({ value }) => value === "No internet service")
      : field.options.filter(({ value }) => value !== "No internet service");
  }

  return field.options;
}

export function isDependentFieldLocked(
  name: CustomerFieldName,
  values: CustomerFormValues,
): boolean {
  return (
    (name === "MultipleLines" && values.PhoneService === "No") ||
    (isInternetAddonField(name) && values.InternetService === "No")
  );
}

export function updateCustomerFormValue(
  current: CustomerFormValues,
  name: CustomerFieldName,
  value: string,
): CustomerFormValues {
  const next = { ...current, [name]: value };

  if (name === "PhoneService") {
    if (value === "No") {
      next.MultipleLines = "No phone service";
    } else if (current.MultipleLines === "No phone service") {
      next.MultipleLines = "No";
    }
  }

  if (name === "InternetService") {
    for (const fieldName of INTERNET_ADDON_FIELDS) {
      if (value === "No") {
        next[fieldName] = "No internet service";
      } else if (current[fieldName] === "No internet service") {
        next[fieldName] = "No";
      }
    }
  }

  return next;
}

export function validateCustomerForm(
  values: CustomerFormValues,
): CustomerFieldErrors {
  const errors: CustomerFieldErrors = {};

  for (const fieldName of MODEL_FIELD_NAMES) {
    if (values[fieldName].trim() === "") {
      errors[fieldName] = "This field is required.";
    }
  }

  const validChoices: Partial<
    Record<CustomerFieldName, readonly string[]>
  > = {
    gender: GENDER_CHOICES,
    SeniorCitizen: seniorCitizenOptions.map(({ value }) => value),
    Partner: YES_NO_CHOICES,
    Dependents: YES_NO_CHOICES,
    PhoneService: YES_NO_CHOICES,
    MultipleLines: MULTIPLE_LINES_CHOICES,
    InternetService: INTERNET_SERVICE_CHOICES,
    OnlineSecurity: INTERNET_ADDON_CHOICES,
    OnlineBackup: INTERNET_ADDON_CHOICES,
    DeviceProtection: INTERNET_ADDON_CHOICES,
    TechSupport: INTERNET_ADDON_CHOICES,
    StreamingTV: INTERNET_ADDON_CHOICES,
    StreamingMovies: INTERNET_ADDON_CHOICES,
    Contract: CONTRACT_CHOICES,
    PaperlessBilling: YES_NO_CHOICES,
    PaymentMethod: PAYMENT_METHOD_CHOICES,
  };

  for (const [fieldName, options] of Object.entries(validChoices)) {
    const name = fieldName as CustomerFieldName;
    if (options && !options.includes(values[name])) {
      errors[name] = "Choose one of the available options.";
    }
  }

  const seniorCitizen = Number(values.SeniorCitizen);
  if (![0, 1].includes(seniorCitizen)) {
    errors.SeniorCitizen = "Choose No or Yes.";
  }

  const tenure = Number(values.tenure);
  if (!Number.isInteger(tenure) || tenure < 0 || tenure > 72) {
    errors.tenure = "Enter a whole number from 0 to 72.";
  }

  for (const fieldName of ["MonthlyCharges", "TotalCharges"] as const) {
    const amount = Number(values[fieldName]);
    if (!Number.isFinite(amount) || amount < 0) {
      errors[fieldName] = "Enter a non-negative amount.";
    }
  }

  if (
    values.PhoneService === "No" &&
    values.MultipleLines !== "No phone service"
  ) {
    errors.MultipleLines = "Choose No phone service when phone service is off.";
  }
  if (
    values.PhoneService === "Yes" &&
    values.MultipleLines === "No phone service"
  ) {
    errors.MultipleLines = "Choose No or Yes when phone service is active.";
  }

  for (const fieldName of INTERNET_ADDON_FIELDS) {
    const value = values[fieldName];
    if (values.InternetService === "No" && value !== "No internet service") {
      errors[fieldName] = "Choose No internet service when internet is off.";
    }
    if (values.InternetService !== "No" && value === "No internet service") {
      errors[fieldName] = "Choose No or Yes when internet service is active.";
    }
  }

  return errors;
}

export function buildPredictionPayload(
  values: CustomerFormValues,
): CustomerPayload {
  return {
    gender: values.gender as CustomerPayload["gender"],
    SeniorCitizen: Number(values.SeniorCitizen) as 0 | 1,
    Partner: values.Partner as CustomerPayload["Partner"],
    Dependents: values.Dependents as CustomerPayload["Dependents"],
    tenure: Number(values.tenure),
    PhoneService: values.PhoneService as CustomerPayload["PhoneService"],
    MultipleLines: values.MultipleLines as CustomerPayload["MultipleLines"],
    InternetService:
      values.InternetService as CustomerPayload["InternetService"],
    OnlineSecurity:
      values.OnlineSecurity as CustomerPayload["OnlineSecurity"],
    OnlineBackup: values.OnlineBackup as CustomerPayload["OnlineBackup"],
    DeviceProtection:
      values.DeviceProtection as CustomerPayload["DeviceProtection"],
    TechSupport: values.TechSupport as CustomerPayload["TechSupport"],
    StreamingTV: values.StreamingTV as CustomerPayload["StreamingTV"],
    StreamingMovies:
      values.StreamingMovies as CustomerPayload["StreamingMovies"],
    Contract: values.Contract as CustomerPayload["Contract"],
    PaperlessBilling:
      values.PaperlessBilling as CustomerPayload["PaperlessBilling"],
    PaymentMethod:
      values.PaymentMethod as CustomerPayload["PaymentMethod"],
    MonthlyCharges: Number(values.MonthlyCharges),
    TotalCharges: Number(values.TotalCharges),
  };
}

export const buildCustomerPayload = buildPredictionPayload;
