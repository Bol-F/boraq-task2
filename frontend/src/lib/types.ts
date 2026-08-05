export type YesNo = "No" | "Yes";

export type Gender = "Female" | "Male";

export type MultipleLines = "No" | "No phone service" | "Yes";

export type InternetService = "DSL" | "Fiber optic" | "No";

export type InternetAddon = "No" | "No internet service" | "Yes";

export type Contract = "Month-to-month" | "One year" | "Two year";

export type PaymentMethod =
  | "Bank transfer (automatic)"
  | "Credit card (automatic)"
  | "Electronic check"
  | "Mailed check";

export interface CustomerPayload {
  gender: Gender;
  SeniorCitizen: 0 | 1;
  Partner: YesNo;
  Dependents: YesNo;
  tenure: number;
  PhoneService: YesNo;
  MultipleLines: MultipleLines;
  InternetService: InternetService;
  OnlineSecurity: InternetAddon;
  OnlineBackup: InternetAddon;
  DeviceProtection: InternetAddon;
  TechSupport: InternetAddon;
  StreamingTV: InternetAddon;
  StreamingMovies: InternetAddon;
  Contract: Contract;
  PaperlessBilling: YesNo;
  PaymentMethod: PaymentMethod;
  MonthlyCharges: number;
  TotalCharges: number;
}

export type CustomerFieldName = keyof CustomerPayload;

export type CustomerFormValues = {
  [FieldName in CustomerFieldName]: string;
};

export type CustomerFieldErrors = Partial<
  Record<CustomerFieldName, string>
>;

export type RiskLevel = "low" | "medium" | "high";

export interface PredictionResponse {
  churn_probability: number;
  will_churn: boolean;
  risk: RiskLevel;
  model_version: string;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  service: string;
  model_loaded: boolean;
  model_version: string | null;
}
