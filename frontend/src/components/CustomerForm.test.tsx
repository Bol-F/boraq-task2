import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { CustomerForm } from "@/components/CustomerForm";
import {
  DEFAULT_CUSTOMER_FORM,
  updateCustomerFormValue,
} from "@/lib/schema";

afterEach(cleanup);

describe("CustomerForm", () => {
  it("renders all 19 required fields in four accessible sections", () => {
    render(
      <CustomerForm
        values={{ ...DEFAULT_CUSTOMER_FORM }}
        errors={{}}
        isSubmitting={false}
        onChange={vi.fn()}
        onReset={vi.fn()}
        onSubmit={vi.fn()}
      />,
    );

    expect(screen.getAllByRole("group")).toHaveLength(4);
    expect(screen.getAllByRole("combobox")).toHaveLength(16);
    expect(screen.getAllByRole("spinbutton")).toHaveLength(3);
    for (const control of [
      ...screen.getAllByRole("combobox"),
      ...screen.getAllByRole("spinbutton"),
    ]) {
      expect(control).toBeRequired();
    }
  });

  it("locks dependent controls to dataset special values", () => {
    const values = updateCustomerFormValue(
      updateCustomerFormValue(
        DEFAULT_CUSTOMER_FORM,
        "PhoneService",
        "No",
      ),
      "InternetService",
      "No",
    );

    render(
      <CustomerForm
        values={values}
        errors={{}}
        isSubmitting={false}
        onChange={vi.fn()}
        onReset={vi.fn()}
        onSubmit={vi.fn()}
      />,
    );

    expect(screen.getByLabelText("Multiple lines")).toBeDisabled();
    expect(screen.getByLabelText("Online security")).toBeDisabled();
    expect(screen.getByLabelText("Streaming movies")).toBeDisabled();
  });
});
