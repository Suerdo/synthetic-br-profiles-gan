import type React from "react";

interface CalloutProps {
  children: React.ReactNode;
  tone?: "warning" | "info";
}

export function Callout({ children, tone = "warning" }: CalloutProps) {
  const classes =
    tone === "warning"
      ? "border-orange-300 bg-orange-50 text-orange-800"
      : "border-blue-200 bg-blue-50 text-blue-900";
  return <div className={`rounded-xl border p-4 text-sm leading-6 ${classes}`}>{children}</div>;
}
