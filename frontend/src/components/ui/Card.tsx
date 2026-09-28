import type React from "react";

interface CardProps {
  children: React.ReactNode;
  className?: string;
}

export function Card({ children, className = "" }: CardProps) {
  return <section className={`rounded-xl border border-borderSoft bg-white p-5 shadow-card ${className}`}>{children}</section>;
}

export function SectionHeader({ step, title, children }: { step?: number; title: string; children?: React.ReactNode }) {
  return (
    <div className="mb-4 rounded-xl border border-blue-200 bg-blue-50 p-4 shadow-sm">
      <div className="flex items-start gap-3">
        {step !== undefined ? (
          <span className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-navy text-sm font-bold text-white">
            {step}
          </span>
        ) : null}
        <div>
          <h2 className="text-lg font-bold text-navy">{title}</h2>
          {children ? <p className="mt-1 max-w-3xl text-sm leading-6 text-slate-600">{children}</p> : null}
        </div>
      </div>
    </div>
  );
}
