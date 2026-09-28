import type React from "react";

const tones: Record<string, string> = {
  approved: "border-emerald-200 bg-emerald-50 text-emerald-800",
  recommended: "border-blue-200 bg-blue-50 text-blue-800",
  candidate: "border-indigo-200 bg-indigo-50 text-indigo-800",
  experimental: "border-amber-200 bg-amber-50 text-amber-800",
  smoke: "border-slate-300 bg-slate-100 text-slate-700",
  legacy: "border-purple-200 bg-purple-50 text-purple-800",
  neutral: "border-slate-300 bg-slate-100 text-slate-700",
  warning: "border-orange-300 bg-orange-50 text-orange-800"
};

interface BadgeProps {
  children: React.ReactNode;
  tone?: keyof typeof tones;
  title?: string;
}

export function Badge({ children, tone = "neutral", title }: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-semibold ${tones[tone]}`}
      title={title}
    >
      {children}
    </span>
  );
}
