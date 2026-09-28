import type { MetricRange } from "../types/api";

export function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined) return "Não avaliado";
  return new Intl.NumberFormat("pt-BR").format(value);
}

export function formatRate(value: number | MetricRange | null | undefined): string {
  if (value === null || value === undefined) return "Não avaliado";
  if (isMetricRange(value)) {
    return `${formatPercent(value.min)} a ${formatPercent(value.max)}`;
  }
  return formatPercent(value);
}

export function formatUnknown(value: unknown): string {
  if (value === null || value === undefined || value === "") return "Não avaliado";
  if (typeof value === "number") return formatNumber(value);
  if (typeof value === "boolean") return value ? "Sim" : "Não";
  if (Array.isArray(value)) return value.length ? value.map(formatUnknown).join(", ") : "Não avaliado";
  if (isMetricRange(value)) return formatRate(value);
  return String(value);
}

export function formatBytes(value: number | null | undefined): string {
  if (value === null || value === undefined) return "Não avaliado";
  if (value < 1024) return `${value} B`;
  const kb = value / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  return `${mb.toFixed(2)} MB`;
}

export function formatDate(value: string | null | undefined): string {
  if (!value) return "Não avaliado";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return new Intl.DateTimeFormat("pt-BR", { dateStyle: "short", timeStyle: "short" }).format(parsed);
}

export function safeFilename(name: string): string {
  return name.replace(/[^a-z0-9._-]+/gi, "-").toLowerCase();
}

function formatPercent(value: number): string {
  return new Intl.NumberFormat("pt-BR", {
    style: "percent",
    minimumFractionDigits: 0,
    maximumFractionDigits: 2
  }).format(value);
}

function isMetricRange(value: unknown): value is MetricRange {
  return (
    typeof value === "object" &&
    value !== null &&
    "min" in value &&
    "max" in value &&
    typeof (value as MetricRange).min === "number" &&
    typeof (value as MetricRange).max === "number"
  );
}
