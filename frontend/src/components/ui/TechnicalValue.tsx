import { Check, Copy } from "lucide-react";
import { useState } from "react";

interface TechnicalValueProps {
  value: unknown;
  source?: string | null;
  className?: string;
  copyable?: boolean;
  maxLength?: number;
}

export function TechnicalValue({ value, source, className = "", copyable = true, maxLength = 24 }: TechnicalValueProps) {
  const [copied, setCopied] = useState(false);
  const full = stringifyValue(value);
  const short = shortenMiddle(full, maxLength);
  const hasValue = full !== "Não avaliado";

  async function copyValue() {
    if (!hasValue) return;
    await navigator.clipboard?.writeText(full);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }

  return (
    <span className={`block min-w-0 ${className}`}>
      <span className="flex min-w-0 items-center gap-2">
        <code
          className="min-w-0 truncate rounded-md bg-slate-100 px-2 py-1 font-mono text-xs font-semibold text-slateInk"
          title={full}
          aria-label={hasValue ? `Valor completo: ${full}` : undefined}
        >
          {short}
        </code>
        {copyable && hasValue ? (
          <button
            type="button"
            className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md border border-borderSoft bg-white text-slate-600 transition hover:border-blueAction hover:text-blueAction focus:outline-none focus:ring-2 focus:ring-blue-200"
            onClick={copyValue}
            aria-label={`Copiar valor completo ${full}`}
            title="Copiar valor completo"
          >
            {copied ? <Check className="h-3.5 w-3.5" aria-hidden="true" /> : <Copy className="h-3.5 w-3.5" aria-hidden="true" />}
          </button>
        ) : null}
      </span>
      {source ? <span className="mt-1 block text-xs font-normal text-slate-500">Fonte: {source}</span> : null}
    </span>
  );
}

export function isTechnicalValue(label: string, value: unknown): boolean {
  const normalized = label.toLowerCase();
  if (normalized.includes("checksum") || normalized.includes("hash") || normalized.includes("artefato")) return true;
  if (normalized.includes("benchmark") || normalized.includes("commit")) return true;
  if (typeof value !== "string") return false;
  return value.length > 28 && (/^[a-f0-9]{24,}$/i.test(value) || value.includes("/") || value.includes("-"));
}

function stringifyValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "Não avaliado";
  if (Array.isArray(value)) return value.length ? value.map(stringifyValue).join(", ") : "Não avaliado";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function shortenMiddle(value: string, maxLength: number): string {
  if (value.length <= maxLength) return value;
  const edge = Math.max(6, Math.floor((maxLength - 1) / 2));
  return `${value.slice(0, edge)}…${value.slice(-edge)}`;
}
