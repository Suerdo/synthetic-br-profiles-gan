export function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined) return "Não avaliado";
  return new Intl.NumberFormat("pt-BR").format(value);
}

export function formatBytes(value: number | null | undefined): string {
  if (value === null || value === undefined) return "Não informado";
  if (value < 1024) return `${value} B`;
  const kb = value / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  return `${mb.toFixed(2)} MB`;
}

export function formatDate(value: string | null | undefined): string {
  if (!value) return "Sem data";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return new Intl.DateTimeFormat("pt-BR", { dateStyle: "short", timeStyle: "short" }).format(parsed);
}

export function safeFilename(name: string): string {
  return name.replace(/[^a-z0-9._-]+/gi, "-").toLowerCase();
}
