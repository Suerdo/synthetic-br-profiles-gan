import type { ModelName } from "../types/api";

const modelLabels: Record<ModelName, string> = {
  programmatic: "Programático",
  ctgan: "CTGAN",
  simple_gan: "GAN Simples"
};

const statusLabels: Record<string, string> = {
  approved: "Aprovado",
  recommended: "Recomendado",
  recommended_candidate: "Candidato Recomendado",
  candidate: "Candidato",
  experimental: "Experimental",
  quarantined: "Quarentena",
  quality_quarantined: "Quarentena",
  rejected: "Reprovado",
  completed: "Concluído",
  failed: "Falhou",
  running: "Em Execução",
  queued: "Na Fila",
  not_evaluated: "Não avaliado",
  not_approved: "Não aprovado",
  not_defined: "Não definido",
  default: "Padrão geral"
};

export function modelDisplayName(value: unknown): string {
  if (typeof value !== "string") return String(value ?? "Não avaliado");
  return modelLabels[value as ModelName] ?? value;
}

export function statusDisplayName(value: unknown): string {
  if (value === null || value === undefined || value === "") return "Não avaliado";
  if (typeof value !== "string") return String(value);
  return value
    .split(/\s*·\s*|\s*\/\s*|\s*,\s*/)
    .map((part) => statusLabels[part.trim().toLowerCase()] ?? part.trim())
    .join(" · ");
}
