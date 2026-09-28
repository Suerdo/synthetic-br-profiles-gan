import { useQuery } from "@tanstack/react-query";
import { ArrowRight, BrainCircuit, Database, FlaskConical } from "lucide-react";
import { useMemo, useState } from "react";

import { getModelArtifacts, getModels } from "../api";
import { Badge } from "../components/ui/Badge";
import { Card } from "../components/ui/Card";
import { isTechnicalValue, TechnicalValue } from "../components/ui/TechnicalValue";
import type { ModelArtifact, ModelEntry, ModelName } from "../types/api";
import { formatBytes, formatDate, formatNumber, formatRate } from "../utils/format";
import { modelDisplayName, statusDisplayName } from "../utils/labels";

const modelOrder: ModelName[] = ["programmatic", "ctgan", "simple_gan"];

export function ModelsPage() {
  const [selectedModel, setSelectedModel] = useState<ModelName>("programmatic");
  const models = useQuery({ queryKey: ["models"], queryFn: getModels, staleTime: 60_000 });
  const entries = useMemo(
    () => modelOrder.map((name) => models.data?.models.find((model) => model.name === name)).filter(Boolean) as ModelEntry[],
    [models.data]
  );
  const selectedEntry = entries.find((entry) => entry.name === selectedModel) ?? entries[0];

  return (
    <div className="mx-auto max-w-7xl space-y-7">
      <header className="max-w-4xl">
        <p className="text-sm font-semibold uppercase tracking-[0.18em] text-blueAction">Modelos</p>
        <h1 className="mt-2 text-3xl font-bold text-slateInk">Modelos</h1>
        <p className="mt-2 max-w-3xl text-base leading-7 text-slate-600">
          Conheça as estratégias disponíveis para geração de perfis sintéticos e suas principais características.
        </p>
      </header>

      {models.isLoading ? <LoadingCard text="Carregando catálogo de modelos..." /> : null}
      {models.isError ? <ErrorCard text="Não foi possível carregar o catálogo de modelos." /> : null}
      {!models.isLoading && !models.isError && entries.length === 0 ? <EmptyCard text="Nenhum modelo foi retornado pela API." /> : null}

      {entries.length > 0 ? (
        <>
          <div className="grid gap-4 lg:grid-cols-3">
            {entries.map((entry) => (
              <ModelCard
                key={entry.name}
                entry={entry}
                active={entry.name === selectedEntry?.name}
                onSelect={() => setSelectedModel(entry.name)}
              />
            ))}
          </div>
          {selectedEntry ? <ModelDetails entry={selectedEntry} /> : null}
        </>
      ) : null}
    </div>
  );
}

function ModelCard({ entry, active, onSelect }: { entry: ModelEntry; active: boolean; onSelect: () => void }) {
  const mainBadge = badgeForModel(entry);
  const Icon = entry.name === "programmatic" ? Database : entry.name === "ctgan" ? BrainCircuit : FlaskConical;
  const version = entry.recommended_artifact
    ? `Vocabulário v${entry.recommended_artifact.vocabulary_version} · Renda v${entry.recommended_artifact.income_model_version} · Geografia v${entry.recommended_artifact.geography_model_version}`
    : entry.requires_saved_artifact
      ? "Artefato neural não disponível neste ambiente"
      : "Geração direta sem artefato";

  return (
    <button
      type="button"
      onClick={onSelect}
      className={`relative min-w-0 rounded-xl border p-5 text-left shadow-card transition hover:border-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-200 ${
        active ? "border-blueAction bg-blue-50/60 ring-2 ring-blue-100" : "border-borderSoft bg-white"
      }`}
    >
      {active ? <span className="absolute inset-y-4 left-0 w-1 rounded-r-full bg-blueAction" aria-hidden="true" /> : null}
      <div className="flex items-start justify-between gap-3">
        <span className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-blue-50 text-blueAction">
          <Icon className="h-5 w-5" aria-hidden="true" />
        </span>
        <Badge tone={mainBadge.tone}>{mainBadge.label}</Badge>
      </div>
      <h2 className="mt-4 text-xl font-bold text-slateInk">{cleanModelName(entry)}</h2>
      <p className="mt-2 line-clamp-3 text-sm leading-6 text-slate-600">{entry.short_description}</p>
      <dl className="mt-4 space-y-2 text-sm">
        <CardLine label="Disponibilidade" value={entry.available ? "Disponível" : "Não disponível"} />
        <CardLine label="Indicação principal" value={entry.recommended_use_cases.slice(0, 2).join(", ") || "Não avaliado"} />
        <CardLine label="Versões" value={version} />
      </dl>
      <span className="mt-4 inline-flex items-center gap-2 text-sm font-semibold text-blueAction">
        Ver Detalhes
        <ArrowRight className="h-4 w-4" aria-hidden="true" />
      </span>
    </button>
  );
}

function ModelDetails({ entry }: { entry: ModelEntry }) {
  const artifacts = useQuery({
    queryKey: ["artifacts", entry.name],
    queryFn: () => getModelArtifacts(entry.name),
    enabled: entry.requires_saved_artifact,
    staleTime: 60_000
  });
  const selectedArtifact =
    entry.recommended_artifact ??
    artifacts.data?.artifacts.find((artifact) => artifact.artifact_id === artifacts.data?.recommended_artifact_id) ??
    artifacts.data?.artifacts[0] ??
    null;

  return (
    <Card className="border-blue-100">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="text-2xl font-bold text-slateInk">{entry.title}</h2>
        <Badge tone={badgeForModel(entry).tone}>{badgeForModel(entry).label}</Badge>
        {selectedArtifact?.recommended_for_neural_generation ? <Badge tone="approved">Artefato Neural Recomendado</Badge> : null}
      </div>
      <p className="mt-3 max-w-4xl text-sm leading-6 text-slate-600">{entry.detailed_description}</p>

      <div className="mt-5 grid items-stretch gap-4 md:grid-cols-2">
        <SummaryCard title="Resumo Simples">{entry.simple_summary || entry.summary}</SummaryCard>
        <SummaryCard title="Resumo Técnico">{entry.technical_summary}</SummaryCard>
      </div>

      <h3 className="mt-6 font-bold text-slateInk">Usos Recomendados</h3>
      <p className="mt-2 text-sm leading-6 text-slate-700">{entry.recommended_for}</p>

      {entry.requires_saved_artifact ? (
        <ArtifactPanel
          model={entry.name}
          artifact={selectedArtifact}
          artifacts={artifacts.data?.artifacts ?? []}
          loading={artifacts.isLoading}
          error={artifacts.isError}
          emptyMessage={entry.availability_message}
        />
      ) : (
        <p className="mt-4 rounded-lg border border-blue-200 bg-blue-50 p-3 text-sm text-blue-900">
          Este modelo não depende de artefato neural salvo e permanece como padrão geral para geração rápida e controlada.
        </p>
      )}

      <div className="mt-5 grid gap-4 lg:grid-cols-2">
        <DetailList title="Ver benefícios" items={entry.benefits} />
        <DetailList title="Ver limitações" items={entry.limitations} />
      </div>
      <DetailList title="Ver notas de governança" items={entry.governance_notes} className="mt-4" />
    </Card>
  );
}

function ArtifactPanel({
  model,
  artifact,
  artifacts,
  loading,
  error,
  emptyMessage
}: {
  model: ModelName;
  artifact: ModelArtifact | null;
  artifacts: ModelArtifact[];
  loading: boolean;
  error: boolean;
  emptyMessage: string | null;
}) {
  if (loading) return <LoadingCard text="Carregando artefatos válidos..." className="mt-5" />;
  if (error) return <ErrorCard text="Não foi possível carregar os artefatos deste modelo." className="mt-5" />;
  if (!artifact) return <EmptyCard text={emptyMessage ?? `Nenhum artefato ${modelDisplayName(model)} válido foi encontrado.`} className="mt-5" />;

  return (
    <div className="mt-5 rounded-xl border border-borderSoft bg-panel p-4">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-lg font-bold text-slateInk">Artefato Selecionado</h3>
        <Badge tone={artifact.approved ? "approved" : artifact.recommended ? "candidate" : "neutral"}>{statusDisplayName(artifact.status)}</Badge>
      </div>
      {artifact.warning ? <p className="mt-3 rounded-lg border border-orange-300 bg-orange-50 p-3 text-sm text-orange-800">{artifact.warning}</p> : null}
      <div className="mt-4 grid gap-3 text-sm md:grid-cols-3">
        <Summary label="Identificador" value={artifact.artifact_id} />
        <Summary label="Criado em" value={formatDate(artifact.created_at_utc)} />
        <Summary label="Status" value={statusDisplayName(artifact.status)} />
        <Summary label="Vocabulário" value={`v${artifact.vocabulary_version}`} />
        <Summary label="Renda" value={`v${artifact.income_model_version}`} />
        <Summary label="Geografia" value={`v${artifact.geography_model_version}`} />
        <Summary label="Treino" value={formatNumber(artifact.train_rows)} />
        <Summary label="Épocas" value={formatNumber(artifact.epochs)} />
        <Summary label="Biblioteca" value={artifact.library ?? "Não avaliado"} />
        <Summary label="Duplicidade-base" value={formatRate(artifact.duplicate_base_row_rate)} />
        <Summary label="Match treino" value={formatRate(artifact.exact_train_match_rate)} />
        <Summary label="Realismo condicional" value={artifact.conditional_income_status ?? "Não avaliado"} />
        <Summary label="Compatibilidade" value={artifact.compatibility} />
        <Summary label="Tamanho" value={formatBytes(artifact.model_size_bytes)} />
        <Summary label="Qualidade" value={artifact.quality_status ?? "Não avaliado"} />
      </div>

      <details className="mt-5 rounded-lg border border-borderSoft bg-white p-4">
        <summary className="cursor-pointer font-semibold text-slateInk">Outros Artefatos Disponíveis</summary>
        <div className="mt-4 grid gap-3">
          {artifacts.map((item) => (
            <div key={item.artifact_id} className="min-w-0 rounded-lg border border-borderSoft bg-panel p-3">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <TechnicalValue value={item.artifact_id} />
                <Badge tone={item.approved ? "approved" : item.recommended ? "candidate" : "neutral"}>{statusDisplayName(item.status)}</Badge>
              </div>
              <div className="mt-3 grid gap-2 text-xs text-slate-600 sm:grid-cols-4">
                <span>Criado em: {formatDate(item.created_at_utc)}</span>
                <span>Vocabulário v{item.vocabulary_version}</span>
                <span>Renda v{item.income_model_version}</span>
                <span>Geografia v{item.geography_model_version}</span>
              </div>
              <details className="mt-2">
                <summary className="cursor-pointer text-xs font-semibold text-blueAction">Ver Detalhes</summary>
                <div className="mt-2 grid gap-2 text-xs sm:grid-cols-3">
                  <Summary label="Treino" value={formatNumber(item.train_rows)} />
                  <Summary label="Duplicidade-base" value={formatRate(item.duplicate_base_row_rate)} />
                  <Summary label="Match treino" value={formatRate(item.exact_train_match_rate)} />
                </div>
              </details>
            </div>
          ))}
        </div>
      </details>
    </div>
  );
}

function SummaryCard({ title, children }: { title: string; children: string }) {
  return (
    <div className="h-full min-h-[190px] rounded-xl border border-borderSoft bg-panel p-4">
      <h3 className="font-bold text-slateInk">{title}</h3>
      <p className="mt-2 text-sm leading-6 text-slate-600">{children || "Não avaliado"}</p>
    </div>
  );
}

function DetailList({ title, items, className = "" }: { title: string; items: string[]; className?: string }) {
  return (
    <details className={`rounded-lg border border-borderSoft bg-white p-4 ${className}`}>
      <summary className="cursor-pointer font-semibold text-slateInk">{title}</summary>
      {items.length ? (
        <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-slate-600">
          {items.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      ) : (
        <p className="mt-3 text-sm text-slate-600">Não avaliado.</p>
      )}
    </details>
  );
}

function Summary({ label, value }: { label: string; value: string }) {
  const technical = isTechnicalValue(label, value);
  return (
    <div className="min-w-0 rounded-lg border border-borderSoft bg-white p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</p>
      <div className="mt-1 min-w-0 text-sm font-bold text-slateInk">
        {technical ? <TechnicalValue value={value} /> : <span className="break-words">{value}</span>}
      </div>
    </div>
  );
}

function CardLine({ label, value, technical = false }: { label: string; value: string; technical?: boolean }) {
  return (
    <div className="grid min-w-0 gap-1 sm:grid-cols-[auto_minmax(0,1fr)]">
      <dt className="text-slate-500">{label}</dt>
      <dd className="min-w-0 text-left font-semibold text-slateInk sm:text-right">
        {technical || isTechnicalValue(label, value) ? <TechnicalValue value={value} copyable={false} /> : value}
      </dd>
    </div>
  );
}

function LoadingCard({ text, className = "" }: { text: string; className?: string }) {
  return <Card className={className}><p className="text-sm text-slate-600">{text}</p></Card>;
}

function ErrorCard({ text, className = "" }: { text: string; className?: string }) {
  return <Card className={className}><p className="text-sm font-semibold text-orange-800">{text}</p></Card>;
}

function EmptyCard({ text, className = "" }: { text: string; className?: string }) {
  return <Card className={className}><p className="text-sm text-slate-600">{text}</p></Card>;
}

function cleanModelName(entry: ModelEntry): string {
  return modelDisplayName(entry.name);
}

function badgeForModel(entry: ModelEntry): { label: string; tone: "approved" | "recommended" | "experimental" | "neutral" } {
  if (entry.name === "programmatic") return { label: "Padrão Geral", tone: "recommended" };
  if (entry.name === "simple_gan") return { label: "Experimental", tone: "experimental" };
  if (entry.recommended_artifact?.approved || entry.recommended_artifact?.recommended_for_neural_generation) {
    return { label: "Recomendado", tone: "approved" };
  }
  return { label: "Artefato Neural", tone: "neutral" };
}
