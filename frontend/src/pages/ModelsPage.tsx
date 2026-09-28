import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import { getModelArtifacts, getModels } from "../api";
import { Badge } from "../components/ui/Badge";
import { Card } from "../components/ui/Card";
import type { ModelArtifact, ModelEntry, ModelName } from "../types/api";
import { formatBytes, formatDate, formatNumber, formatRate } from "../utils/format";

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
    <div className="space-y-6">
      <header>
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
  const version = entry.recommended_artifact
    ? `Vocabulário v${entry.recommended_artifact.vocabulary_version} · Renda v${entry.recommended_artifact.income_model_version} · Geografia v${entry.recommended_artifact.geography_model_version}`
    : entry.requires_saved_artifact
      ? "Artefato neural não disponível neste ambiente"
      : "Geração direta sem artefato";

  return (
    <button
      type="button"
      onClick={onSelect}
      className={`rounded-xl border bg-white p-5 text-left shadow-card transition hover:border-blue-300 ${
        active ? "border-blueAction ring-2 ring-blue-100" : "border-borderSoft"
      }`}
    >
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="text-xl font-bold text-slateInk">{cleanModelName(entry)}</h2>
        <Badge tone={mainBadge.tone}>{mainBadge.label}</Badge>
      </div>
      <p className="mt-3 text-sm leading-6 text-slate-600">{entry.short_description}</p>
      <dl className="mt-4 space-y-2 text-sm">
        <CardLine label="Disponibilidade" value={entry.available ? "Disponível" : "Não disponível"} />
        <CardLine label="Indicação" value={entry.recommended_use_cases.slice(0, 3).join(", ") || "Não avaliado"} />
        <CardLine label="Versões" value={version} />
      </dl>
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
    <Card>
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="text-2xl font-bold text-slateInk">{entry.title}</h2>
        <Badge tone={badgeForModel(entry).tone}>{badgeForModel(entry).label}</Badge>
        {selectedArtifact?.recommended_for_neural_generation ? <Badge tone="approved">Artefato neural recomendado</Badge> : null}
      </div>
      <p className="mt-3 max-w-4xl text-sm leading-6 text-slate-600">{entry.detailed_description}</p>

      <div className="mt-5 grid items-stretch gap-4 md:grid-cols-2">
        <SummaryCard title="Resumo simples">{entry.simple_summary || entry.summary}</SummaryCard>
        <SummaryCard title="Resumo técnico">{entry.technical_summary}</SummaryCard>
      </div>

      <h3 className="mt-6 font-bold text-slateInk">Usos recomendados</h3>
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
  if (!artifact) return <EmptyCard text={emptyMessage ?? `Nenhum artefato ${model} válido foi encontrado.`} className="mt-5" />;

  return (
    <div className="mt-5 rounded-xl border border-borderSoft bg-panel p-4">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-lg font-bold text-slateInk">Artefato selecionado</h3>
        <Badge tone={artifact.approved ? "approved" : artifact.recommended ? "candidate" : "neutral"}>{artifact.status}</Badge>
      </div>
      {artifact.warning ? <p className="mt-3 rounded-lg border border-orange-300 bg-orange-50 p-3 text-sm text-orange-800">{artifact.warning}</p> : null}
      <div className="mt-4 grid gap-3 text-sm md:grid-cols-3">
        <Summary label="Identificador" value={artifact.artifact_id} />
        <Summary label="Criado em" value={formatDate(artifact.created_at_utc)} />
        <Summary label="Status" value={artifact.status} />
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
        <summary className="cursor-pointer font-semibold text-slateInk">Outros artefatos disponíveis</summary>
        <div className="mt-4 overflow-x-auto">
          <table className="min-w-full border-collapse text-left text-sm">
            <thead>
              <tr className="bg-slate-200 text-slateInk">
                <th className="border border-borderSoft px-3 py-2">Artefato</th>
                <th className="border border-borderSoft px-3 py-2">Criado em</th>
                <th className="border border-borderSoft px-3 py-2">Vocabulário</th>
                <th className="border border-borderSoft px-3 py-2">Renda</th>
                <th className="border border-borderSoft px-3 py-2">Geografia</th>
                <th className="border border-borderSoft px-3 py-2">Status</th>
              </tr>
            </thead>
            <tbody>
              {artifacts.map((item) => (
                <tr key={item.artifact_id} className="odd:bg-white even:bg-panel">
                  <td className="border border-borderSoft px-3 py-2">{item.artifact_id}</td>
                  <td className="border border-borderSoft px-3 py-2">{formatDate(item.created_at_utc)}</td>
                  <td className="border border-borderSoft px-3 py-2">v{item.vocabulary_version}</td>
                  <td className="border border-borderSoft px-3 py-2">v{item.income_model_version}</td>
                  <td className="border border-borderSoft px-3 py-2">v{item.geography_model_version}</td>
                  <td className="border border-borderSoft px-3 py-2">{item.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
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
  return (
    <div className="rounded-lg border border-borderSoft bg-white p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-1 break-words text-sm font-bold text-slateInk">{value}</p>
    </div>
  );
}

function CardLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-3">
      <dt className="text-slate-500">{label}</dt>
      <dd className="text-right font-semibold text-slateInk">{value}</dd>
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
  if (entry.name === "programmatic") return "Programático";
  if (entry.name === "simple_gan") return "GAN Simples";
  return "CTGAN";
}

function badgeForModel(entry: ModelEntry): { label: string; tone: "approved" | "recommended" | "experimental" | "neutral" } {
  if (entry.name === "programmatic") return { label: "Padrão geral", tone: "recommended" };
  if (entry.name === "simple_gan") return { label: "Experimental", tone: "experimental" };
  if (entry.recommended_artifact?.approved || entry.recommended_artifact?.recommended_for_neural_generation) {
    return { label: "Recomendado", tone: "approved" };
  }
  return { label: "Artefato neural", tone: "neutral" };
}
