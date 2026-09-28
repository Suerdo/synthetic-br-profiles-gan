import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import { getGovernance } from "../api";
import { Badge } from "../components/ui/Badge";
import { Callout } from "../components/ui/Callout";
import { Card } from "../components/ui/Card";
import type { GovernanceIndicator, GovernanceSnapshot } from "../types/api";
import { formatDate, formatRate, formatUnknown } from "../utils/format";

export function GovernancePage() {
  const governance = useQuery({ queryKey: ["governance"], queryFn: getGovernance, staleTime: 45_000, retry: false });

  return (
    <div className="space-y-6">
      <header>
        <p className="text-sm font-semibold uppercase tracking-[0.18em] text-blueAction">Governança</p>
        <h1 className="mt-2 text-3xl font-bold text-slateInk">Governança</h1>
        <p className="mt-2 max-w-3xl text-base leading-7 text-slate-600">
          Acompanhe qualidade, diversidade, rastreabilidade e evidências dos dados sintéticos gerados.
        </p>
      </header>

      <Callout>
        A aprovação representa uma decisão técnica interna baseada nos critérios do projeto. Ela não constitui certificação
        externa, garantia de anonimização ou validação populacional oficial.
      </Callout>

      {governance.isLoading ? <Card><p className="text-sm text-slate-600">Carregando evidências de governança...</p></Card> : null}
      {governance.isError ? <ErrorState /> : null}
      {governance.data ? <GovernanceContent snapshot={governance.data} /> : null}
    </div>
  );
}

function GovernanceContent({ snapshot }: { snapshot: GovernanceSnapshot }) {
  return (
    <>
      <OperationalSummary snapshot={snapshot} />
      <RecommendedModel model={snapshot.recommended_model} />
      <IndicatorSection
        title="Qualidade dos Dados"
        description="Indicadores estruturais, qualidade categórica e quality gates disponíveis nos artefatos agregados."
        indicators={snapshot.quality.indicators}
      />
      <IndicatorSection
        title="Diversidade e Memorização"
        description="Métricas calculadas sobre as 11 colunas-base, sem identificadores derivados."
        indicators={snapshot.privacy.diversity_memorization}
      />
      <IndicatorSection
        title="Realismo Condicional"
        description="Indicadores de renda e distribuição condicional quando os artefatos agregados estiverem disponíveis."
        indicators={snapshot.income.indicators}
      />
      <ExecutionsTable rows={snapshot.executions} />
      <AuditSection events={snapshot.audit} />
      <Glossary terms={snapshot.glossary} />
    </>
  );
}

function OperationalSummary({ snapshot }: { snapshot: GovernanceSnapshot }) {
  return (
    <Card>
      <h2 className="text-xl font-bold text-slateInk">Resumo Operacional</h2>
      <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-5">
        {snapshot.operational.metrics.map((metric) => (
          <MetricCard key={metric.key} label={metric.label} value={formatUnknown(metric.value)} source={metric.source} help={metric.help} />
        ))}
      </div>
    </Card>
  );
}

function RecommendedModel({ model }: { model: Record<string, unknown> | null }) {
  if (!model) {
    return (
      <Card>
        <h2 className="text-xl font-bold text-slateInk">Modelo Neural Recomendado</h2>
        <p className="mt-3 text-sm leading-6 text-slate-600">Artefato neural recomendado não disponível neste ambiente.</p>
      </Card>
    );
  }
  const metrics = (model.metrics && typeof model.metrics === "object" ? model.metrics : {}) as Record<string, unknown>;
  const environment = (model.environment && typeof model.environment === "object" ? model.environment : {}) as Record<string, unknown>;
  return (
    <Card>
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="text-xl font-bold text-slateInk">Modelo Neural Recomendado</h2>
        <Badge tone="approved">{formatUnknown(model.status)}</Badge>
      </div>
      <div className="mt-4 grid gap-3 text-sm md:grid-cols-3">
        <Summary label="Identificador" value={formatUnknown(model.artifact_id)} />
        <Summary label="Vocabulário" value={`v${formatUnknown(model.vocabulary_version)}`} />
        <Summary label="Renda" value={`v${formatUnknown(model.income_model_version)}`} />
        <Summary label="Geografia" value={`v${formatUnknown(model.geography_model_version)}`} />
        <Summary label="Benchmark" value={formatUnknown(model.benchmark)} />
        <Summary label="Seeds" value={formatUnknown(model.seeds)} />
        <Summary label="Resultado" value={formatUnknown(model.result)} />
        <Summary label="Validade geográfica raw" value={formatRate(metricValue(metrics.raw_geographic_validity_rate))} />
        <Summary label="Validade global raw" value={formatRate(metricValue(metrics.raw_global_validity_rate))} />
        <Summary label="Duplicidade-base" value={formatRate(metricValue(metrics.duplicate_base_row_rate))} />
        <Summary label="Match com treino" value={formatRate(metricValue(metrics.exact_train_match_rate))} />
        <Summary label="Cobertura Geo_Key" value={formatRate(metricValue(metrics.geography_key_coverage))} />
        <Summary label="Ambiente" value={formatUnknown(environment.summary)} />
        <Summary label="Checksum geográfico" value={formatUnknown(model.geography_catalog_checksum)} />
        <Summary label="Limitações" value={formatUnknown(model.limitations)} />
      </div>
      <p className="mt-4 rounded-lg border border-blue-200 bg-blue-50 p-3 text-sm leading-6 text-blue-900">
        {formatUnknown(model.approval_note)}
      </p>
    </Card>
  );
}

function IndicatorSection({ title, description, indicators }: { title: string; description: string; indicators: GovernanceIndicator[] }) {
  return (
    <Card>
      <h2 className="text-xl font-bold text-slateInk">{title}</h2>
      <p className="mt-2 text-sm leading-6 text-slate-600">{description}</p>
      {indicators.length ? (
        <div className="mt-4 grid gap-3 lg:grid-cols-2">
          {indicators.map((indicator) => (
            <IndicatorCard key={`${indicator.metric ?? indicator["métrica"] ?? indicator.indicator ?? indicator.indicador}`} indicator={indicator} />
          ))}
        </div>
      ) : (
        <p className="mt-4 text-sm text-slate-600">Não avaliado.</p>
      )}
    </Card>
  );
}

function IndicatorCard({ indicator }: { indicator: GovernanceIndicator }) {
  const label = String(indicator.indicator ?? indicator.indicador ?? "Indicador");
  const value = indicator.value ?? indicator.valor;
  const source = String(indicator.source ?? indicator.fonte ?? "Fonte não avaliada");
  const interpretation = String(indicator.interpretation ?? indicator.interpretação ?? "Não avaliado");
  const isAbsent = value === null || value === undefined;
  return (
    <div className="rounded-lg border border-borderSoft bg-panel p-4">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3 className="font-bold text-slateInk">{label}</h3>
          <p className="mt-1 text-2xl font-bold text-slateInk">{formatUnknown(value)}</p>
        </div>
        <Badge tone={isAbsent ? "neutral" : indicator.gate_type === "mandatory" ? "recommended" : "neutral"}>
          {isAbsent ? "Não avaliado" : indicator.gate_type === "mandatory" ? "Obrigatório" : "Informativo"}
        </Badge>
      </div>
      <p className="mt-3 text-sm leading-6 text-slate-600">{interpretation}</p>
      <p className="mt-2 text-xs font-semibold text-slate-500">Fonte: {source}</p>
    </div>
  );
}

function ExecutionsTable({ rows }: { rows: Array<Record<string, unknown>> }) {
  const [modelFilter, setModelFilter] = useState("todos");
  const [statusFilter, setStatusFilter] = useState("todos");
  const models = useMemo(() => uniqueOptions(rows, "model"), [rows]);
  const statuses = useMemo(() => uniqueOptions(rows, "status"), [rows]);
  const filtered = rows
    .filter((row) => modelFilter === "todos" || String(row.model ?? "") === modelFilter)
    .filter((row) => statusFilter === "todos" || String(row.status ?? "") === statusFilter)
    .slice(0, 50);

  return (
    <Card>
      <h2 className="text-xl font-bold text-slateInk">Execuções Recentes</h2>
      <div className="mt-4 grid gap-3 md:grid-cols-2">
        <FilterSelect label="Modelo" value={modelFilter} options={models} onChange={setModelFilter} />
        <FilterSelect label="Status" value={statusFilter} options={statuses} onChange={setStatusFilter} />
      </div>
      <div className="mt-4 overflow-x-auto">
        {filtered.length ? (
          <table className="min-w-full border-collapse text-left text-sm">
            <thead>
              <tr className="bg-slate-200 text-slateInk">
                <th className="border border-borderSoft px-3 py-2">Data</th>
                <th className="border border-borderSoft px-3 py-2">Modelo</th>
                <th className="border border-borderSoft px-3 py-2">Status</th>
                <th className="border border-borderSoft px-3 py-2">Linhas</th>
                <th className="border border-borderSoft px-3 py-2">Duplicidade-base</th>
                <th className="border border-borderSoft px-3 py-2">Match treino</th>
                <th className="border border-borderSoft px-3 py-2">Combinações únicas</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((row, index) => (
                <tr key={`${row.identifier ?? index}`} className="odd:bg-white even:bg-panel">
                  <td className="border border-borderSoft px-3 py-2">{formatDate(row.created_at_utc as string | null | undefined)}</td>
                  <td className="border border-borderSoft px-3 py-2">{formatUnknown(row.model)}</td>
                  <td className="border border-borderSoft px-3 py-2">{formatUnknown(row.status)}</td>
                  <td className="border border-borderSoft px-3 py-2">{formatUnknown(row.rows)}</td>
                  <td className="border border-borderSoft px-3 py-2">{formatRate(metricValue(row.duplicate_base_row_rate))}</td>
                  <td className="border border-borderSoft px-3 py-2">{formatRate(metricValue(row.exact_train_match_rate))}</td>
                  <td className="border border-borderSoft px-3 py-2">{formatUnknown(row.unique_combinations)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="text-sm text-slate-600">Nenhuma execução recente disponível para os filtros atuais.</p>
        )}
      </div>
    </Card>
  );
}

function AuditSection({ events }: { events: Array<Record<string, unknown>> }) {
  return (
    <Card>
      <h2 className="text-xl font-bold text-slateInk">Auditoria</h2>
      {events.length ? (
        <div className="mt-4 space-y-3">
          {events.slice(0, 20).map((event, index) => (
            <div key={`${event.timestamp_utc ?? index}`} className="rounded-lg border border-borderSoft bg-panel p-3 text-sm">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <strong className="text-slateInk">{formatUnknown(event.event_type ?? event.event)}</strong>
                <span className="text-xs font-semibold text-slate-500">{formatDate(event.timestamp_utc as string | null | undefined)}</span>
              </div>
              <p className="mt-2 text-slate-600">Sessão: {formatUnknown(event.session_id)}</p>
            </div>
          ))}
        </div>
      ) : (
        <p className="mt-3 text-sm text-slate-600">Nenhum evento de auditoria disponível.</p>
      )}
    </Card>
  );
}

function Glossary({ terms }: { terms: Array<{ term: string; definition: string }> }) {
  return (
    <Card>
      <details open>
        <summary className="cursor-pointer text-lg font-bold text-slateInk">Glossário</summary>
        <div className="mt-4 grid gap-3 md:grid-cols-2">
          {terms.map((term) => (
            <div key={term.term} className="rounded-lg bg-panel p-3 text-sm">
              <h3 className="font-bold text-slateInk">{term.term}</h3>
              <p className="mt-1 leading-6 text-slate-600">{term.definition}</p>
            </div>
          ))}
        </div>
      </details>
    </Card>
  );
}

function MetricCard({ label, value, source, help }: { label: string; value: string; source: string; help: string }) {
  const absent = value === "Não avaliado";
  return (
    <div className="rounded-lg border border-borderSoft bg-panel p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-slate-600">{label}</p>
          <p className="mt-2 text-xl font-bold text-slateInk">{value}</p>
        </div>
        <Badge tone={absent ? "neutral" : "approved"}>{absent ? "Não avaliado" : "OK"}</Badge>
      </div>
      <p className="mt-3 text-sm leading-6 text-slate-600">{help}</p>
      <p className="mt-2 text-xs font-semibold text-slate-500">Fonte: {source}</p>
    </div>
  );
}

function Summary({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-borderSoft bg-panel p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-1 break-words text-sm font-bold text-slateInk">{value}</p>
    </div>
  );
}

function FilterSelect({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (value: string) => void }) {
  return (
    <label className="text-sm font-semibold text-slate-700">
      {label}
      <select
        className="mt-1 w-full rounded-lg border border-slate-500 bg-[#E8EEF7] px-3 py-2 text-slateInk focus:border-navy focus:outline-none focus:ring-2 focus:ring-blue-200"
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        <option value="todos">Todos</option>
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}

function ErrorState() {
  return (
    <Card>
      <h2 className="text-xl font-bold text-slateInk">Não foi possível carregar a governança</h2>
      <p className="mt-2 text-sm leading-6 text-slate-600">
        As demais páginas continuam disponíveis. Tente novamente após verificar se a API consegue ler os artefatos agregados.
      </p>
    </Card>
  );
}

function uniqueOptions(rows: Array<Record<string, unknown>>, key: string): string[] {
  return Array.from(new Set(rows.map((row) => row[key]).filter((value) => value !== null && value !== undefined).map(String))).sort();
}

function metricValue(value: unknown): number | { min: number; max: number } | null | undefined {
  if (typeof value === "number") return value;
  if (
    typeof value === "object" &&
    value !== null &&
    "min" in value &&
    "max" in value &&
    typeof (value as { min: unknown }).min === "number" &&
    typeof (value as { max: unknown }).max === "number"
  ) {
    return value as { min: number; max: number };
  }
  return value === null || value === undefined ? null : undefined;
}
