import { useQuery } from "@tanstack/react-query";
import {
  BarChart3,
  BookOpen,
  Boxes,
  CalendarClock,
  Database,
  FileCheck2,
  Fingerprint,
  GitCommit,
  History,
  Info,
  ListChecks,
  LockKeyhole,
  ShieldCheck,
  Workflow
} from "lucide-react";
import { useMemo, useState } from "react";

import { getGovernance } from "../api";
import { Badge } from "../components/ui/Badge";
import { Callout } from "../components/ui/Callout";
import { Card } from "../components/ui/Card";
import { isTechnicalValue, TechnicalValue } from "../components/ui/TechnicalValue";
import type {
  GovernanceGate,
  GovernanceIndicator,
  GovernanceModelEvidence,
  GovernanceProvenance,
  GovernanceSnapshot,
  GovernanceStrategy,
  ModelName
} from "../types/api";
import { formatBytes, formatDate, formatRate, formatUnknown } from "../utils/format";
import { modelDisplayName, statusDisplayName } from "../utils/labels";

type BadgeTone = "approved" | "recommended" | "candidate" | "experimental" | "smoke" | "legacy" | "neutral" | "warning";

const privacyGroups = {
  diversity: ["unique", "combina", "duplic"],
  memorization: ["exact", "correspond"],
  proximity: ["dcr", "nndr", "nearest", "closest"]
};

const auditPriority = [
  "generation_started",
  "generation_completed",
  "generation_succeeded",
  "model_selected",
  "configuration_changed",
  "artifact_created",
  "evaluation_completed",
  "gate_passed",
  "gate_failed",
  "artifact_quarantined",
  "model_promoted",
  "model_demoted",
  "export_completed",
  "configuration_restored"
];

export function GovernancePage() {
  const governance = useQuery({ queryKey: ["governance"], queryFn: getGovernance, staleTime: 45_000, retry: false });

  return (
    <div className="space-y-7">
      <PageHeader
        eyebrow="Governança"
        title="Governança"
        description="Acompanhe decisões, evidências e rastreabilidade dos dados sintéticos."
      />

      {governance.isLoading ? <GovernanceSkeleton /> : null}
      {governance.isError ? <ErrorState /> : null}
      {governance.data ? <GovernanceContent snapshot={governance.data} /> : null}
    </div>
  );
}

function GovernanceContent({ snapshot }: { snapshot: GovernanceSnapshot }) {
  const evidenceOptions = snapshot.evidence_by_model?.models ?? [];
  const [selectedEvidenceModel, setSelectedEvidenceModel] = useState<ModelName>(snapshot.evidence_by_model?.default_model ?? "programmatic");
  const selectedEvidence =
    evidenceOptions.find((item) => item.model === selectedEvidenceModel) ??
    evidenceOptions.find((item) => item.model === snapshot.evidence_by_model?.default_model) ??
    evidenceOptions[0] ??
    null;
  return (
    <>
      <DecisionHero snapshot={snapshot} />
      <StrategiesSection strategies={snapshot.available_strategies} />
      <RecommendedModel model={snapshot.recommended_model} />
      <ProvenanceSection provenance={snapshot.provenance} />
      <EvidenceSelector
        options={evidenceOptions}
        selected={selectedEvidence}
        value={selectedEvidence?.model ?? selectedEvidenceModel}
        onChange={setSelectedEvidenceModel}
      />
      <QualityGatesSection gates={selectedEvidence?.quality_gates ?? []} />
      <PrivacySection indicators={selectedEvidence?.privacy.diversity_memorization ?? []} />
      <FidelitySection indicators={selectedEvidence?.income.indicators ?? []} />
      <ExecutionsSection rows={snapshot.executions} summary={snapshot.operational.summary} />
      <AuditSection events={snapshot.audit} />
      <GlossarySection terms={snapshot.glossary} />
    </>
  );
}

function EvidenceSelector({
  options,
  selected,
  value,
  onChange
}: {
  options: GovernanceModelEvidence[];
  selected: GovernanceModelEvidence | null;
  value: ModelName;
  onChange: (value: ModelName) => void;
}) {
  return (
    <SectionShell
      icon={<FileCheck2 className="h-5 w-5" aria-hidden="true" />}
      title="Evidências do Modelo"
      description="A seleção abaixo controla Quality Gates, Privacidade, Diversidade e Memorização, e Realismo e Fidelidade Estatística. Execuções Recentes mantém filtros próprios."
    >
      <div className="grid gap-4 lg:grid-cols-[minmax(220px,320px)_minmax(0,1fr)]">
        <label className="text-sm font-semibold text-slate-700">
          Modelo avaliado
          <select
            className="mt-1 w-full rounded-lg border border-slate-500 bg-[#E8EEF7] px-3 py-2 text-sm text-slateInk focus:border-navy focus:outline-none focus:ring-2 focus:ring-blue-200"
            value={value}
            onChange={(event) => onChange(event.target.value as ModelName)}
            aria-label="Modelo das evidências"
          >
            {options.map((option) => (
              <option key={option.model} value={option.model}>
                {modelDisplayName(option.model)} — {option.role === "Modelo neural recomendado" ? "Recomendado" : evidenceRoleLabel(option.role)}
              </option>
            ))}
          </select>
        </label>
        <div className="min-w-0 rounded-xl border border-borderSoft bg-panel p-4">
          {selected ? (
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Contexto selecionado</p>
                <h3 className="mt-1 text-xl font-bold text-slateInk">{modelDisplayName(selected.model)}</h3>
                <p className="mt-2 text-sm leading-6 text-slate-600">{evidenceRoleLabel(selected.role)}</p>
                <dl className="mt-3 grid gap-2 text-sm sm:grid-cols-2">
                  <Definition label="Fonte" value={selected.source_identifier ?? "Não avaliado"} />
                  <Definition label="Último Status" value={statusDisplayName(selected.latest_execution_status)} />
                </dl>
              </div>
              <StatusBadge status={selected.status_label} label={statusDisplayName(selected.status_label)} />
            </div>
          ) : (
            <EmptyState title="Nenhum modelo disponível" text="A API não retornou estratégias de evidência para consulta." compact />
          )}
        </div>
      </div>
    </SectionShell>
  );
}

function DecisionHero({ snapshot }: { snapshot: GovernanceSnapshot }) {
  const decision = snapshot.governance_decision;
  const passed = decision.mandatory_passed;
  const total = decision.mandatory_total;
  return (
    <section className="overflow-hidden rounded-2xl border border-blue-200 bg-white shadow-card">
      <div className="border-b border-blue-200 bg-gradient-to-r from-slateInk to-navy px-5 py-5 text-white md:px-7">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <p className="text-sm font-semibold uppercase tracking-[0.18em] text-blue-200">Modelo Neural Recomendado</p>
            <h2 className="mt-2 text-2xl font-bold">CTGAN</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-200">
              configuração → geração → validação → avaliação → evidências → decisão → artefato → rastreabilidade
            </p>
          </div>
          <StatusBadge status={decision.recommendation_status} label="Recomendado" />
        </div>
      </div>
      <div className="grid gap-5 p-5 md:grid-cols-[1.4fr_1fr] md:p-7">
        <div>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            <DecisionItem label="Avaliação interna" value={statusLabel(decision.evaluation_status)} />
            <DecisionItem label="Recomendação neural" value={statusLabel(decision.recommendation_status)} />
            <DecisionItem label="Padrão geral" value={decision.general_default ? "Sim" : "Não"} />
            <DecisionItem label="Produção" value={statusLabel(decision.production_status)} />
            <DecisionItem label="Critérios obrigatórios" value={passed !== null && total !== null ? `${passed}/${total}` : "Não avaliado"} />
            <DecisionItem label="Ressalvas" value={formatUnknown(decision.caveats_count)} />
          </div>
          <div className="mt-5">
            <Callout>{decision.disclaimer}</Callout>
          </div>
        </div>
        <div className="rounded-xl border border-borderSoft bg-panel p-4">
          <h3 className="flex items-center gap-2 font-bold text-slateInk">
            <ShieldCheck className="h-5 w-5 text-blueAction" aria-hidden="true" />
            Escopo Avaliado
          </h3>
          <dl className="mt-4 space-y-3 text-sm">
            <Definition label="Escopo" value={decision.scope} />
            <Definition label="Artefato" value={decision.artifact_id} />
            <Definition label="Benchmark" value={decision.benchmark} />
            <Definition label="Data da decisão" value={formatDate(decision.decided_at_utc)} />
          </dl>
        </div>
      </div>
    </section>
  );
}

function StrategiesSection({ strategies }: { strategies: GovernanceStrategy[] }) {
  return (
    <SectionShell
      icon={<Workflow className="h-5 w-5" aria-hidden="true" />}
      title="Estratégias / Modelos Disponíveis"
      description="Estratégia do projeto e disponibilidade de artefato são exibidas separadamente."
    >
      <div className="grid gap-4 lg:grid-cols-3">
        {strategies.map((strategy) => (
          <div key={strategy.model} className="rounded-xl border border-borderSoft bg-white p-4">
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="text-lg font-bold text-slateInk">{modelDisplayName(strategy.model)}</h3>
              <StatusBadge
                status={
                  strategy.model === "simple_gan"
                    ? "experimental"
                    : strategy.model === "ctgan" && strategy.recommended_artifact_id
                      ? "recommended"
                      : strategy.model === "programmatic"
                        ? "default"
                        : "neutral"
                }
              />
            </div>
            <p className="mt-3 text-sm leading-6 text-slate-600">{strategy.role}</p>
            <dl className="mt-4 space-y-2 text-sm">
              <Definition label="Status" value={strategy.status} />
              <Definition label="Disponibilidade Operacional" value={strategy.operational_availability} />
              <Definition label="Exige Artefato" value={strategy.artifact_required ? "Sim" : "Não"} />
              <Definition label="Artefato Disponível" value={strategy.artifact_available ? "Sim" : "Não"} />
              <Definition label="Artefatos Válidos" value={strategy.artifact_count} />
            </dl>
          </div>
        ))}
      </div>
    </SectionShell>
  );
}

function RecommendedModel({ model }: { model: Record<string, unknown> | null }) {
  if (!model) {
    return (
      <SectionShell icon={<Boxes className="h-5 w-5" aria-hidden="true" />} title="Modelo Neural Recomendado">
        <EmptyState title="Nenhum artefato neural recomendado" text="A API não encontrou um artefato CTGAN aprovado e recomendado neste ambiente." />
      </SectionShell>
    );
  }
  const metrics = objectValue(model.metrics);
  return (
    <SectionShell
      icon={<Boxes className="h-5 w-5" aria-hidden="true" />}
      title="Modelo Neural Recomendado"
      description="O programático continua sendo o padrão geral. Este card descreve apenas o artefato neural recomendado."
    >
      <div className="rounded-2xl border border-blue-200 bg-blue-50/60 p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h3 className="text-2xl font-bold text-slateInk">CTGAN</h3>
            <p className="mt-2 text-sm leading-6 text-slate-700">
              Artefato neural recomendado do projeto, com vocabulário v{formatUnknown(model.vocabulary_version)}, renda v
              {formatUnknown(model.income_model_version)} e geografia v{formatUnknown(model.geography_model_version)}.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <StatusBadge status={String(model.evaluation_status ?? model.status ?? "")} label="Aprovado Internamente" />
            <StatusBadge status={String(model.recommendation_status ?? "")} label="Modelo Neural Recomendado" />
          </div>
        </div>
        <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          <MetricTile label="Evidência" value={formatUnknown(model.result)} />
          <MetricTile label="Validade geográfica bruta" value={formatRate(metricValue(metrics.raw_geographic_validity_rate))} />
          <MetricTile label="Validade global bruta" value={formatRate(metricValue(metrics.raw_global_validity_rate))} />
          <MetricTile label="Duplicidade-base" value={formatRate(metricValue(metrics.duplicate_base_row_rate))} />
          <MetricTile label="Match exato com treino" value={formatRate(metricValue(metrics.exact_train_match_rate))} />
          <MetricTile label="Cobertura Geo_Key" value={formatRate(metricValue(metrics.geography_key_coverage))} />
          <MetricTile label="Tamanho do modelo" value={formatMetricMaybeBytes(metrics.model_size_mb, model)} />
          <MetricTile label="Biblioteca" value={formatUnknown(metrics.library)} />
        </div>
        <dl className="mt-5 grid gap-3 text-sm md:grid-cols-2">
          <Definition label="Artefato" value={model.artifact_id} />
          <Definition label="Benchmark" value={model.benchmark} />
          <Definition label="Seeds" value={model.seeds} />
          <Definition label="Checksum geográfico" value={model.geography_catalog_checksum} />
        </dl>
      </div>
    </SectionShell>
  );
}

function ProvenanceSection({ provenance }: { provenance: GovernanceProvenance }) {
  const byLabel = new Map(provenance.items.map((item) => [item.label, item]));
  return (
    <SectionShell
      icon={<Fingerprint className="h-5 w-5" aria-hidden="true" />}
      title="Proveniência e Reprodutibilidade"
      description="Somente informações registradas nos manifestos, configs ou no repositório local são exibidas."
    >
      <ol className="mb-5 grid gap-3 md:grid-cols-4">
        {(provenance.timeline ?? []).map((item) => (
          <li key={item.step} className="rounded-xl border border-borderSoft bg-panel p-3">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{item.step}</p>
            <p className="mt-1 break-words text-sm font-bold text-slateInk">{formatTimelineValue(item.step, item.value)}</p>
            <p className="mt-1 text-xs text-slate-500">Fonte: {item.source}</p>
          </li>
        ))}
      </ol>
      <div className="grid gap-4 lg:grid-cols-2">
        {(provenance.groups ?? []).map((group) => (
          <div key={group.title} className="rounded-xl border border-borderSoft bg-white p-4">
            <h3 className="mb-3 flex items-center gap-2 font-bold text-slateInk">
              {group.title === "Identificação" ? <GitCommit className="h-4 w-4 text-blueAction" aria-hidden="true" /> : null}
              {group.title === "Dados e split" ? <Database className="h-4 w-4 text-blueAction" aria-hidden="true" /> : null}
              {group.title === "Hiperparâmetros" ? <FileCheck2 className="h-4 w-4 text-blueAction" aria-hidden="true" /> : null}
              {group.title === "Ambiente e versões" ? <CalendarClock className="h-4 w-4 text-blueAction" aria-hidden="true" /> : null}
              {group.title}
            </h3>
            <dl className="space-y-2 text-sm">
              {group.keys.map((key) => {
                const item = byLabel.get(key);
                return <Definition key={key} label={key} value={item?.value} source={item?.source} />;
              })}
            </dl>
          </div>
        ))}
      </div>
    </SectionShell>
  );
}

function QualityGatesSection({ gates }: { gates: GovernanceGate[] }) {
  const rows = gates;
  const passedMandatory = rows.filter((gate) => gate.mandatory && gate.passed === true).length;
  const mandatoryTotal = rows.filter((gate) => gate.mandatory).length;
  return (
    <SectionShell
      icon={<ListChecks className="h-5 w-5" aria-hidden="true" />}
      title="Quality Gates"
      description="Critérios obrigatórios e informativos usados para interpretar a aprovação técnica interna."
    >
      <div className="mb-4 grid gap-3 md:grid-cols-3">
        <MetricTile label="Obrigatórios Aprovados" value={mandatoryTotal ? `${passedMandatory}/${mandatoryTotal}` : "Não avaliado"} />
        <MetricTile label="Gates Informativos" value={rows.filter((gate) => !gate.mandatory).length} />
        <MetricTile label="Fonte Principal" value={rows.length ? "quality_gates.json / approval_manifest.json" : "Não avaliado"} />
      </div>
      {rows.length ? (
      <div className="overflow-x-auto">
        <table className="min-w-full border-collapse text-left text-sm">
          <thead>
            <tr className="bg-slate-200 text-slateInk">
              <th className="border border-borderSoft px-3 py-2">Métrica</th>
              <th className="border border-borderSoft px-3 py-2">Resultado</th>
              <th className="border border-borderSoft px-3 py-2">Critério</th>
              <th className="border border-borderSoft px-3 py-2">Tipo</th>
              <th className="border border-borderSoft px-3 py-2">Status</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 40).map((gate) => (
              <tr key={gate.id} className="odd:bg-white even:bg-panel">
                <td className="border border-borderSoft px-3 py-2 font-semibold text-slateInk">
                  <details>
                    <summary className="cursor-pointer">{gate.metric}</summary>
                    <div className="mt-2 rounded-lg bg-white p-2 text-xs font-normal text-slate-600">
                      <p>Fonte: {formatUnknown(gate.source)}</p>
                      <p>Métrica: {formatUnknown(gate.id)}</p>
                      <p>Evidência: {formatEvidence(gate.evidence)}</p>
                    </div>
                  </details>
                </td>
                <td className="border border-borderSoft px-3 py-2">{formatUnknown(gate.observed)}</td>
                <td className="border border-borderSoft px-3 py-2">{formatCriterion(gate)}</td>
                <td className="border border-borderSoft px-3 py-2">
                  <Badge tone={gate.mandatory ? "recommended" : "neutral"}>{gate.mandatory ? "Obrigatório" : "Informativo"}</Badge>
                </td>
                <td className="border border-borderSoft px-3 py-2">
                  <StatusBadge status={gate.passed === null ? "not_evaluated" : gate.passed ? "approved" : gate.mandatory ? "rejected" : "informational"} label={statusDisplayName(gate.status)} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      ) : (
        <EmptyState title="Nenhuma evidência disponível para esta métrica" text="Não há quality gates agregados para o modelo selecionado neste ambiente." />
      )}
    </SectionShell>
  );
}

function PrivacySection({ indicators }: { indicators: GovernanceIndicator[] }) {
  const diversity = indicators.filter((indicator) => matchesGroup(indicator, privacyGroups.diversity));
  const memorization = indicators.filter((indicator) => matchesGroup(indicator, privacyGroups.memorization));
  const proximity = indicators.filter((indicator) => matchesGroup(indicator, privacyGroups.proximity));
  return (
    <SectionShell
      icon={<LockKeyhole className="h-5 w-5" aria-hidden="true" />}
      title="Privacidade, Diversidade e Memorização"
      description="Essas métricas usam as 11 colunas-base; identificadores derivados não participam."
    >
      <div className="grid gap-4 lg:grid-cols-3">
        <IndicatorGroup title="Diversidade" indicators={diversity} />
        <IndicatorGroup title="Memorização" indicators={memorization} />
        <IndicatorGroup title="Proximidade" indicators={proximity} diagnostic />
      </div>
      <p className="mt-4 rounded-lg border border-orange-200 bg-orange-50 p-3 text-sm leading-6 text-orange-900">
        A ausência de correspondência exata não constitui, isoladamente, garantia de anonimização. DCR e NNDR são métricas diagnósticas quando não há limiar obrigatório configurado.
      </p>
    </SectionShell>
  );
}

function FidelitySection({ indicators }: { indicators: GovernanceIndicator[] }) {
  const summary = indicators.slice(0, 6);
  return (
    <SectionShell
      icon={<BarChart3 className="h-5 w-5" aria-hidden="true" />}
      title="Realismo e Fidelidade Estatística"
      description="Resumo diagnóstico de realismo condicional de renda e fidelidade agregada."
    >
      <h3 className="font-bold text-slateInk">Realismo condicional de renda</h3>
      <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {summary.map((indicator) => (
          <MetricTile key={indicatorKey(indicator)} label={indicatorLabel(indicator)} value={formatUnknown(indicator.value ?? indicator.valor)} source={String(indicator.source ?? indicator.fonte ?? "Fonte não avaliada")} />
        ))}
      </div>
      <details className="mt-5 rounded-xl border border-borderSoft bg-white p-4">
        <summary className="cursor-pointer font-semibold text-slateInk">Ver análise detalhada</summary>
        <div className="mt-4 overflow-x-auto">
          <table className="min-w-full border-collapse text-left text-sm">
            <thead>
              <tr className="bg-slate-200 text-slateInk">
                <th className="border border-borderSoft px-3 py-2">Indicador</th>
                <th className="border border-borderSoft px-3 py-2">Valor</th>
                <th className="border border-borderSoft px-3 py-2">Fonte</th>
                <th className="border border-borderSoft px-3 py-2">Interpretação</th>
              </tr>
            </thead>
            <tbody>
              {indicators.map((indicator) => (
                <tr key={indicatorKey(indicator)} className="odd:bg-white even:bg-panel">
                  <td className="border border-borderSoft px-3 py-2 font-semibold">{indicatorLabel(indicator)}</td>
                  <td className="border border-borderSoft px-3 py-2">{formatUnknown(indicator.value ?? indicator.valor)}</td>
                  <td className="border border-borderSoft px-3 py-2">{formatUnknown(indicator.source ?? indicator.fonte)}</td>
                  <td className="border border-borderSoft px-3 py-2">{formatUnknown(indicator.interpretation ?? indicator.interpretação)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </SectionShell>
  );
}

function ExecutionsSection({ rows, summary }: { rows: Array<Record<string, unknown>>; summary: Record<string, unknown> }) {
  const [statusFilter, setStatusFilter] = useState("todos");
  const [showMore, setShowMore] = useState(false);
  const statuses = useMemo(() => uniqueOptions(rows, "status"), [rows]);
  const filtered = rows.filter((row) => statusFilter === "todos" || String(row.status ?? "") === statusFilter);
  const visible = filtered.slice(0, showMore ? 50 : 20);
  return (
    <SectionShell
      icon={<History className="h-5 w-5" aria-hidden="true" />}
      title="Execuções Recentes"
      description="A lista usa apenas manifestos e agregados; datasets completos não são carregados para montar o histórico."
    >
      <div className="mb-4 grid gap-3 md:grid-cols-4">
        <MetricTile label="Execuções Registradas" value={formatUnknown(summary.registered_executions ?? summary.total_runs)} />
        <MetricTile label="Execuções com Avaliação Completa" value={formatUnknown(summary.evaluated_executions)} />
        <MetricTile label="Última Avaliação" value={formatUnknown(summary.latest_evaluation)} />
        <MetricTile label="Estratégias Disponíveis" value={formatUnknown(summary.strategy_count)} />
      </div>
      <div className="max-w-sm">
        <FilterSelect label="Status" value={statusFilter} options={statuses} onChange={setStatusFilter} formatOption={statusDisplayName} />
      </div>
      <div className="mt-4 overflow-x-auto">
        {visible.length ? (
          <table className="min-w-full border-collapse text-left text-sm">
            <thead>
              <tr className="bg-slate-200 text-slateInk">
                <th className="border border-borderSoft px-3 py-2">Data</th>
                <th className="border border-borderSoft px-3 py-2">Modelo</th>
                <th className="border border-borderSoft px-3 py-2">Status</th>
                <th className="border border-borderSoft px-3 py-2">Linhas</th>
                <th className="border border-borderSoft px-3 py-2">Avaliação</th>
                <th className="border border-borderSoft px-3 py-2">Duplicidade-base</th>
                <th className="border border-borderSoft px-3 py-2">Match treino</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((row, index) => (
                <tr key={`${row.identifier ?? index}`} className="odd:bg-white even:bg-panel">
                  <td className="border border-borderSoft px-3 py-2">{formatDate(row.created_at_utc as string | null | undefined)}</td>
                  <td className="border border-borderSoft px-3 py-2">{modelDisplayName(row.model)}</td>
                  <td className="border border-borderSoft px-3 py-2">{statusDisplayName(row.status)}</td>
                  <td className="border border-borderSoft px-3 py-2">{formatUnknown(row.rows)}</td>
                  <td className="border border-borderSoft px-3 py-2">{evaluationStatus(row)}</td>
                  <td className="border border-borderSoft px-3 py-2">{formatRate(metricValue(row.duplicate_base_row_rate))}</td>
                  <td className="border border-borderSoft px-3 py-2">{formatRate(metricValue(row.exact_train_match_rate))}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <EmptyState title="Nenhuma execução disponível" text="Não há registros para os filtros atuais." />
        )}
      </div>
      {filtered.length > visible.length ? (
        <button type="button" className="mt-4 rounded-lg border border-borderSoft bg-white px-4 py-2 text-sm font-semibold text-slateInk hover:border-blueAction" onClick={() => setShowMore(true)}>
          Ver Mais Execuções
        </button>
      ) : null}
    </SectionShell>
  );
}

function AuditSection({ events }: { events: Array<Record<string, unknown>> }) {
  const sorted = [...events].sort((left, right) => eventRank(left) - eventRank(right));
  const last = events[0];
  const period = events.length ? `${formatDate(events[events.length - 1].timestamp_utc as string | null | undefined)} a ${formatDate(events[0].timestamp_utc as string | null | undefined)}` : "Não avaliado";
  return (
    <SectionShell icon={<History className="h-5 w-5" aria-hidden="true" />} title="Trilha de Auditoria">
      <details className="rounded-xl border border-borderSoft bg-white p-4">
        <summary className="cursor-pointer font-semibold text-slateInk">Ver Trilha de Auditoria</summary>
        <div className="mt-4 grid gap-3 md:grid-cols-3">
          <MetricTile label="Eventos" value={events.length} />
          <MetricTile label="Último Evento" value={formatUnknown(last?.event_type ?? last?.event)} />
          <MetricTile label="Período Coberto" value={period} />
        </div>
        <div className="mt-5 space-y-3">
          {sorted.slice(0, 30).map((event, index) => (
            <div key={`${event.timestamp_utc ?? index}`} className="rounded-lg border border-borderSoft bg-panel p-3 text-sm">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <strong className="text-slateInk">{eventDescription(event)}</strong>
                <span className="text-xs font-semibold text-slate-500">{formatDate(event.timestamp_utc as string | null | undefined)}</span>
              </div>
              <details className="mt-2">
                <summary className="cursor-pointer text-xs font-semibold text-slate-500">Detalhes Técnicos Sanitizados</summary>
                <pre className="mt-2 overflow-x-auto rounded bg-white p-2 text-xs text-slate-600">{formatEvidence(event)}</pre>
              </details>
            </div>
          ))}
          {!events.length ? <EmptyState title="Nenhum evento de auditoria" text="A trilha sanitizada ainda não possui eventos registrados." /> : null}
        </div>
      </details>
    </SectionShell>
  );
}

function GlossarySection({ terms }: { terms: Array<{ term: string; definition: string }> }) {
  return (
    <SectionShell icon={<BookOpen className="h-5 w-5" aria-hidden="true" />} title="Glossário e Metodologia">
      <details className="rounded-xl border border-borderSoft bg-white p-4">
        <summary className="cursor-pointer font-semibold text-slateInk">Abrir Glossário Completo</summary>
        <div className="mt-4 grid gap-3 md:grid-cols-2">
          {terms.map((term) => (
            <div key={term.term} className="rounded-lg bg-panel p-3 text-sm">
              <h3 className="font-bold text-slateInk">{term.term}</h3>
              <p className="mt-1 leading-6 text-slate-600">{term.definition}</p>
            </div>
          ))}
        </div>
      </details>
    </SectionShell>
  );
}

function SectionShell({ icon, title, description, children }: { icon: React.ReactNode; title: string; description?: string; children: React.ReactNode }) {
  return (
    <section className="rounded-2xl border border-borderSoft bg-white p-5 shadow-card md:p-6">
      <div className="mb-5 flex items-start gap-3">
        <span className="mt-1 inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-blue-50 text-blueAction">{icon}</span>
        <div>
          <h2 className="text-xl font-bold text-slateInk">{title}</h2>
          {description ? <p className="mt-1 max-w-4xl text-sm leading-6 text-slate-600">{description}</p> : null}
        </div>
      </div>
      {children}
    </section>
  );
}

function PageHeader({ eyebrow, title, description }: { eyebrow: string; title: string; description: string }) {
  return (
    <header>
      <p className="text-sm font-semibold uppercase tracking-[0.18em] text-blueAction">{eyebrow}</p>
      <h1 className="mt-2 text-3xl font-bold text-slateInk">{title}</h1>
      <p className="mt-2 max-w-3xl text-base leading-7 text-slate-600">{description}</p>
    </header>
  );
}

function DecisionItem({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="rounded-xl border border-borderSoft bg-white p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-1 break-words text-sm font-bold text-slateInk">{formatUnknown(value)}</p>
    </div>
  );
}

function MetricTile({ label, value, source }: { label: string; value: unknown; source?: string }) {
  const technical = isTechnicalValue(label, value);
  return (
    <div className="min-w-0 rounded-xl border border-borderSoft bg-panel p-3">
      <p className="flex items-center gap-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
        {label}
        {source ? <Info className="h-3.5 w-3.5" aria-label={`Fonte: ${source}`} /> : null}
      </p>
      <div className="mt-1 min-w-0 text-lg font-bold text-slateInk">
        {technical ? <TechnicalValue value={value} source={source} /> : <span className="break-words">{formatUnknown(value)}</span>}
      </div>
      {source && !technical ? <p className="mt-1 text-xs text-slate-500">Fonte: {source}</p> : null}
    </div>
  );
}

function Definition({ label, value, source }: { label: string; value: unknown; source?: string }) {
  const technical = isTechnicalValue(label, value);
  return (
    <div className="min-w-0">
      <dt className="text-slate-500">{label}</dt>
      <dd className="mt-0.5 min-w-0 font-semibold text-slateInk">
        {technical ? (
          <TechnicalValue value={value} source={source} />
        ) : (
          <>
            <span className="break-words">{formatUnknown(value)}</span>
            {source ? <span className="mt-1 block text-xs font-normal text-slate-500">Fonte: {source}</span> : null}
          </>
        )}
      </dd>
    </div>
  );
}

function IndicatorGroup({ title, indicators, diagnostic = false }: { title: string; indicators: GovernanceIndicator[]; diagnostic?: boolean }) {
  return (
    <div className="rounded-xl border border-borderSoft bg-white p-4">
      <h3 className="font-bold text-slateInk">{title}</h3>
      <div className="mt-3 space-y-3">
        {indicators.length ? (
          indicators.map((indicator) => (
            <div key={indicatorKey(indicator)} className="rounded-lg bg-panel p-3">
              <p className="text-sm font-semibold text-slateInk">{indicatorLabel(indicator)}</p>
              <p className="mt-1 text-xl font-bold text-slateInk">{formatUnknown(indicator.value ?? indicator.valor)}</p>
              <p className="mt-1 text-xs text-slate-500">Fonte: {formatUnknown(indicator.source ?? indicator.fonte)}</p>
              {diagnostic ? <p className="mt-2 text-xs text-slate-600">Métrica diagnóstica; não há limiar obrigatório configurado quando o backend não informar critério.</p> : null}
            </div>
          ))
        ) : (
          <EmptyState title="Não avaliado" text="Esta execução não produziu métricas suficientes para este grupo." compact />
        )}
      </div>
    </div>
  );
}

function StatusBadge({ status, label }: { status: string; label?: string }) {
  const mapped = statusTone(status);
  return (
    <Badge tone={mapped.tone} title={mapped.help}>
      {label ?? mapped.label}
    </Badge>
  );
}

function FilterSelect({
  label,
  value,
  options,
  onChange,
  formatOption = (option) => option
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
  formatOption?: (value: string) => string;
}) {
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
            {formatOption(option)}
          </option>
        ))}
      </select>
    </label>
  );
}

function EmptyState({ title, text, compact = false }: { title: string; text: string; compact?: boolean }) {
  return (
    <div className={`rounded-xl border border-dashed border-borderSoft bg-panel ${compact ? "p-3" : "p-5"}`}>
      <p className="font-bold text-slateInk">{title}</p>
      <p className="mt-1 text-sm leading-6 text-slate-600">{text}</p>
    </div>
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

function GovernanceSkeleton() {
  return (
    <div className="space-y-5" aria-label="Carregando governança">
      <div className="h-52 animate-pulse rounded-2xl bg-slate-200" />
      <div className="grid gap-4 lg:grid-cols-3">
        <div className="h-36 animate-pulse rounded-xl bg-slate-200" />
        <div className="h-36 animate-pulse rounded-xl bg-slate-200" />
        <div className="h-36 animate-pulse rounded-xl bg-slate-200" />
      </div>
      <div className="h-64 animate-pulse rounded-2xl bg-slate-200" />
    </div>
  );
}

function statusTone(status: string): { tone: BadgeTone; label: string; help: string } {
  const normalized = status.toLowerCase();
  if (normalized.includes("recommended_candidate")) {
    return { tone: "candidate", label: "Candidato Recomendado", help: "Candidato recomendado para avaliação técnica posterior." };
  }
  if (["not_approved", "production", "not_defined"].some((item) => normalized.includes(item))) {
    return { tone: "neutral", label: "Não aprovado / não definido", help: "Não há aprovação de produção registrada." };
  }
  if (["not_evaluated"].some((item) => normalized.includes(item))) {
    return { tone: "neutral", label: "Não avaliado", help: "Métrica ou decisão ausente." };
  }
  if (["approved", "aprovado", "success", "default"].some((item) => normalized.includes(item))) {
    return { tone: "approved", label: normalized.includes("default") ? "Padrão Geral" : "Aprovado", help: "Atendeu ao critério interno aplicável." };
  }
  if (["recommended", "recomend"].some((item) => normalized.includes(item))) {
    return { tone: "recommended", label: "Recomendado", help: "Recomendação técnica interna." };
  }
  if (["candidate", "candidato"].some((item) => normalized.includes(item))) {
    return { tone: "candidate", label: "Candidato", help: "Artefato ainda em avaliação." };
  }
  if (["experimental"].some((item) => normalized.includes(item))) {
    return { tone: "experimental", label: "Experimental", help: "Baseline ou uso metodológico controlado." };
  }
  if (["quarantined", "quarantine", "quarentena"].some((item) => normalized.includes(item))) {
    return { tone: "warning", label: "Quarentena", help: "Execução tecnicamente concluída com ressalvas ou métricas informativas." };
  }
  if (["rejected", "reprov"].some((item) => normalized.includes(item))) {
    return { tone: "warning", label: "Reprovado", help: "Não atendeu a pelo menos um critério aplicável." };
  }
  if (["completed", "conclu"].some((item) => normalized.includes(item))) {
    return { tone: "approved", label: "Concluído", help: "Execução concluída." };
  }
  if (["failed", "falh"].some((item) => normalized.includes(item))) {
    return { tone: "warning", label: "Falhou", help: "Execução não concluída." };
  }
  if (["running", "execu"].some((item) => normalized.includes(item))) {
    return { tone: "candidate", label: "Em Execução", help: "Execução em andamento." };
  }
  if (["queued", "fila"].some((item) => normalized.includes(item))) {
    return { tone: "neutral", label: "Na Fila", help: "Execução aguardando processamento." };
  }
  return { tone: "neutral", label: statusDisplayName(status), help: "Status informado pela API." };
}

function statusLabel(status: string): string {
  return statusTone(status).label;
}

function evidenceRoleLabel(role: string): string {
  if (role === "Modelo neural recomendado") return "Modelo Neural Recomendado";
  if (role === "Padrão geral") return "Padrão Geral";
  return role;
}

function matchesGroup(indicator: GovernanceIndicator, terms: string[]) {
  const haystack = `${indicatorLabel(indicator)} ${indicator.metric ?? ""}`.toLowerCase();
  return terms.some((term) => haystack.includes(term));
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

function objectValue(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null ? (value as Record<string, unknown>) : {};
}

function formatCriterion(gate: GovernanceGate) {
  if (gate.operator && gate.threshold !== null && gate.threshold !== undefined) return `${gate.operator} ${formatUnknown(gate.threshold)}`;
  return gate.mandatory ? "Critério obrigatório registrado" : "Diagnóstico";
}

function formatEvidence(value: unknown): string {
  if (value === null || value === undefined) return "Não avaliado";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return formatUnknown(value);
  return JSON.stringify(value, null, 2);
}

function formatMetricMaybeBytes(value: unknown, model: Record<string, unknown>) {
  if (typeof value === "number") return `${value.toFixed(2)} MB`;
  const bytes = model.model_size_bytes;
  return typeof bytes === "number" ? formatBytes(bytes) : formatUnknown(value);
}

function formatTimelineValue(step: unknown, value: unknown): string {
  const normalized = String(step ?? "").toLowerCase();
  if (normalized.includes("decisão") || normalized.includes("status")) return statusDisplayName(value);
  return formatUnknown(value);
}

function indicatorLabel(indicator: GovernanceIndicator) {
  return String(indicator.indicator ?? indicator.indicador ?? "Indicador");
}

function indicatorKey(indicator: GovernanceIndicator) {
  return `${indicator.metric ?? indicator["métrica"] ?? indicatorLabel(indicator)}`;
}

function evaluationStatus(row: Record<string, unknown>) {
  if (row.duplicate_base_row_rate !== null && row.duplicate_base_row_rate !== undefined) return "Avaliada";
  if (row.exact_train_match_rate !== null && row.exact_train_match_rate !== undefined) return "Avaliada";
  return "Não avaliada";
}

function eventRank(event: Record<string, unknown>) {
  const name = String(event.event_type ?? event.event ?? "");
  const index = auditPriority.indexOf(name);
  return index === -1 ? auditPriority.length : index;
}

function eventDescription(event: Record<string, unknown>) {
  const name = String(event.event_type ?? event.event ?? "Evento");
  const descriptions: Record<string, string> = {
    generation_started: "Geração iniciada",
    generation_completed: "Geração concluída",
    generation_succeeded: "Geração concluída",
    model_selected: "Modelo selecionado",
    configuration_changed: "Configuração alterada",
    artifact_created: "Artefato criado",
    evaluation_completed: "Avaliação concluída",
    gate_passed: "Gate aprovado",
    gate_failed: "Gate reprovado",
    artifact_quarantined: "Artefato em quarentena",
    model_promoted: "Modelo promovido",
    model_demoted: "Modelo rebaixado",
    export_completed: "Exportação concluída",
    configuration_restored: "Configuração restaurada"
  };
  return descriptions[name] ?? name;
}
