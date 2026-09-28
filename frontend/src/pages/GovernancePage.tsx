import { useQuery } from "@tanstack/react-query";
import type React from "react";

import { getHealth, getModelArtifacts, getModels } from "../api";
import { Badge } from "../components/ui/Badge";
import { Callout } from "../components/ui/Callout";
import { Card } from "../components/ui/Card";
import { formatDate, formatNumber } from "../utils/format";

export function GovernancePage() {
  const health = useQuery({ queryKey: ["health"], queryFn: getHealth });
  const models = useQuery({ queryKey: ["models"], queryFn: getModels });
  const ctganArtifacts = useQuery({ queryKey: ["artifacts", "ctgan"], queryFn: () => getModelArtifacts("ctgan") });
  const recommendedCtgan = ctganArtifacts.data?.artifacts.find((artifact) => artifact.recommended_for_neural_generation);

  return (
    <div className="space-y-6">
      <header>
        <p className="text-sm font-semibold uppercase tracking-[0.18em] text-blueAction">Governança</p>
        <h1 className="mt-2 text-3xl font-bold text-slateInk">Rastreabilidade e uso responsável</h1>
        <p className="mt-2 max-w-3xl text-base leading-7 text-slate-600">
          Esta primeira versão React apresenta os metadados disponíveis pela API e preserva avisos de uso. A leitura
          completa de execuções históricas será ampliada nas próximas fases.
        </p>
      </header>

      <Callout>
        Dados sintéticos devem permanecer identificados como sintéticos e não substituem avaliação de governança,
        privacidade ou uso responsável. A aprovação de artefatos é uma decisão técnica interna e não constitui
        certificação externa, garantia de anonimização ou validação populacional oficial.
      </Callout>

      <section className="grid gap-4 md:grid-cols-3">
        <MetricCard
          label="Status operacional"
          value={health.data?.status === "ok" ? "Operacional" : "Não avaliado"}
          description="Situação técnica do endpoint de saúde da API."
          source="Fonte: /api/health"
        />
        <MetricCard
          label="Modelos disponíveis"
          value={models.data ? formatNumber(models.data.models.filter((model) => model.available).length) : "Não avaliado"}
          description="Quantidade de tipos de modelo utilizáveis na interface."
          source="Fonte: /api/models"
        />
        <MetricCard
          label="Artefato neural recomendado"
          value={recommendedCtgan ? "Disponível" : "Não avaliado"}
          description="CTGAN aprovada internamente para geração neural, quando presente."
          source="Fonte: /api/models/ctgan/artifacts"
        />
      </section>

      <Card>
        <h2 className="text-xl font-bold text-slateInk">Modelo Neural Recomendado</h2>
        {recommendedCtgan ? (
          <div className="mt-4 grid gap-4 text-sm md:grid-cols-3">
            <Summary label="Identificador" value={recommendedCtgan.artifact_id} />
            <Summary label="Status" value={recommendedCtgan.status} />
            <Summary label="Criado em" value={formatDate(recommendedCtgan.created_at_utc)} />
            <Summary label="Vocabulário" value={`v${recommendedCtgan.categorical_vocabulary_version}`} />
            <Summary label="Renda" value={`v${recommendedCtgan.income_model_version}`} />
            <Summary label="Geografia" value={`v${recommendedCtgan.geography_model_version}`} />
            <Summary label="Checksum geográfico" value={recommendedCtgan.geography_catalog_checksum ?? "Não avaliado"} />
            <Summary label="Treino" value={recommendedCtgan.train_rows ? formatNumber(recommendedCtgan.train_rows) : "Não avaliado"} />
            <Summary label="Seed" value={recommendedCtgan.seed !== null ? String(recommendedCtgan.seed) : "Não avaliado"} />
          </div>
        ) : (
          <p className="mt-3 text-sm text-slate-600">Nenhuma CTGAN recomendada foi encontrada nos artefatos válidos.</p>
        )}
      </Card>

      <section className="grid gap-4 lg:grid-cols-2">
        <Card>
          <h2 className="text-xl font-bold text-slateInk">Diversidade e Memorização</h2>
          <p className="mt-2 text-sm leading-6 text-slate-600">
            As métricas históricas detalhadas ainda não são carregadas nesta primeira fase da API React.
          </p>
          <div className="mt-4 grid gap-3">
            <AbsentMetric label="Duplicidade de combinações-base" source="evaluation.json → privacy → duplicate_base_rows" />
            <AbsentMetric label="Correspondência exata com treino" source="evaluation.json → privacy → exact_matches → train" />
            <AbsentMetric label="Correspondência exata com holdout" source="evaluation.json → privacy → exact_matches → holdout" />
          </div>
        </Card>
        <Card>
          <h2 className="text-xl font-bold text-slateInk">Realismo Condicional</h2>
          <p className="mt-2 text-sm leading-6 text-slate-600">
            Percentis condicionais e eventos de cauda serão integrados em endpoint próprio nas próximas fases.
          </p>
          <div className="mt-4 grid gap-3">
            <AbsentMetric label="Mediana de renda por ocupação" source="conditional_income_summary.csv" />
            <AbsentMetric label="P95/P99 por ocupação" source="income_plausibility_summary.json" />
            <AbsentMetric label="Status da avaliação condicional" source="evaluation.json → income_realism" />
          </div>
        </Card>
      </section>

      <Card>
        <details open>
          <summary className="cursor-pointer text-lg font-bold text-slateInk">Como interpretar os indicadores</summary>
          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <Glossary title="Execuções registradas">
              Quantidade de manifestos de execução identificados pela aplicação.
            </Glossary>
            <Glossary title="Aprovadas">
              Execuções sem falha nos quality gates obrigatórios.
            </Glossary>
            <Glossary title="Em quarentena">
              Execuções tecnicamente concluídas, mas com alertas ou falhas em métricas informativas.
            </Glossary>
            <Glossary title="Rejeitadas">
              Execuções que falharam em pelo menos um gate obrigatório.
            </Glossary>
            <Glossary title="Duplicidade de combinações-base">
              Repetição exata das 11 colunas-base produzidas pelo modelo. Identificadores derivados não participam.
            </Glossary>
            <Glossary title="Correspondência exata com treino">
              Percentual de registros sintéticos cujas 11 colunas-base coincidem com pelo menos um registro de treinamento.
            </Glossary>
            <Glossary title="Correspondência exata com holdout">
              Métrica de controle para distinguir memorização de coincidências inerentes à distribuição.
            </Glossary>
            <Glossary title="Realismo condicional">
              Capacidade de preservar distribuições dentro de contextos específicos, como renda por ocupação e escolaridade.
            </Glossary>
            <Glossary title="Risco de privacidade">
              Classificação derivada de métricas explícitas disponíveis. Não constitui garantia de anonimização.
            </Glossary>
          </div>
        </details>
      </Card>
    </div>
  );
}

function MetricCard({ label, value, description, source }: { label: string; value: string; description: string; source: string }) {
  return (
    <Card>
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-slate-600">{label}</p>
          <p className="mt-2 text-2xl font-bold text-slateInk">{value}</p>
        </div>
        <Badge tone={value === "Não avaliado" ? "neutral" : "approved"}>{value === "Não avaliado" ? "Não avaliado" : "OK"}</Badge>
      </div>
      <p className="mt-3 text-sm leading-6 text-slate-600">{description}</p>
      <p className="mt-2 text-xs font-semibold text-slate-500">{source}</p>
    </Card>
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

function AbsentMetric({ label, source }: { label: string; source: string }) {
  return (
    <div className="rounded-lg border border-borderSoft bg-panel p-3 text-sm">
      <div className="flex items-center justify-between gap-3">
        <strong className="text-slateInk">{label}</strong>
        <Badge>Não avaliado</Badge>
      </div>
      <p className="mt-2 text-slate-600">Esta execução foi produzida antes da inclusão desta métrica ou não contém os artefatos necessários.</p>
      <p className="mt-1 text-xs font-semibold text-slate-500">Fonte: {source}</p>
    </div>
  );
}

function Glossary({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg bg-panel p-3 text-sm">
      <h3 className="font-bold text-slateInk">{title}</h3>
      <p className="mt-1 leading-6 text-slate-600">{children}</p>
    </div>
  );
}
