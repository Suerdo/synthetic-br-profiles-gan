import { useQuery } from "@tanstack/react-query";

import { getModelArtifacts, getModels } from "../api";
import { Badge } from "../components/ui/Badge";
import { Card } from "../components/ui/Card";
import type { ModelName } from "../types/api";
import { formatBytes, formatDate, formatNumber } from "../utils/format";

const modelOrder: ModelName[] = ["programmatic", "ctgan", "simple_gan"];

export function ModelsPage() {
  const models = useQuery({ queryKey: ["models"], queryFn: getModels });

  return (
    <div className="space-y-6">
      <header>
        <p className="text-sm font-semibold uppercase tracking-[0.18em] text-blueAction">Modelos</p>
        <h1 className="mt-2 text-3xl font-bold text-slateInk">Conheça os sintetizadores</h1>
        <p className="mt-2 max-w-3xl text-base leading-7 text-slate-600">
          A interface React apresenta os três modelos do projeto, preservando seus papéis: programático como padrão
          geral, CTGAN aprovada como opção neural recomendada e GAN simples como baseline experimental.
        </p>
      </header>

      <div className="grid gap-4 md:grid-cols-3">
        <Comparison label="Programático" training="Não exige" cost="Baixo" status="Recomendado" />
        <Comparison label="CTGAN" training="Exige artefato" cost="Alto" status="Neural recomendado" />
        <Comparison label="GAN simples" training="Exige artefato" cost="Médio" status="Experimental" />
      </div>

      {modelOrder.map((modelName) => {
        const entry = models.data?.models.find((model) => model.name === modelName);
        if (!entry) return null;
        return <ModelSection key={entry.name} modelName={entry.name} />;
      })}
    </div>
  );
}

function Comparison({ label, training, cost, status }: { label: string; training: string; cost: string; status: string }) {
  return (
    <Card>
      <h2 className="font-bold text-slateInk">{label}</h2>
      <dl className="mt-3 space-y-2 text-sm text-slate-600">
        <div className="flex justify-between gap-3">
          <dt>Treinamento</dt>
          <dd className="font-semibold text-slateInk">{training}</dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt>Custo</dt>
          <dd className="font-semibold text-slateInk">{cost}</dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt>Situação</dt>
          <dd className="font-semibold text-slateInk">{status}</dd>
        </div>
      </dl>
    </Card>
  );
}

function ModelSection({ modelName }: { modelName: ModelName }) {
  const models = useQuery({ queryKey: ["models"], queryFn: getModels });
  const artifacts = useQuery({ queryKey: ["artifacts", modelName], queryFn: () => getModelArtifacts(modelName) });
  const entry = models.data?.models.find((model) => model.name === modelName);
  if (!entry) return null;

  return (
    <Card>
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="text-2xl font-bold text-slateInk">{entry.label}</h2>
        {entry.recommended ? <Badge tone="recommended">Recomendado</Badge> : null}
        {entry.experimental ? <Badge tone="experimental">Experimental</Badge> : null}
        {entry.recommended_artifact ? <Badge tone="approved">Artefato neural recomendado</Badge> : null}
      </div>
      <p className="mt-3 max-w-4xl text-sm leading-6 text-slate-600">{entry.detailed_description}</p>
      <div className="mt-5 grid items-stretch gap-4 md:grid-cols-2">
        <div className="h-full min-h-[190px] rounded-xl border border-borderSoft bg-panel p-4">
          <h3 className="font-bold text-slateInk">Resumo simples</h3>
          <p className="mt-2 text-sm leading-6 text-slate-600">{String(entry.metadata.realism_profile ?? entry.short_description)}</p>
        </div>
        <div className="h-full min-h-[190px] rounded-xl border border-borderSoft bg-panel p-4">
          <h3 className="font-bold text-slateInk">Resumo técnico</h3>
          <p className="mt-2 text-sm leading-6 text-slate-600">{entry.short_description}</p>
        </div>
      </div>
      <h3 className="mt-6 font-bold text-slateInk">Usos recomendados</h3>
      <p className="mt-2 text-sm leading-6 text-slate-700">{indicatedFor(entry.name)}</p>
      <div className="mt-4 grid gap-4 md:grid-cols-2">
        <details className="rounded-lg border border-borderSoft bg-white p-4">
          <summary className="cursor-pointer font-semibold text-slateInk">Ver benefícios</summary>
          <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-slate-600">
            {entry.benefits.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </details>
        <details className="rounded-lg border border-borderSoft bg-white p-4">
          <summary className="cursor-pointer font-semibold text-slateInk">Ver limitações</summary>
          <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-slate-600">
            {entry.limitations.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </details>
      </div>
      {entry.requires_saved_artifact ? (
        <details className="mt-4 rounded-lg border border-borderSoft bg-white p-4">
          <summary className="cursor-pointer font-semibold text-slateInk">Ver artefatos disponíveis</summary>
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
                  <th className="border border-borderSoft px-3 py-2">Tamanho</th>
                </tr>
              </thead>
              <tbody>
                {artifacts.data?.artifacts.map((artifact) => (
                  <tr key={artifact.artifact_id} className="odd:bg-white even:bg-panel">
                    <td className="border border-borderSoft px-3 py-2">{artifact.artifact_id}</td>
                    <td className="border border-borderSoft px-3 py-2">{formatDate(artifact.created_at_utc)}</td>
                    <td className="border border-borderSoft px-3 py-2">v{artifact.categorical_vocabulary_version}</td>
                    <td className="border border-borderSoft px-3 py-2">v{artifact.income_model_version}</td>
                    <td className="border border-borderSoft px-3 py-2">v{artifact.geography_model_version}</td>
                    <td className="border border-borderSoft px-3 py-2">{artifact.status}</td>
                    <td className="border border-borderSoft px-3 py-2">{formatBytes(artifact.model_size_bytes)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!artifacts.data?.artifacts.length ? <p className="mt-3 text-sm text-slate-600">Nenhum artefato válido disponível.</p> : null}
          </div>
        </details>
      ) : null}
    </Card>
  );
}

function indicatedFor(model: ModelName): string {
  if (model === "ctgan") {
    return "Indicado para: Experimentos tabulares, comparação com baseline programático, pesquisa metodológica e uso de artefatos treinados e avaliados pela equipe responsável.";
  }
  if (model === "simple_gan") {
    return "Indicado para: Estudos acadêmicos sobre GANs, comparação de arquiteturas, análise de colapso categórico e experimentos metodológicos controlados.";
  }
  return "Indicado para: Desenvolvimento, testes funcionais, demonstrações técnicas, validação de pipelines, ensino e grandes volumes locais.";
}
