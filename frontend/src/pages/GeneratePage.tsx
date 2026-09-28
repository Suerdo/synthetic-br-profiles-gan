import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { CheckCircle2, Download, Loader2, RefreshCw } from "lucide-react";

import {
  createGeneration,
  downloadDataset,
  downloadManifest,
  getColumns,
  getGeneration,
  getGenerationPreview,
  getHealth,
  getModelArtifacts,
  getModels
} from "../api";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { Callout } from "../components/ui/Callout";
import { Card, SectionHeader } from "../components/ui/Card";
import type { GenerationStatusResponse, ModelArtifact, ModelName, OutputFormat } from "../types/api";
import { formatBytes, formatNumber, safeFilename } from "../utils/format";

interface GeneratePageProps {
  sessionId: string;
}

const modelOrder: ModelName[] = ["programmatic", "ctgan", "simple_gan"];
const formats: Array<{ value: OutputFormat; label: string; description: string }> = [
  { value: "csv", label: "CSV", description: "Compatível com planilhas e ferramentas de análise." },
  { value: "json", label: "JSON", description: "Adequado para integrações e desenvolvimento." },
  { value: "parquet", label: "Parquet", description: "Indicado para análise com preservação de tipos." }
];

export function GeneratePage({ sessionId }: GeneratePageProps) {
  const health = useQuery({ queryKey: ["health"], queryFn: getHealth });
  const columns = useQuery({ queryKey: ["columns"], queryFn: getColumns });
  const models = useQuery({ queryKey: ["models"], queryFn: getModels });

  const [selectedModel, setSelectedModel] = useState<ModelName>("programmatic");
  const [selectedArtifactId, setSelectedArtifactId] = useState<string | null>(null);
  const [rows, setRows] = useState(1000);
  const [seed, setSeed] = useState(41);
  const [format, setFormat] = useState<OutputFormat>("csv");
  const [columnMode, setColumnMode] = useState<"preset" | "custom">("preset");
  const [preset, setPreset] = useState("completo");
  const [customColumns, setCustomColumns] = useState<string[]>([]);
  const [generationId, setGenerationId] = useState<string | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  const artifacts = useQuery({
    queryKey: ["artifacts", selectedModel],
    queryFn: () => getModelArtifacts(selectedModel),
    enabled: selectedModel !== "programmatic"
  });

  useEffect(() => {
    if (health.data) {
      setSelectedModel((current) => current ?? health.data.defaults.model);
      setRows((current) => current || health.data.defaults.rows);
      setSeed((current) => current || health.data.defaults.seed);
      setFormat((current) => current || health.data.defaults.format);
    }
  }, [health.data]);

  useEffect(() => {
    if (columns.data) {
      setPreset((current) => current || columns.data.default_preset);
      setCustomColumns((current) => (current.length > 0 ? current : columns.data.final_columns));
    }
  }, [columns.data]);

  useEffect(() => {
    if (selectedModel === "programmatic") {
      setSelectedArtifactId(null);
      return;
    }
    if (artifacts.data?.artifacts.length) {
      const recommended = artifacts.data.recommended_artifact_id;
      setSelectedArtifactId((current) => {
        if (current && artifacts.data.artifacts.some((artifact) => artifact.artifact_id === current)) {
          return current;
        }
        return recommended ?? artifacts.data.artifacts[0].artifact_id;
      });
    } else {
      setSelectedArtifactId(null);
    }
  }, [artifacts.data, selectedModel]);

  const modelMap = useMemo(() => {
    return new Map(models.data?.models.map((model) => [model.name, model]) ?? []);
  }, [models.data]);
  const selectedModelEntry = modelMap.get(selectedModel);
  const selectedArtifact = artifacts.data?.artifacts.find((artifact) => artifact.artifact_id === selectedArtifactId) ?? null;
  const selectedColumns =
    columnMode === "preset" ? columns.data?.presets.find((item) => item.name === preset)?.columns ?? [] : customColumns;
  const canGenerate =
    Boolean(columns.data && selectedModelEntry) &&
    rows >= 1 &&
    rows <= (selectedModelEntry?.row_limit ?? 0) &&
    selectedColumns.length > 0 &&
    (selectedModel === "programmatic" || Boolean(selectedArtifactId));

  const startGeneration = useMutation({
    mutationFn: () =>
      createGeneration(
        {
          model: selectedModel,
          artifact_id: selectedModel === "programmatic" ? null : selectedArtifactId,
          num_rows: rows,
          output_format: format,
          seed,
          selected_columns: columnMode === "custom" ? selectedColumns : null,
          column_preset: columnMode === "preset" ? preset : null
        },
        sessionId
      ),
    onSuccess: (response) => setGenerationId(response.generation_id)
  });

  const status = useQuery({
    queryKey: ["generation", generationId, sessionId],
    queryFn: () => getGeneration(generationId!, sessionId),
    enabled: Boolean(generationId),
    refetchInterval: (query) => {
      const data = query.state.data as GenerationStatusResponse | undefined;
      return data?.status === "completed" || data?.status === "failed" ? false : 1200;
    }
  });

  const preview = useQuery({
    queryKey: ["generation-preview", generationId, sessionId],
    queryFn: () => getGenerationPreview(generationId!, sessionId),
    enabled: Boolean(generationId && status.data?.status === "completed")
  });

  async function handleDownload(kind: "dataset" | "manifest") {
    if (!generationId) return;
    setDownloadError(null);
    try {
      const blob = kind === "dataset" ? await downloadDataset(generationId, sessionId) : await downloadManifest(generationId, sessionId);
      const extension = kind === "dataset" ? format : "manifest.json";
      const filename = safeFilename(`perfis-sinteticos-${selectedModel}-${rows}.${extension}`);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      link.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      setDownloadError(error instanceof Error ? error.message : "Falha ao baixar arquivo.");
    }
  }

  return (
    <div className="space-y-6">
      <header>
        <p className="text-sm font-semibold uppercase tracking-[0.18em] text-blueAction">Gerar dados</p>
        <h1 className="mt-2 text-3xl font-bold text-slateInk">Gerador de perfis sintéticos brasileiros</h1>
        <p className="mt-2 max-w-3xl text-base leading-7 text-slate-600">
          Escolha o modelo, o volume, as colunas e o formato. A aplicação mantém internamente o schema completo e
          usa o serviço de geração do projeto para validar e exportar o dataset.
        </p>
      </header>

      <Callout>
        <strong>Atenção:</strong> Os dados gerados são sintéticos e não foram consultados ou validados em bases
        oficiais. A validade estrutural de documentos não comprova existência, regularidade ou associação a uma
        pessoa real. Os dados não devem ser utilizados para fraude, autenticação, identificação real ou acesso a
        serviços.
        <br />
        A plataforma auxilia atividades de desenvolvimento de software, testes, engenharia de requisitos,
        demonstrações técnicas, pesquisa acadêmica, treinamento de modelos de Inteligência Artificial e validação de
        pipelines de dados. Embora utilize técnicas de geração de dados sintéticos, não oferece garantia absoluta de
        anonimização nem elimina completamente a possibilidade de coincidências estatísticas com informações reais.
      </Callout>

      <section>
        <SectionHeader step={1} title="Modelo">
          O programático gera diretamente. CTGAN e GAN simples usam somente artefatos válidos administrados pela
          aplicação.
        </SectionHeader>
        <div className="grid gap-4 md:grid-cols-3">
          {modelOrder.map((modelName) => {
            const model = modelMap.get(modelName);
            const active = selectedModel === modelName;
            return (
              <button
                key={modelName}
                type="button"
                onClick={() => setSelectedModel(modelName)}
                className={`rounded-xl border bg-white p-4 text-left shadow-card transition hover:border-navy ${
                  active ? "border-navy ring-2 ring-blue-100" : "border-borderSoft"
                }`}
              >
                <div className="flex flex-wrap items-center gap-2">
                  <h3 className="text-lg font-bold text-slateInk">{model?.label ?? modelName}</h3>
                  {model?.recommended ? <Badge tone="recommended">Recomendado</Badge> : null}
                  {model?.experimental ? <Badge tone="experimental">Experimental</Badge> : null}
                  {!model?.available ? <Badge tone="warning">Indisponível</Badge> : null}
                </div>
                <p className="mt-3 text-sm leading-6 text-slate-600">{model?.short_description ?? "Carregando..."}</p>
                <p className="mt-3 text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Limite operacional: {formatNumber(model?.row_limit)} registros
                </p>
              </button>
            );
          })}
        </div>
        {selectedModelEntry?.availability_message ? (
          <div className="mt-4">
            <Callout tone="info">{selectedModelEntry.availability_message}</Callout>
          </div>
        ) : null}
        {selectedModel !== "programmatic" ? (
          <Card className="mt-4">
            <label className="block text-sm font-bold text-slateInk" htmlFor="artifact-select">
              Artefato do modelo
            </label>
            <select
              id="artifact-select"
              className="mt-2 w-full rounded-lg border border-slate-500 bg-[#E8EEF7] px-3 py-2 text-sm text-slateInk focus:border-navy focus:outline-none focus:ring-2 focus:ring-blue-200"
              value={selectedArtifactId ?? ""}
              onChange={(event) => setSelectedArtifactId(event.target.value)}
              disabled={!artifacts.data?.artifacts.length}
            >
              {artifacts.data?.artifacts.map((artifact) => (
                <option key={artifact.artifact_id} value={artifact.artifact_id}>
                  {artifact.label}
                </option>
              ))}
            </select>
            {selectedArtifact ? (
              <div className="mt-3 flex flex-wrap gap-2 text-sm">
                <Badge tone={selectedArtifact.status === "Aprovado" ? "approved" : "candidate"}>{selectedArtifact.status}</Badge>
                <Badge>Vocabulário v{selectedArtifact.categorical_vocabulary_version}</Badge>
                <Badge>Renda v{selectedArtifact.income_model_version}</Badge>
                <Badge>Geografia v{selectedArtifact.geography_model_version}</Badge>
              </div>
            ) : null}
            {selectedArtifact?.warning ? <p className="mt-3 text-sm text-orange-800">{selectedArtifact.warning}</p> : null}
          </Card>
        ) : null}
      </section>

      <section>
        <SectionHeader step={2} title="Volume e reprodutibilidade">
          A seed ajuda a reproduzir a geração. Backends neurais podem variar conforme hardware e bibliotecas.
        </SectionHeader>
        <div className="grid gap-4 md:grid-cols-2">
          <Card>
            <label className="block text-sm font-bold text-slateInk" htmlFor="rows">
              Quantidade de registros
            </label>
            <input
              id="rows"
              type="number"
              min={1}
              max={selectedModelEntry?.row_limit ?? 1000}
              value={rows}
              onChange={(event) => setRows(Number(event.target.value))}
              className="mt-2 w-full rounded-lg border border-slate-500 bg-[#E8EEF7] px-3 py-2 text-sm text-slateInk focus:border-navy focus:outline-none focus:ring-2 focus:ring-blue-200"
            />
          </Card>
          <Card>
            <label className="block text-sm font-bold text-slateInk" htmlFor="seed">
              Seed
            </label>
            <input
              id="seed"
              type="number"
              min={0}
              value={seed}
              onChange={(event) => setSeed(Number(event.target.value))}
              className="mt-2 w-full rounded-lg border border-slate-500 bg-[#E8EEF7] px-3 py-2 text-sm text-slateInk focus:border-navy focus:outline-none focus:ring-2 focus:ring-blue-200"
            />
          </Card>
        </div>
      </section>

      <section>
        <SectionHeader step={3} title="Colunas">
          As dependências necessárias são geradas internamente, mas somente as colunas selecionadas são exportadas.
        </SectionHeader>
        <Card>
          <div className="flex gap-3">
            <Button variant={columnMode === "preset" ? "primary" : "secondary"} onClick={() => setColumnMode("preset")}>
              Preset
            </Button>
            <Button variant={columnMode === "custom" ? "primary" : "secondary"} onClick={() => setColumnMode("custom")}>
              Personalizado
            </Button>
          </div>
          {columnMode === "preset" ? (
            <div className="mt-4">
              <label className="block text-sm font-bold text-slateInk" htmlFor="preset">
                Preset
              </label>
              <select
                id="preset"
                value={preset}
                onChange={(event) => setPreset(event.target.value)}
                className="mt-2 w-full rounded-lg border border-slate-500 bg-[#E8EEF7] px-3 py-2 text-sm text-slateInk focus:border-navy focus:outline-none focus:ring-2 focus:ring-blue-200"
              >
                {columns.data?.presets.map((item) => (
                  <option key={item.name} value={item.name}>
                    {item.name} — {item.columns.length} colunas
                  </option>
                ))}
              </select>
            </div>
          ) : (
            <div className="mt-4 grid gap-4 md:grid-cols-2">
              {columns.data?.groups.map((group) => (
                <fieldset key={group} className="rounded-lg border border-borderSoft bg-panel p-4">
                  <legend className="px-1 text-sm font-bold text-slateInk">{group}</legend>
                  <div className="mt-2 space-y-2">
                    {columns.data.columns
                      .filter((column) => column.group === group)
                      .map((column) => (
                        <label key={column.name} className="flex gap-2 text-sm text-slate-700">
                          <input
                            type="checkbox"
                            className="mt-1 h-4 w-4 rounded border-slate-500 text-navy focus:ring-blue-200"
                            checked={customColumns.includes(column.name)}
                            onChange={(event) => {
                              setCustomColumns((current) =>
                                event.target.checked
                                  ? columns.data!.final_columns.filter((name) => [...current, column.name].includes(name))
                                  : current.filter((name) => name !== column.name)
                              );
                            }}
                          />
                          <span>
                            <strong>{column.label}</strong> <span className="text-slate-500">({column.name})</span>
                            <span className="block text-xs leading-5 text-slate-500">{column.description}</span>
                          </span>
                        </label>
                      ))}
                  </div>
                </fieldset>
              ))}
            </div>
          )}
        </Card>
      </section>

      <section>
        <SectionHeader step={4} title="Formato">
          Escolha o formato de exportação do arquivo.
        </SectionHeader>
        <div className="grid gap-4 md:grid-cols-3">
          {formats.map((item) => (
            <button
              key={item.value}
              type="button"
              onClick={() => setFormat(item.value)}
              className={`rounded-xl border bg-white p-4 text-left shadow-card transition hover:border-navy ${
                format === item.value ? "border-navy ring-2 ring-blue-100" : "border-borderSoft"
              }`}
            >
              <h3 className="font-bold text-slateInk">{item.label}</h3>
              <p className="mt-2 text-sm leading-6 text-slate-600">{item.description}</p>
            </button>
          ))}
        </div>
      </section>

      <section>
        <SectionHeader step={5} title="Revisão">
          Confira a solicitação antes de iniciar a geração assíncrona.
        </SectionHeader>
        <Card>
          <div className="grid gap-4 text-sm md:grid-cols-4">
            <Summary label="Modelo" value={selectedModel} />
            <Summary label="Registros" value={formatNumber(rows)} />
            <Summary label="Colunas exportadas" value={formatNumber(selectedColumns.length)} />
            <Summary label="Formato" value={format.toUpperCase()} />
          </div>
          <div className="mt-5 flex flex-wrap items-center gap-3">
            <Button
              disabled={!canGenerate || startGeneration.isPending || status.data?.status === "running"}
              onClick={() => startGeneration.mutate()}
            >
              {startGeneration.isPending || status.data?.status === "running" || status.data?.status === "queued" ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4" />
              )}
              Gerar dados sintéticos
            </Button>
            {!canGenerate ? <span className="text-sm text-orange-800">Revise modelo, artefato e colunas selecionadas.</span> : null}
          </div>
          {startGeneration.error ? <p className="mt-3 text-sm text-red-700">{startGeneration.error.message}</p> : null}
        </Card>
      </section>

      {status.data ? (
        <ResultPanel
          generation={status.data}
          preview={preview.data}
          selectedArtifact={selectedArtifact}
          downloadError={downloadError}
          onDownloadDataset={() => handleDownload("dataset")}
          onDownloadManifest={() => handleDownload("manifest")}
        />
      ) : null}
    </div>
  );
}

function Summary({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-panel p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-1 font-bold text-slateInk">{value}</p>
    </div>
  );
}

function ResultPanel({
  generation,
  preview,
  selectedArtifact,
  downloadError,
  onDownloadDataset,
  onDownloadManifest
}: {
  generation: GenerationStatusResponse;
  preview?: { columns: string[]; rows: Array<Record<string, unknown>> };
  selectedArtifact: ModelArtifact | null;
  downloadError: string | null;
  onDownloadDataset: () => void;
  onDownloadManifest: () => void;
}) {
  const completed = generation.status === "completed";
  return (
    <Card className="border-blue-200">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-xl font-bold text-slateInk">Resultado da geração</h2>
          <p className="text-sm text-slate-600">Status: {generation.status}</p>
        </div>
        {completed ? (
          <Badge tone="approved">
            <CheckCircle2 className="mr-1 h-4 w-4" />
            Concluída
          </Badge>
        ) : generation.status === "failed" ? (
          <Badge tone="warning">Falha</Badge>
        ) : (
          <Badge tone="candidate">Em processamento</Badge>
        )}
      </div>
      {generation.error ? <p className="mt-4 text-sm text-red-700">{generation.error.message}</p> : null}
      <div className="mt-5 grid gap-4 text-sm md:grid-cols-4">
        <Summary label="Modelo" value={generation.model} />
        <Summary label="Artefato" value={selectedArtifact?.status ?? "Geração direta"} />
        <Summary label="Registros" value={formatNumber(generation.num_rows)} />
        <Summary label="Duração" value={generation.duration_seconds ? `${generation.duration_seconds.toFixed(2)} s` : "Em andamento"} />
      </div>
      {preview ? (
        <div className="mt-6 overflow-x-auto">
          <h3 className="mb-3 font-bold text-slateInk">Amostra</h3>
          <table className="min-w-full border-collapse text-left text-sm">
            <thead>
              <tr className="bg-slate-200 text-slateInk">
                {preview.columns.map((column) => (
                  <th key={column} className="border border-borderSoft px-3 py-2">
                    {column}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview.rows.map((row, index) => (
                <tr key={index} className="odd:bg-white even:bg-panel">
                  {preview.columns.map((column) => (
                    <td key={column} className="border border-borderSoft px-3 py-2 text-slate-700">
                      {String(row[column] ?? "")}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
      {completed ? (
        <div className="mt-6 flex flex-wrap gap-3">
          <Button onClick={onDownloadDataset}>
            <Download className="h-4 w-4" />
            Baixar dataset
          </Button>
          <Button variant="secondary" onClick={onDownloadManifest}>
            <Download className="h-4 w-4" />
            Baixar manifesto
          </Button>
        </div>
      ) : null}
      {downloadError ? <p className="mt-3 text-sm text-red-700">{downloadError}</p> : null}
    </Card>
  );
}
