import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import type { ModelArtifact, ModelName } from "./types/api";

const health = {
  status: "ok",
  api_version: "0.1.0",
  defaults: { model: "programmatic", rows: 1000, format: "csv", seed: 41, preset: "completo" },
  models: {
    programmatic: { available: true, requires_artifact: false },
    ctgan: { available: true, requires_artifact: true },
    simple_gan: { available: true, requires_artifact: true }
  }
};

const columns = {
  final_columns: ["Nome", "Idade", "Estado", "CPF"],
  groups: ["Identificação sintética", "Demografia", "Localização e contato"],
  columns: [
    column("Nome", "Identificação sintética", "postprocessing", true),
    column("Idade", "Demografia", "model", false),
    column("Estado", "Localização e contato", "model", false),
    column("CPF", "Identificação sintética", "postprocessing", true)
  ],
  presets: [
    { name: "completo", columns: ["Nome", "Idade", "Estado", "CPF"] },
    { name: "minimo", columns: ["Nome", "Idade", "Estado", "CPF"] }
  ],
  default_preset: "completo"
};

const ctganApproved = artifact("ctgan/approved", "Aprovado", true);
const ctganSmoke = artifact("ctgan/smoke", "Smoke", false);
const simpleGanSmoke = artifact("simple_gan/smoke", "Smoke", false, "simple_gan");

const models = {
  default_model: "programmatic",
  models: [
    model("programmatic", "Programático — Recomendado", false, true, false, null),
    model("ctgan", "CTGAN — Artefato Neural Recomendado", true, true, false, ctganApproved),
    model("simple_gan", "GAN Simples — Experimental", true, true, true, null)
  ]
};

const ctganArtifacts = {
  model: "ctgan",
  recommended_artifact_id: "ctgan/approved",
  artifacts: [ctganApproved, ctganSmoke]
};

const simpleGanArtifacts = {
  model: "simple_gan",
  recommended_artifact_id: null,
  artifacts: [simpleGanSmoke]
};

const governance = {
  operational: {
    metrics: [
      { label: "Modelos disponíveis", key: "available_models", value: "Programático, CTGAN, GAN Simples", source: "ModelRegistry", help: "Modelos utilizáveis." },
      { label: "Modelo padrão geral", key: "default_model", value: "programmatic", source: "configs/ui.yaml", help: "Modelo inicial da interface." },
      { label: "Modelo neural recomendado", key: "recommended_neural_model", value: "ctgan/approved", source: "ModelRegistry", help: "Artefato neural recomendado." },
      { label: "Execuções avaliadas", key: "total_runs", value: 3, source: "manifestos", help: "Execuções locais." }
    ],
    summary: {}
  },
  recommended_model: {
    artifact_id: "ctgan/approved",
    status: "approved",
    vocabulary_version: 2,
    income_model_version: 3,
    geography_model_version: 2,
    geography_catalog_checksum: "checksum",
    benchmark: "ctgan-confirmation",
    seeds: [47, 48, 49],
    result: "3/3 seeds aprovadas",
    metrics: {
      raw_geographic_validity_rate: { min: 1, max: 1 },
      raw_global_validity_rate: { min: 0.9156, max: 0.9685 },
      duplicate_base_row_rate: 0,
      exact_train_match_rate: 0,
      geography_key_coverage: 1
    },
    limitations: ["Diretor ausente na seed 48."],
    environment: { summary: "Python 3.13; CTGAN 0.12.1" },
    approval_note:
      "A aprovação representa uma decisão técnica interna baseada nos critérios do projeto. Ela não constitui certificação externa, garantia de anonimização ou validação populacional oficial."
  },
  quality: {
    status: "Disponível",
    indicators: [
      indicator("Validade estrutural", "validation.is_valid", true, "validation.json", "Schema final válido.", "mandatory"),
      indicator("Cobertura categórica", "coverage", null, "evaluation.json", "Esta execução não produziu essa métrica.", "informational")
    ]
  },
  privacy: {
    status: "Disponível",
    indicators: [],
    diversity_memorization: [
      indicator("Combinações-base únicas", "privacy.unique_combinations", 19990, "evaluation.json → privacy", "Combinações distintas.", "informational"),
      indicator("Taxa de duplicidade", "privacy.duplicate_base_rows.duplicate_row_rate", 0, "evaluation.json → privacy", "Duplicidade nas colunas-base.", "informational"),
      indicator("Correspondência exata com treino", "privacy.exact_matches.train.exact_match_rate", 0, "evaluation.json → privacy", "Coincidência com treino.", "mandatory")
    ]
  },
  income: {
    status: "Disponível",
    indicators: [
      indicator("Versão do modelo de renda", "manifest.income_model_version", 3, "manifest.json", "Versão de renda sintética.", "informational"),
      indicator("Maior diferença de p99", "conditional_income.p99", null, "evaluation.json", "Esta execução não produziu essa métrica.", "informational")
    ]
  },
  executions: [
    {
      identifier: "run-1",
      created_at_utc: "2026-07-30T00:00:00Z",
      model: "ctgan",
      status: "approved",
      rows: 20000,
      duplicate_base_row_rate: 0,
      exact_train_match_rate: 0,
      unique_combinations: 20000
    },
    {
      identifier: "run-2",
      created_at_utc: "2026-07-29T00:00:00Z",
      model: "simple_gan",
      status: "quality_quarantined",
      rows: 20000,
      duplicate_base_row_rate: null,
      exact_train_match_rate: null,
      unique_combinations: null
    }
  ],
  audit: [{ event: "generation_succeeded", session_id: "abc", timestamp_utc: "2026-07-30T00:00:00Z" }],
  glossary: [
    { term: "Duplicidade de combinações-base", definition: "Repetição exata das 11 colunas-base." },
    { term: "Geo_Key", definition: "Chave categórica interna de geografia." }
  ]
};

describe("React frontend", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(Storage.prototype, "setItem");
    vi.stubGlobal("fetch", vi.fn(mockFetch));
  });

  it("renderiza somente as três páginas principais com Gerar Dados inicial", async () => {
    renderApp("/");

    expect(await screen.findByRole("heading", { name: /gerador de perfis sintéticos brasileiros/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /gerar dados/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /modelos/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /governança/i })).toBeInTheDocument();
    expect(screen.getByText(/dados sintéticos brasileiro/i)).toBeInTheDocument();
    expect(screen.queryByText(/visão geral/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/conformidade regulatória/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/vocabulário v2/i)).not.toBeInTheDocument();
  });

  it("pré-seleciona o artefato recomendado da CTGAN e preserva outros válidos em Gerar Dados", async () => {
    const user = userEvent.setup();
    renderApp("/");

    await user.click(await screen.findByRole("button", { name: /ctgan/i }));
    const select = await screen.findByLabelText(/artefato do modelo/i);

    await waitFor(() => expect(select).toHaveValue("ctgan/approved"));
    expect(screen.getByText(/ctgan\/smoke/i)).toBeInTheDocument();
  });

  it("envia geração com sessão efêmera e mostra preview", async () => {
    const user = userEvent.setup();
    renderApp("/");

    await screen.findByRole("button", { name: /gerar dados sintéticos/i });
    await user.click(screen.getByRole("button", { name: /gerar dados sintéticos/i }));

    expect(await screen.findByText(/resultado da geração/i)).toBeInTheDocument();
    expect(await screen.findByText("Ana")).toBeInTheDocument();
    const postCall = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls.find(([url, init]) => {
      return String(url).includes("/api/generations") && init?.method === "POST";
    });
    const headers = postCall?.[1].headers as Headers | Record<string, string> | undefined;
    const sessionHeader = headers instanceof Headers ? headers.get("X-UI-Session-ID") : headers?.["X-UI-Session-ID"];
    expect(sessionHeader).toMatch(/[0-9a-f-]{36}/i);
    expect(Storage.prototype.setItem).not.toHaveBeenCalled();
  });

  it("renderiza a página Modelos com badges, resumos e artefatos compactos", async () => {
    const user = userEvent.setup();
    renderApp("/modelos");

    expect(await screen.findByRole("heading", { name: "Modelos" })).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: /programático/i })).toHaveTextContent(/padrão geral/i);
    expect(screen.getByRole("button", { name: /ctgan/i })).toHaveTextContent(/recomendado/i);
    expect(screen.getByRole("button", { name: /gan simples/i })).toHaveTextContent(/experimental/i);

    await user.click(screen.getByRole("button", { name: /ctgan/i }));
    await waitFor(() => expect(screen.getAllByText(/Resumo simples/i).length).toBeGreaterThan(0));
    expect(screen.getAllByText(/Resumo técnico/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Artefato selecionado/i)).toBeInTheDocument();
    expect(screen.getAllByText(/ctgan\/approved/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Outros artefatos disponíveis/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Não avaliado/i).length).toBeGreaterThan(0);
  });

  it("renderiza Governança com seções, glossário, filtros e dados ausentes explícitos", async () => {
    const user = userEvent.setup();
    renderApp("/governanca");

    expect(await screen.findByRole("heading", { name: "Governança" })).toBeInTheDocument();
    expect(await screen.findByText(/Resumo Operacional/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Modelo Neural Recomendado/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Qualidade dos Dados/i)).toBeInTheDocument();
    expect(screen.getByText(/Diversidade e Memorização/i)).toBeInTheDocument();
    expect(screen.getByText(/Realismo Condicional/i)).toBeInTheDocument();
    expect(screen.getByText(/Execuções Recentes/i)).toBeInTheDocument();
    expect(screen.getByText(/Auditoria/i)).toBeInTheDocument();
    expect(screen.getByText(/Glossário/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Não avaliado/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Fonte: validation\.json/i)).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText(/Modelo/i), "simple_gan");
    expect(screen.queryByText("run-1")).not.toBeInTheDocument();
    expect(within(screen.getByRole("table")).getByText(/simple_gan/i)).toBeInTheDocument();
  });

  it("exibe erro amigável quando a governança falha", async () => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/governance")) return Promise.resolve(new Response(JSON.stringify({ error: { code: "GOVERNANCE_UNAVAILABLE", message: "Falha." } }), { status: 503 }));
      return mockFetch(input, init);
    }));

    renderApp("/governanca");

    expect(await screen.findByText(/não foi possível carregar a governança/i)).toBeInTheDocument();
  });
});

function renderApp(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>
  );
}

function ok(payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" }
  });
}

function accepted(payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status: 202,
    headers: { "Content-Type": "application/json" }
  });
}

function mockFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const url = String(input);
  if (url.endsWith("/api/health")) return Promise.resolve(ok(health));
  if (url.endsWith("/api/columns")) return Promise.resolve(ok(columns));
  if (url.endsWith("/api/models")) return Promise.resolve(ok(models));
  if (url.endsWith("/api/models/ctgan")) return Promise.resolve(ok(models.models[1]));
  if (url.endsWith("/api/models/simple_gan")) return Promise.resolve(ok(models.models[2]));
  if (url.endsWith("/api/models/ctgan/recommended")) return Promise.resolve(ok({ model: "ctgan", artifact: ctganApproved, message: null }));
  if (url.endsWith("/api/models/ctgan/artifacts")) return Promise.resolve(ok(ctganArtifacts));
  if (url.endsWith("/api/models/simple_gan/artifacts")) return Promise.resolve(ok(simpleGanArtifacts));
  if (url.endsWith("/api/governance")) return Promise.resolve(ok(governance));
  if (url.endsWith("/api/generations") && init?.method === "POST") {
    return Promise.resolve(accepted({ generation_id: "gen-1", status: "queued", status_url: "/api/generations/gen-1" }));
  }
  if (url.endsWith("/api/generations/gen-1")) {
    return Promise.resolve(
      ok({
        generation_id: "gen-1",
        status: "completed",
        model: "programmatic",
        artifact_id: null,
        num_rows: 1000,
        output_format: "csv",
        seed: 41,
        submitted_at_utc: "2026-07-30T00:00:00Z",
        started_at_utc: "2026-07-30T00:00:00Z",
        completed_at_utc: "2026-07-30T00:00:01Z",
        duration_seconds: 1,
        exported_columns: ["Nome", "Idade"],
        internal_columns: columns.final_columns,
        validation: { is_valid: true },
        dataset_download_url: "/api/generations/gen-1/download/dataset",
        manifest_download_url: "/api/generations/gen-1/download/manifest",
        preview_url: "/api/generations/gen-1/preview",
        error: null
      })
    );
  }
  if (url.endsWith("/api/generations/gen-1/preview")) {
    return Promise.resolve(ok({ generation_id: "gen-1", columns: ["Nome", "Idade"], rows: [{ Nome: "Ana", Idade: 30 }], preview_rows: 20, total_rows: 1000 }));
  }
  if (url.includes("/download/")) {
    return Promise.resolve(new Response("payload", { status: 200 }));
  }
  return Promise.resolve(new Response("not found", { status: 404 }));
}

function column(name: string, group: string, generatedBy: string, sensitive: boolean) {
  return {
    name,
    label: name,
    description: `${name} sintético.`,
    group,
    kind: "string",
    generated_by: generatedBy,
    dependencies: [],
    sensitive_like: sensitive,
    default_selected: true
  };
}

function model(
  name: ModelName,
  label: string,
  artifactRequired: boolean,
  available: boolean,
  experimental = false,
  recommendedArtifact: ModelArtifact | null = null
) {
  return {
    id: name,
    name,
    title: label,
    label,
    category: label,
    status: experimental ? "Experimental" : name === "ctgan" ? "Artefato neural recomendado" : "Recomendado",
    recommended: name === "programmatic",
    experimental,
    requires_training: artifactRequired,
    requires_saved_artifact: artifactRequired,
    available,
    row_limit: 1000,
    short_description: `${label} curto.`,
    detailed_description: `${label} detalhado.`,
    summary: `${label} resumo.`,
    simple_summary: `${label} resumo simples.`,
    technical_summary: `${label} resumo técnico.`,
    recommended_for:
      name === "ctgan"
        ? "Indicado para: Experimentos tabulares, aprendizado de distribuições e relações complexas, pesquisa metodológica e geração com artefatos avaliados pela equipe responsável."
        : name === "simple_gan"
          ? "Indicado para: Estudos acadêmicos sobre GANs, comparação de arquiteturas, análise de colapso categórico e experimentos metodológicos controlados."
          : "Indicado para: Desenvolvimento, testes funcionais, demonstrações técnicas, validação de pipelines, ensino e grandes volumes locais.",
    recommended_use_cases: ["testes", "ensino", "pesquisa"],
    benefits: ["Benefício."],
    limitations: ["Limitação."],
    governance_notes: ["Nota de governança."],
    recommended_artifact: recommendedArtifact,
    artifact_count: artifactRequired ? 1 : 0,
    availability_message: available ? null : "Indisponível.",
    metadata: { realism_profile: "Resumo técnico." }
  };
}

function artifact(id: string, status: string, recommended: boolean, modelName: ModelName = "ctgan"): ModelArtifact {
  return {
    artifact_id: id,
    model: modelName,
    label: `${id} — ${status}`,
    created_at_utc: "2026-07-30T00:00:00Z",
    created_at: "2026-07-30T00:00:00Z",
    train_rows: 20000,
    seed: 47,
    status,
    purpose: status.toLowerCase(),
    approved: status === "Aprovado",
    recommended,
    recommended_for_neural_generation: recommended,
    general_platform_default: false,
    schema_version: 1,
    categorical_vocabulary_version: 2,
    vocabulary_version: 2,
    income_model_version: 3,
    geography_model_version: 2,
    geography_catalog_checksum: "checksum",
    training_required: true,
    model_size_bytes: 1024,
    epochs: null,
    library: null,
    quality_status: null,
    duplicate_base_row_rate: recommended ? 0 : null,
    exact_train_match_rate: recommended ? 0 : null,
    conditional_income_status: null,
    compatibility: "compatível",
    is_legacy_vocabulary: false,
    is_legacy_income_model: false,
    is_legacy_geography_model: false,
    compatibility_normalization_required: false,
    warning: null
  };
}

function indicator(label: string, metric: string, value: unknown, source: string, interpretation: string, gateType: string) {
  return {
    indicator: label,
    indicador: label,
    metric,
    value,
    unit: metric.endsWith("_rate") ? "taxa" : "valor",
    source,
    fonte: source,
    interpretation,
    interpretação: value === null ? "Esta execução foi produzida antes da inclusão desta métrica ou não contém os artefatos necessários." : interpretation,
    gate_type: gateType,
    risk: value === null ? "Não avaliado" : "Diagnóstico"
  };
}
