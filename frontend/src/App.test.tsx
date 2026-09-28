import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import type { ModelArtifact, ModelName } from "./types/api";
import indexHtml from "../index.html?raw";

const longChecksum = "0b12f8466842767c637a37cbff3939d730c1a06c87770c0846cfdeebd8ccf033";

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
  governance_decision: {
    title: "Decisão de Governança",
    status_label: "Aprovado nos gates internos",
    evaluation_status: "approved",
    recommendation_status: "recommended",
    general_default: false,
    production_status: "not_approved",
    scope: "Artefato neural CTGAN com vocabulário v2, renda v3 e geografia v2.",
    decided_at_utc: "2026-07-30T12:32:08Z",
    benchmark: "ctgan-confirmation",
    mandatory_passed: 3,
    mandatory_total: 3,
    caveats_count: 1,
    artifact_id: "ctgan/approved",
    model: "ctgan",
    disclaimer:
      "A aprovação representa uma decisão técnica interna baseada nos critérios do projeto. Ela não constitui certificação externa, garantia de anonimização ou validação populacional oficial."
  },
  available_strategies: [
    {
      model: "programmatic",
      label: "Programático",
      role: "Padrão geral da plataforma",
      status: "Disponível",
      operational_availability: "Disponível",
      artifact_required: false,
      artifact_available: true,
      artifact_count: 0,
      recommended_artifact_id: null
    },
    {
      model: "ctgan",
      label: "CTGAN",
      role: "Modelo neural recomendado",
      status: "Artefato aprovado disponível",
      operational_availability: "Disponível",
      artifact_required: true,
      artifact_available: true,
      artifact_count: 2,
      recommended_artifact_id: "ctgan/approved"
    },
    {
      model: "simple_gan",
      label: "GAN simples",
      role: "Baseline acadêmico experimental",
      status: "Artefato válido disponível",
      operational_availability: "Disponível",
      artifact_required: true,
      artifact_available: true,
      artifact_count: 1,
      recommended_artifact_id: null
    }
  ],
  provenance: {
    items: [
      { label: "Artefato", value: "ctgan/approved", source: "training_manifest.json" },
      { label: "Benchmark", value: "ctgan-confirmation", source: "approval_manifest.json" },
      { label: "Seeds", value: [47, 48, 49], source: "approval_manifest.json" },
      { label: "Commit atual da aplicação", value: "abc123", source: ".git/HEAD" },
      { label: "Treino", value: 20000, source: "training_manifest.json" },
      { label: "Holdout", value: 5000, source: "training_manifest.json" },
      { label: "Calibração", value: 25000, source: "training_manifest.json" },
      { label: "Épocas", value: 20, source: "ctgan_config" },
      { label: "Batch size", value: 500, source: "ctgan_config" },
      { label: "Python", value: "Python 3.13", source: "training_manifest.json" },
      { label: "Checksum geográfico", value: longChecksum, source: "training_manifest.json" }
    ],
    groups: [
      { title: "Identificação", keys: ["Artefato", "Benchmark", "Seeds", "Commit atual da aplicação"] },
      { title: "Dados e split", keys: ["Treino", "Holdout", "Calibração"] },
      { title: "Hiperparâmetros", keys: ["Épocas", "Batch size"] },
      { title: "Ambiente e versões", keys: ["Python", "Checksum geográfico"] }
    ],
    timeline: [
      { step: "Configuração", source: "ctgan_config", value: "ctgan_income_v3_geo_v2_candidate" },
      { step: "Decisão", source: "approval_manifest.json", value: "approved" }
    ],
    current_code_commit: "abc123"
  },
  quality_gates: [
    { id: "invalid_rows_max", metric: "Erros estruturais", observed: 0, operator: "=", threshold: 0, mandatory: true, passed: true, status: "Aprovado", source: "validation.json", evidence: 0 },
    { id: "exact_train_match_rate_max", metric: "Match exato treino", observed: 0, operator: "<=", threshold: 0.01, mandatory: true, passed: true, status: "Aprovado", source: "quality_gates.json", evidence: 0 },
    { id: "tvd", metric: "TVD", observed: null, operator: null, threshold: null, mandatory: false, passed: null, status: "Não avaliado", source: "evaluation.json", evidence: null }
  ],
  evidence_by_model: {
    default_model: "ctgan",
    models: [
      evidence("programmatic", "Programático", "Padrão geral", "Padrão geral", "run-programmatic", "approved", 5e-5, 0.059),
      evidence("ctgan", "CTGAN", "Recomendado", "Modelo neural recomendado", "ctgan/approved", "approved", 0, null),
      evidence("simple_gan", "GAN simples", "Experimental · quarantined", "Experimental", "run-simple", "quarantined", null, 0.984)
    ]
  },
  operational: {
    metrics: [
      { label: "Estratégias disponíveis", key: "strategy_count", value: 3, source: "ModelRegistry", help: "Estratégias do projeto." },
      { label: "Modelo padrão geral", key: "default_model", value: "programmatic", source: "configs/api.yaml", help: "Modelo inicial da interface." },
      { label: "Modelo neural recomendado", key: "recommended_neural_model", value: "ctgan/approved", source: "ModelRegistry", help: "Artefato neural recomendado." },
      { label: "Execuções registradas", key: "registered_executions", value: 3, source: "manifestos", help: "Execuções locais." },
      { label: "Execuções com avaliação completa", key: "evaluated_executions", value: 2, source: "evaluation.json", help: "Execuções avaliadas." }
    ],
    summary: { registered_executions: 3, evaluated_executions: 2, latest_evaluation: "run-1", strategy_count: 3 }
  },
  recommended_model: {
    artifact_id: "ctgan/approved",
    status: "approved",
    vocabulary_version: 2,
    income_model_version: 3,
    geography_model_version: 2,
    geography_catalog_checksum: longChecksum,
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
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText: vi.fn().mockResolvedValue(undefined) },
      configurable: true
    });
    vi.stubGlobal("fetch", vi.fn(mockFetch));
  });

  it("renderiza somente as três páginas principais com Gerar Dados inicial", async () => {
    renderApp("/");

    expect(await screen.findByRole("heading", { name: /gerar dados sint/i })).toBeInTheDocument();
    expect(screen.getAllByLabelText(/^dados sintéticos brasileiros$/i).length).toBeGreaterThan(0);
    expect(screen.getByRole("link", { name: /gerar dados/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /modelos/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /governança/i })).toBeInTheDocument();
    expect(screen.getAllByText(/^dados sintéticos$/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/^brasileiros$/i).length).toBeGreaterThan(0);
    expect(screen.queryByText(/^Dados Sintéticos Brasileiro$/)).not.toBeInTheDocument();
    expect(screen.queryByText(/visão geral/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/conformidade regulatória/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/vocabulário v2/i)).not.toBeInTheDocument();
  });

  it("pré-seleciona o artefato recomendado da CTGAN e preserva outros válidos em Gerar Dados", async () => {
    const user = userEvent.setup();
    renderApp("/");

    await user.click(await screen.findByRole("button", { name: /ctgan/i }));
    const select = await screen.findByLabelText(/alterar artefato/i);

    await waitFor(() => expect(select).toHaveValue("ctgan/approved"));
    expect(screen.getByText(/ctgan\/smoke/i)).toBeInTheDocument();
  });

  it("abre e fecha o drawer de navegação mobile", async () => {
    const user = userEvent.setup();
    renderApp("/");

    await user.click(await screen.findByRole("button", { name: /abrir navegação/i }));
    const drawer = screen.getByRole("dialog", { name: /navegação principal/i });
    expect(within(drawer).getByRole("link", { name: /gerar dados/i })).toBeInTheDocument();
    expect(within(drawer).getByRole("link", { name: /modelos/i })).toBeInTheDocument();
    expect(within(drawer).getByRole("link", { name: /governança/i })).toBeInTheDocument();
    expect(within(drawer).getByLabelText(/^dados sintéticos brasileiros$/i)).toBeInTheDocument();

    await user.click(within(drawer).getByRole("link", { name: /modelos/i }));
    await waitFor(() => expect(screen.queryByRole("dialog", { name: /navegação principal/i })).not.toBeInTheDocument());
  });

  it("envia geração com sessão efêmera e mostra preview", async () => {
    const user = userEvent.setup();
    renderApp("/");

    await screen.findByRole("button", { name: /gerar dados sintéticos/i });
    await user.click(screen.getByRole("button", { name: /gerar dados sintéticos/i }));

    expect(await screen.findByText(/dados gerados com sucesso/i)).toBeInTheDocument();
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
    await waitFor(() => expect(screen.getAllByText(/Resumo Simples/i).length).toBeGreaterThan(0));
    expect(screen.getAllByText(/Resumo Técnico/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Artefato Selecionado/i)).toBeInTheDocument();
    expect(screen.getAllByLabelText(/copiar valor completo ctgan\/approved/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Outros Artefatos Disponíveis/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Não avaliado/i).length).toBeGreaterThan(0);
  });

  it("renderiza Governança com decisão, proveniência, gates e seções recolhidas", async () => {
    const user = userEvent.setup();
    renderApp("/governanca");

    expect(await screen.findByRole("heading", { name: /^Governan/i })).toBeInTheDocument();
    expect((await screen.findAllByText(/Modelo neural recomendado/i)).length).toBeGreaterThan(0);
    expect(screen.queryByText(/Aprovado nos gates internos/i)).not.toBeInTheDocument();
    expect(screen.getByText(/N.*o aprovado \/ n.*o definido/i)).toBeInTheDocument();
    expect(screen.getByText(/Estrat.*gias \/ Modelos Dispon/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Program/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/GAN simples/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Modelo Neural Recomendado/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Proveni.*ncia e Reprodutibilidade/i)).toBeInTheDocument();
    expect(screen.getByText(/Evid.*ncias do Modelo/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Quality Gates/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Match exato com treino/i)).toBeInTheDocument();
    const evidenceSelect = screen.getByLabelText(/Modelo das evidências/i);
    expect(evidenceSelect).toHaveValue("ctgan");
    expect(within(evidenceSelect).getByRole("option", { name: /Programático/i })).toBeInTheDocument();
    expect(within(evidenceSelect).getByRole("option", { name: /CTGAN/i })).toBeInTheDocument();
    expect(within(evidenceSelect).getByRole("option", { name: /GAN Simples/i })).toBeInTheDocument();
    expect(within(evidenceSelect).queryByRole("option", { name: /^programmatic$/i })).not.toBeInTheDocument();
    expect(within(evidenceSelect).queryByRole("option", { name: /^simple_gan$/i })).not.toBeInTheDocument();
    expect(screen.getAllByText(/ctgan\/approved/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Aprovado/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Privacidade, Diversidade e Memoriza/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Realismo e Fidelidade Estat/i).length).toBeGreaterThan(0);
    expect(screen.getByRole("heading", { name: /Execu.*es Recentes/i })).toBeInTheDocument();
    expect(screen.getByText(/Ver trilha de auditoria/i)).toBeInTheDocument();
    expect(screen.getByText(/Abrir gloss.*rio completo/i)).toBeInTheDocument();
    expect(screen.getAllByText(/N.*o avaliado/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Fonte: quality_gates\.json/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/0b12f846684.*d8ccf033/i).length).toBeGreaterThan(0);
    expect(screen.queryByText(new RegExp(`${longChecksum}\\(training_manifest\\.json\\)`, "i"))).not.toBeInTheDocument();
    const copyChecksum = screen.getAllByLabelText(new RegExp(`copiar valor completo ${longChecksum}`, "i"))[0];
    const clipboardWrite = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText: clipboardWrite },
      configurable: true
    });
    await user.click(copyChecksum);
    expect(clipboardWrite).toHaveBeenCalledWith(longChecksum);

    await user.selectOptions(evidenceSelect, "programmatic");
    expect(screen.getByText(/run-programmatic/i)).toBeInTheDocument();
    expect(screen.getAllByText(/0,059/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Modelo Neural Recomendado/i).length).toBeGreaterThan(0);

    await user.selectOptions(evidenceSelect, "simple_gan");
    expect(screen.getByText(/run-simple/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Experimental/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Quarentena/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Não avaliado/i).length).toBeGreaterThan(0);
    expect(screen.queryByLabelText(/^Modelo$/i)).not.toBeInTheDocument();
    expect((screen.getByLabelText(/^Status$/i) as HTMLSelectElement).value).toBe("todos");
  });

  it("configura favicon e título HTML da aplicação", () => {
    expect(indexHtml).toContain('<link rel="icon" type="image/png" href="/brand/dados-sinteticos-br-logo.png" />');
    expect(indexHtml).toContain("<title>Dados Sintéticos Brasileiros</title>");
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
    geography_catalog_checksum: longChecksum,
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

function evidence(
  modelName: ModelName,
  label: string,
  statusLabel: string,
  role: string,
  sourceIdentifier: string,
  latestStatus: string,
  exactTrainRate: number | null,
  tvd: number | null
) {
  return {
    model: modelName,
    label,
    status_label: statusLabel,
    role,
    source_kind: "pipeline_run",
    source_identifier: sourceIdentifier,
    latest_execution_status: latestStatus,
    artifact_id: modelName === "ctgan" ? sourceIdentifier : null,
    has_evidence: true,
    quality_gates: [
      {
        id: `${modelName}-invalid_rows`,
        metric: "Erros estruturais",
        observed: 0,
        operator: "=",
        threshold: 0,
        mandatory: true,
        passed: true,
        status: "Aprovado",
        source: "quality_gates.json",
        evidence: 0
      },
      {
        id: `${modelName}-tvd`,
        metric: "TVD",
        observed: tvd,
        operator: tvd === null ? null : "<=",
        threshold: tvd === null ? null : 0.25,
        mandatory: false,
        passed: tvd === null ? null : tvd <= 0.25,
        status: tvd === null ? "Não avaliado" : tvd <= 0.25 ? "Aprovado" : "Quarentena",
        source: "quality_gates.json",
        evidence: tvd
      }
    ],
    privacy: {
      status: "Disponível",
      diversity_memorization: [
        indicator("Combinações-base únicas", "privacy.unique_combinations", 19990, `evaluation.json → ${modelName}`, "Combinações distintas.", "informational"),
        indicator("Taxa de duplicidade", "privacy.duplicate_base_rows.duplicate_row_rate", 0, `evaluation.json → ${modelName}`, "Duplicidade nas colunas-base.", "informational"),
        indicator("Correspondência exata com treino", "privacy.exact_matches.train.exact_match_rate", exactTrainRate, `evaluation.json → ${modelName}`, "Coincidência com treino.", "mandatory")
      ]
    },
    income: {
      status: "Disponível",
      indicators: [
        indicator("Versão do modelo de renda", "manifest.income_model_version", 3, `manifest.json → ${modelName}`, "Versão de renda sintética.", "informational"),
        indicator("Maior diferença de p99", "conditional_income.p99", modelName === "simple_gan" ? null : 120.5, `evaluation.json → ${modelName}`, "Diferença de cauda.", "informational")
      ]
    }
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
