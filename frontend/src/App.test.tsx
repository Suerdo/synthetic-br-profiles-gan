import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";

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
    {
      name: "Nome",
      label: "Nome",
      description: "Nome sintético.",
      group: "Identificação sintética",
      kind: "string",
      generated_by: "postprocessing",
      dependencies: [],
      sensitive_like: true,
      default_selected: true
    },
    {
      name: "Idade",
      label: "Idade",
      description: "Idade sintética.",
      group: "Demografia",
      kind: "integer",
      generated_by: "model",
      dependencies: [],
      sensitive_like: false,
      default_selected: true
    },
    {
      name: "Estado",
      label: "Estado",
      description: "Estado sintético.",
      group: "Localização e contato",
      kind: "categorical",
      generated_by: "model",
      dependencies: [],
      sensitive_like: false,
      default_selected: true
    },
    {
      name: "CPF",
      label: "CPF",
      description: "CPF sintético.",
      group: "Identificação sintética",
      kind: "identifier",
      generated_by: "postprocessing",
      dependencies: [],
      sensitive_like: true,
      default_selected: true
    }
  ],
  presets: [
    { name: "completo", columns: ["Nome", "Idade", "Estado", "CPF"] },
    { name: "minimo", columns: ["Nome", "Idade", "Estado", "CPF"] }
  ],
  default_preset: "completo"
};

const models = {
  default_model: "programmatic",
  models: [
    model("programmatic", "Programático — Recomendado", false, true),
    model("ctgan", "CTGAN — Artefato Neural Recomendado", true, true),
    model("simple_gan", "GAN simples — Experimental", true, true, true)
  ]
};

const ctganArtifacts = {
  model: "ctgan",
  recommended_artifact_id: "ctgan/approved",
  artifacts: [
    artifact("ctgan/approved", "Aprovado", true),
    artifact("ctgan/smoke", "Smoke", false)
  ]
};

const simpleGanArtifacts = {
  model: "simple_gan",
  recommended_artifact_id: null,
  artifacts: [artifact("simple/smoke", "Smoke", false, "simple_gan")]
};

describe("React frontend", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(Storage.prototype, "setItem");
    vi.stubGlobal("fetch", vi.fn(mockFetch));
  });

  it("renderiza somente as três páginas principais com Gerar dados inicial", async () => {
    renderApp("/");

    expect(await screen.findByRole("heading", { name: /gerador de perfis sintéticos brasileiros/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /gerar dados/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /modelos/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /governança/i })).toBeInTheDocument();
    expect(screen.queryByText(/visão geral/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/conformidade regulatória/i)).not.toBeInTheDocument();
  });

  it("pré-seleciona o artefato recomendado da CTGAN e preserva outros válidos", async () => {
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

  it("exibe os shells de Modelos e Governança", async () => {
    renderApp("/modelos");
    expect(await screen.findByRole("heading", { name: /conheça os sintetizadores/i })).toBeInTheDocument();

    renderApp("/governanca");
    expect(await screen.findByRole("heading", { name: /rastreabilidade e uso responsável/i })).toBeInTheDocument();
    expect(await screen.findByText(/como interpretar os indicadores/i)).toBeInTheDocument();
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
  if (url.endsWith("/api/models/ctgan/artifacts")) return Promise.resolve(ok(ctganArtifacts));
  if (url.endsWith("/api/models/simple_gan/artifacts")) return Promise.resolve(ok(simpleGanArtifacts));
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

function model(name: string, label: string, artifact: boolean, available: boolean, experimental = false) {
  return {
    name,
    label,
    category: label,
    status: experimental ? "Experimental" : "Recomendado",
    recommended: name === "programmatic",
    experimental,
    requires_training: artifact,
    requires_saved_artifact: artifact,
    available,
    row_limit: 1000,
    short_description: `${label} curto.`,
    detailed_description: `${label} detalhado.`,
    recommended_use_cases: ["testes"],
    benefits: ["Benefício."],
    limitations: ["Limitação."],
    recommended_artifact: null,
    artifact_count: artifact ? 1 : 0,
    availability_message: available ? null : "Indisponível.",
    metadata: { realism_profile: "Resumo técnico." }
  };
}

function artifact(id: string, status: string, recommended: boolean, modelName = "ctgan") {
  return {
    artifact_id: id,
    model: modelName,
    label: `${id} — ${status}`,
    created_at_utc: "2026-07-30T00:00:00Z",
    train_rows: 20000,
    seed: 47,
    status,
    purpose: status.toLowerCase(),
    recommended_for_neural_generation: recommended,
    general_platform_default: false,
    schema_version: 1,
    categorical_vocabulary_version: 2,
    income_model_version: 3,
    geography_model_version: 2,
    geography_catalog_checksum: "checksum",
    training_required: true,
    model_size_bytes: 1024,
    is_legacy_vocabulary: false,
    is_legacy_income_model: false,
    is_legacy_geography_model: false,
    compatibility_normalization_required: false,
    warning: null
  };
}
