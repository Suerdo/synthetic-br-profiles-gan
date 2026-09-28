import { apiDownload, apiGet, apiPost } from "./client";
import type {
  ColumnsResponse,
  GenerationCreatePayload,
  GenerationCreateResponse,
  GenerationManifestResponse,
  GenerationPreviewResponse,
  GenerationStatusResponse,
  GovernanceSnapshot,
  HealthResponse,
  ModelArtifactsResponse,
  ModelEntry,
  ModelName,
  ModelsResponse,
  RecommendedArtifactResponse
} from "../types/api";

export function getHealth(): Promise<HealthResponse> {
  return apiGet<HealthResponse>("/api/health");
}

export function getColumns(): Promise<ColumnsResponse> {
  return apiGet<ColumnsResponse>("/api/columns");
}

export function getModels(): Promise<ModelsResponse> {
  return apiGet<ModelsResponse>("/api/models");
}

export function getModel(model: ModelName): Promise<ModelEntry> {
  return apiGet<ModelEntry>(`/api/models/${model}`);
}

export function getModelArtifacts(model: ModelName): Promise<ModelArtifactsResponse> {
  return apiGet<ModelArtifactsResponse>(`/api/models/${model}/artifacts`);
}

export function getRecommendedModelArtifact(model: ModelName): Promise<RecommendedArtifactResponse> {
  return apiGet<RecommendedArtifactResponse>(`/api/models/${model}/recommended`);
}

export function getGovernance(): Promise<GovernanceSnapshot> {
  return apiGet<GovernanceSnapshot>("/api/governance");
}

export function getGovernanceSummary(): Promise<GovernanceSnapshot["operational"]> {
  return apiGet<GovernanceSnapshot["operational"]>("/api/governance/summary");
}

export function getGovernanceQuality(): Promise<GovernanceSnapshot["quality"]> {
  return apiGet<GovernanceSnapshot["quality"]>("/api/governance/quality");
}

export function getGovernancePrivacy(): Promise<GovernanceSnapshot["privacy"]> {
  return apiGet<GovernanceSnapshot["privacy"]>("/api/governance/privacy");
}

export function getGovernanceIncome(): Promise<GovernanceSnapshot["income"]> {
  return apiGet<GovernanceSnapshot["income"]>("/api/governance/income");
}

export function getGovernanceExecutions(): Promise<GovernanceSnapshot["executions"]> {
  return apiGet<GovernanceSnapshot["executions"]>("/api/governance/executions");
}

export function getGovernanceAudit(): Promise<GovernanceSnapshot["audit"]> {
  return apiGet<GovernanceSnapshot["audit"]>("/api/governance/audit");
}

export function createGeneration(payload: GenerationCreatePayload, sessionId: string): Promise<GenerationCreateResponse> {
  return apiPost<GenerationCreateResponse>("/api/generations", payload, sessionId);
}

export function getGeneration(generationId: string, sessionId: string): Promise<GenerationStatusResponse> {
  return apiGet<GenerationStatusResponse>(`/api/generations/${generationId}`, sessionId);
}

export function getGenerationPreview(generationId: string, sessionId: string): Promise<GenerationPreviewResponse> {
  return apiGet<GenerationPreviewResponse>(`/api/generations/${generationId}/preview`, sessionId);
}

export function getGenerationManifest(generationId: string, sessionId: string): Promise<GenerationManifestResponse> {
  return apiGet<GenerationManifestResponse>(`/api/generations/${generationId}/manifest`, sessionId);
}

export function downloadDataset(generationId: string, sessionId: string): Promise<Blob> {
  return apiDownload(`/api/generations/${generationId}/download/dataset`, sessionId);
}

export function downloadManifest(generationId: string, sessionId: string): Promise<Blob> {
  return apiDownload(`/api/generations/${generationId}/download/manifest`, sessionId);
}
