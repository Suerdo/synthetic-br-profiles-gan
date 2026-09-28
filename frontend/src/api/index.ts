import { apiDownload, apiGet, apiPost } from "./client";
import type {
  ColumnsResponse,
  GenerationCreatePayload,
  GenerationCreateResponse,
  GenerationManifestResponse,
  GenerationPreviewResponse,
  GenerationStatusResponse,
  HealthResponse,
  ModelArtifactsResponse,
  ModelName,
  ModelsResponse
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

export function getModelArtifacts(model: ModelName): Promise<ModelArtifactsResponse> {
  return apiGet<ModelArtifactsResponse>(`/api/models/${model}/artifacts`);
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
