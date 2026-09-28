export type ModelName = "programmatic" | "ctgan" | "simple_gan";
export type OutputFormat = "csv" | "json" | "parquet";
export type GenerationStatus = "queued" | "running" | "completed" | "failed";

export interface HealthResponse {
  status: string;
  api_version: string;
  defaults: {
    model: ModelName;
    rows: number;
    format: OutputFormat;
    seed: number;
    preset: string;
  };
  models: Record<ModelName, { available: boolean; requires_artifact: boolean }>;
}

export interface ColumnEntry {
  name: string;
  label: string;
  description: string;
  group: string;
  kind: string;
  generated_by: string;
  dependencies: string[];
  sensitive_like: boolean;
  default_selected: boolean;
}

export interface ColumnPreset {
  name: string;
  columns: string[];
}

export interface ColumnsResponse {
  final_columns: string[];
  groups: string[];
  columns: ColumnEntry[];
  presets: ColumnPreset[];
  default_preset: string;
}

export interface ModelArtifact {
  artifact_id: string;
  model: ModelName;
  label: string;
  created_at_utc: string | null;
  created_at: string | null;
  train_rows: number | null;
  seed: number | null;
  status: string;
  purpose: string;
  approved: boolean;
  recommended: boolean;
  recommended_for_neural_generation: boolean;
  general_platform_default: boolean;
  schema_version: number;
  categorical_vocabulary_version: number;
  vocabulary_version: number;
  income_model_version: number;
  geography_model_version: number;
  geography_catalog_checksum: string | null;
  training_required: boolean;
  model_size_bytes: number | null;
  epochs: number | null;
  library: string | null;
  quality_status: string | null;
  duplicate_base_row_rate: number | MetricRange | null;
  exact_train_match_rate: number | MetricRange | null;
  conditional_income_status: string | null;
  compatibility: string;
  is_legacy_vocabulary: boolean;
  is_legacy_income_model: boolean;
  is_legacy_geography_model: boolean;
  compatibility_normalization_required: boolean;
  warning: string | null;
}

export interface ModelEntry {
  id: ModelName;
  name: ModelName;
  title: string;
  label: string;
  category: string;
  status: string;
  recommended: boolean;
  experimental: boolean;
  requires_training: boolean;
  requires_saved_artifact: boolean;
  available: boolean;
  row_limit: number;
  short_description: string;
  detailed_description: string;
  summary: string;
  technical_summary: string;
  simple_summary: string;
  recommended_for: string;
  recommended_use_cases: string[];
  benefits: string[];
  limitations: string[];
  governance_notes: string[];
  recommended_artifact: ModelArtifact | null;
  artifact_count: number;
  availability_message: string | null;
  metadata: Record<string, unknown>;
}

export interface ModelsResponse {
  models: ModelEntry[];
  default_model: ModelName;
}

export interface ModelArtifactsResponse {
  model: ModelName;
  artifacts: ModelArtifact[];
  recommended_artifact_id: string | null;
}

export interface RecommendedArtifactResponse {
  model: ModelName;
  artifact: ModelArtifact | null;
  message: string | null;
}

export interface MetricRange {
  min: number;
  max: number;
}

export interface GovernanceIndicator {
  indicator?: string;
  indicador?: string;
  metric?: string;
  "métrica"?: string;
  value?: unknown;
  valor?: unknown;
  unit?: string;
  source?: string;
  fonte?: string;
  interpretation?: string;
  interpretação?: string;
  risk?: string;
  gate_type?: string;
  date?: string | null;
  [key: string]: unknown;
}

export interface GovernanceMetric {
  label: string;
  key: string;
  value: unknown;
  source: string;
  help: string;
}

export interface GovernanceSnapshot {
  operational: {
    metrics: GovernanceMetric[];
    summary: Record<string, unknown>;
  };
  recommended_model: Record<string, unknown> | null;
  quality: {
    indicators: GovernanceIndicator[];
    status: string;
  };
  privacy: {
    indicators: GovernanceIndicator[];
    diversity_memorization: GovernanceIndicator[];
    status: string;
  };
  income: {
    indicators: GovernanceIndicator[];
    status: string;
  };
  executions: Array<Record<string, unknown>>;
  audit: Array<Record<string, unknown>>;
  glossary: Array<{ term: string; definition: string }>;
}

export interface GenerationCreatePayload {
  model: ModelName;
  artifact_id?: string | null;
  num_rows: number;
  output_format: OutputFormat;
  seed: number;
  selected_columns?: string[] | null;
  column_preset?: string | null;
}

export interface GenerationCreateResponse {
  generation_id: string;
  status: GenerationStatus;
  status_url: string;
}

export interface GenerationStatusResponse {
  generation_id: string;
  status: GenerationStatus;
  model: ModelName;
  artifact_id: string | null;
  num_rows: number;
  output_format: OutputFormat;
  seed: number;
  submitted_at_utc: string;
  started_at_utc: string | null;
  completed_at_utc: string | null;
  duration_seconds: number | null;
  exported_columns: string[];
  internal_columns: string[];
  validation: Record<string, unknown> | null;
  dataset_download_url: string | null;
  manifest_download_url: string | null;
  preview_url: string | null;
  error: { type: string; message: string } | null;
}

export interface GenerationPreviewResponse {
  generation_id: string;
  rows: Array<Record<string, unknown>>;
  columns: string[];
  preview_rows: number;
  total_rows: number;
}

export interface GenerationManifestResponse {
  generation_id: string;
  manifest: Record<string, unknown>;
}
