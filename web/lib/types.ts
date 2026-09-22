/**
 * Mirrors the Pydantic models in `api/models.py` **by hand**.
 *
 * There is no code generation here on purpose. The contract is small and
 * stable, and a generated client would hide the one thing that matters when
 * these drift: the build fails loudly on a field the API stopped returning.
 * If you change a model in `api/models.py`, change it here too.
 */

export type StageName =
  | "retrieval"
  | "rerank"
  | "extraction"
  | "layout"
  | "synthesis";

export type StageStatus = "pending" | "running" | "done" | "error";

export type LandscapeStatus = "pending" | "running" | "ready" | "failed";

export type EdgeKind =
  | "builds_on"
  | "improves"
  | "compares_against"
  | "contradicts"
  | "shares_method"
  | "applies";

export const STAGE_ORDER: StageName[] = [
  "retrieval",
  "rerank",
  "extraction",
  "layout",
  "synthesis",
];

export const STAGE_LABELS: Record<StageName, string> = {
  retrieval: "Retrieval",
  rerank: "Reranking",
  extraction: "Extraction",
  layout: "Layout",
  synthesis: "Synthesis",
};

export interface StageProgress {
  current: number;
  total: number;
}

export interface StageEvent {
  run_id: string;
  landscape_id: number | null;
  stage: StageName;
  status: StageStatus;
  message: string;
  progress: StageProgress;
  payload: Record<string, unknown>;
  ts: string;
}

export interface StreamError {
  run_id: string;
  landscape_id: number | null;
  stage: StageName;
  message: string;
  /** True when retrying the same topic is likely to work (e.g. arXiv throttled us). */
  retryable: boolean;
}

export interface StreamDone {
  run_id: string;
  landscape_id: number;
  url: string;
  papers: number;
  clusters: number;
}

export interface PaperExtraction {
  problem: string | null;
  method: string | null;
  results: string | null;
  contribution: string | null;
  limitations: string | null;
  datasets: string[];
  metrics: string[];
  novelty: "incremental" | "substantial" | "unclear";
  evidence: Record<string, string>;
  confidence: number | null;
}

export interface ClusterOut {
  id: number;
  label: string;
  description: string;
  paper_count: number;
  x: number;
  y: number;
  color: string;
  is_unclustered: boolean;
}

export interface PaperInLandscape {
  paper_id: string;
  title: string;
  abstract: string;
  authors: string[];
  published: string;
  primary_category: string;
  categories: string[];
  abs_url: string;
  pdf_url: string;
  rank: number;
  /** Absolute, calibrated 0-10. Saturates near 10 for a pre-filtered corpus. */
  relevance_score: number;
  /** Raw cross-encoder output. This is what resolves ordering. */
  cross_encoder_logit: number | null;
  /** Percentile rank within this landscape, 0-10. Relative, display only. */
  relative_score: number | null;
  rerank_source: string;
  cluster_id: number | null;
  x: number;
  y: number;
  is_seed: boolean;
  extraction: PaperExtraction | null;
}

export interface EdgeOut {
  src_paper_id: string;
  dst_paper_id: string;
  kind: EdgeKind;
  weight: number;
  rationale: string;
}

export interface TensionOut {
  statement: string;
  paper_a_id: string;
  paper_b_id: string;
}

export interface OpenProblemOut {
  statement: string;
  why_open: string;
  supporting_paper_ids: string[];
}

export interface ReadingStepOut {
  paper_id: string;
  position: number;
  why: string;
  title: string;
}

export interface LandscapeSummary {
  id: number;
  topic: string;
  title: string;
  summary: string;
  status: LandscapeStatus;
  generation: number;
  paper_count: number;
  cluster_count: number;
  created_at: string;
  updated_at: string;
}

export interface LandscapeDetail {
  id: number;
  topic: string;
  title: string;
  summary: string;
  status: LandscapeStatus;
  generation: number;
  created_at: string;
  updated_at: string;
  clusters: ClusterOut[];
  papers: PaperInLandscape[];
  edges: EdgeOut[];
  tensions: TensionOut[];
  open_problems: OpenProblemOut[];
  reading_path: ReadingStepOut[];
}

export interface HealthOut {
  status: "ok";
  llm_configured: boolean;
  llm_provider: string;
  llm_model: string;
  prompt_version: string;
}

export const EDGE_LABELS: Record<EdgeKind, string> = {
  builds_on: "builds on",
  improves: "improves",
  compares_against: "compares against",
  contradicts: "contradicts",
  shares_method: "shares method",
  applies: "applies",
};
