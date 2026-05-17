/** Shared TypeScript types mirroring backend Pydantic schemas. */

export interface Paper {
  id: string;
  external_id: string;
  namespace_key: string;
  title: string;
  authors: string[];
  abstract: string;
  source_url: string;
  pdf_url: string | null;
  published_at: string | null;
  key_concepts: string[];
  methods_used: string[];
  implications: string | null;
  novelty_score: number;
  relevance_score: number;
  is_breakthrough: boolean;
  tldr: string | null;
  ingested_at: string;
  is_manually_imported?: boolean;
}

export interface FeedItem {
  paper: Paper;
  score: number;
  why_tag: string;
}

export interface FeedResponse {
  papers: FeedItem[];
  total: number;
  namespace_key: string;
}

export interface GraphNode {
  id: string;
  type: "TOPIC" | "SUBTOPIC" | "CONCEPT" | "METHOD" | "PAPER";
  label: string;
  namespace_key: string | null;
  paper_id: string | null;
  description?: string;
  source_url?: string | null;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  weight: number;
  cross_namespace: boolean;
}

export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface ChatResponse {
  answer: string;
  citation_paper_ids: string[];
  highlight_node_ids: string[];
  scope_level: string;
}

export interface SourcePaperInfo {
  id: string;
  title: string;
  authors: string[];
  year: number | null;
  url: string;
}

export interface IdeaCapsule {
  id: string;
  title: string;
  hypothesis: string;
  rationale: string;
  mechanism: string | null;
  predicted_outcome: string | null;
  experimental_design: string | null;
  anti_finding: string | null;
  risks_and_limitations: string | null;
  open_questions: string | null;
  novelty_score: number;
  feasibility_score: number;
  impact_score: number;
  diagrams: DiagramSpec[];
  poc_code: string | null;
  seed_element_ids: string[];
  status: "draft" | "saved" | "dismissed";
  is_scout_generated: boolean;
  source_mode: "manual" | "auto" | "query" | "combined";
  source_query: string | null;
  deep_dive_content?: string | null;
  deep_dive_status?: string;
  created_at: string;
  source_papers?: SourcePaperInfo[];
}

export interface DiagramSpec {
  type: "mermaid" | "mermaid_algo" | "image" | "hero_image";
  spec?: string;       // Mermaid syntax
  blob_path?: string;  // Image blob
}

export interface GenieElement {
  id: string;
  label: string;
  type: "concept" | "method" | "paper" | "idea";
  paper_id?: string | null;
  tldr?: string | null;
}

export interface User {
  id: string;
  email: string;
  display_name: string;
  expertise_level: "newcomer" | "practitioner" | "expert";
  orientation: "research" | "production" | "both";
  onboarding_complete: boolean;
}

export interface BookmarkFolder {
  id: string;
  name: string;
  color: string | null;
  created_at: string;
  bookmark_count: number;
}

export interface Bookmark {
  id: string;
  paper_id: string;
  folder_ids: string[];
  note: string | null;
  created_at: string;
  paper: Paper | null;
}

export interface StudySection {
  type: "section" | "diagram" | "related" | "start" | "done" | "error";
  label?: string;
  content?: string;
  paper_ids?: string[];
  spec?: string;
  blob_path?: string;
  caption?: string;
  diagram_kind?: string;
}

export interface SearchResultItem {
  paper_id: string;
  title: string;
  abstract: string;
  authors: string[];
  namespace_key: string;
  source_url: string;
  pdf_url: string | null;
  novelty_score: number;
  relevance_score: number;
  is_breakthrough: boolean;
  key_concepts: string[] | null;
  methods_used: string[] | null;
  implications: string | null;
  published_at: string | null;
  ingested_at: string | null;
  tldr: string | null;
  search_score: number;
  match_type: "keyword" | "semantic" | "hybrid";
}

export interface SearchResponse {
  results: SearchResultItem[];
  total: number;
  query: string;
  mode: string;
}

// ── Media generation ──────────────────────────────────────────────────────────

export type GenerationType = "podcast" | "slides";
export type SourceType = "paper" | "capsule" | "folder";
export type GenerationSourceType = "paper" | "capsule";   // folders excluded from media generation
export type ArtifactStatus = "queued" | "running" | "completed" | "failed";

export interface GeneratedArtifact {
  id: string;
  generation_type: GenerationType;
  source_type: GenerationSourceType;
  source_id: string;
  source_title: string;
  status: ArtifactStatus;
  blob_path: string | null;
  content: Record<string, unknown> | null;
  expertise_level: string | null;
  orientation: string | null;
  provider: string | null;
  model_used: string | null;
  input_tokens: number;
  output_tokens: number;
  generation_duration_ms: number;
  error_message: string | null;
  created_at: string;
  completed_at: string | null;
}

export interface TriggerResponse {
  artifact_id: string;
  job_id: string;
  status: ArtifactStatus | "cached";
  message: string;
  source_title: string;
}
