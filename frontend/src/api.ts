/**
 * The typed client. Every call to the backend goes through here.
 *
 * Two things this file is responsible for. It mirrors the wire types from
 * `src/storygit/api/schemas.py` — when those change, this is the file that has to change,
 * and nothing else does. And it turns the backend's single problem shape into a typed
 * `ApiError`, so every caller has exactly one failure path and can tell a rate limit
 * (retry, and we know for how long) from a locked node (tell the writer) without parsing
 * English.
 */

export type NodeType = "story" | "episode" | "scene" | "beat" | "prose";
export type NodeStatus = "draft" | "accepted" | "locked" | "stale";
export type Authorship = "human" | "ai" | "ai_edited_by_human";

export interface NodeSummary {
  id: string;
  parent_id: string | null;
  node_type: NodeType;
  title: string;
  status: NodeStatus;
  stale_reason: string | null;
  locked: boolean;
  position: number;
  seq: number;
  has_prose: boolean;
  children: string[];
  flag_count: number;
}

export interface TreeResponse {
  branch: string;
  root_id: string | null;
  nodes: NodeSummary[];
  stale_count: number;
  review_count: number;
}

export interface Flag {
  kind: string;
  severity: "hard" | "soft";
  layer: number;
  message: string;
  node_id: string | null;
  fact_ids: string[];
  established_by: string | null;
  entity_ids: string[];
  subgraph: string[];
  score: number | null;
}

export interface ProseSpan {
  start: number;
  end: number;
  source: Authorship;
  proposal_id: string | null;
}

export interface NodeDetail {
  node: NodeSummary;
  what_happens: string;
  audience_learns: string;
  audience_feels: string;
  location: string;
  time: string;
  prose: string;
  spans: ProseSpan[];
  authorship: Record<string, number>;
  produces: string[];
  consumes: string[];
  episode: Record<string, unknown> | null;
  flags: Flag[];
}

export interface Entity {
  id: string;
  kind: string;
  name: string;
  aliases: string[];
  description: string;
}

export interface FactView {
  fact: {
    id: string;
    subject: string;
    predicate: string;
    valid_from_beat: string;
    valid_until_beat: string | null;
    established_by_beat: string;
    source: string;
  };
  sentence: string;
  subject_name: string;
  established_in: string;
  known_by: string[];
}

export interface Thread {
  id: string;
  description: string;
  opened_at_beat: string;
  last_touched_beat: string;
  status: "open" | "paid_off" | "dropped";
}

export interface SliceResponse {
  beat_id: string | null;
  entities: Entity[];
  facts: FactView[];
  threads: Thread[];
  hard_constraints: string[];
}

export interface Candidate {
  proposal_id: string;
  level: string;
  axis_label: string;
  rationale: string;
  delta_summary: string[];
  notes: string[];
  text: string;
  op_count: number;
  stale_preview: number;
  flags: Flag[];
  base_quality: number;
  surprise: number;
  effective_quality: number;
  selected: boolean;
}

export interface ProposeResponse {
  candidates: Candidate[];
  shown: string[];
}

export interface StaleMark {
  node_id: string;
  kind: "stale" | "review" | "maybe_affected";
  reason: string;
  origin_beat: string | null;
  origin_fact: string | null;
}

export interface ActionResponse {
  snapshot_id: string;
  bible_diff: string[];
  added: number;
  ended: number;
  removed: number;
  marks: StaleMark[];
  flags: Flag[];
  extracted: boolean;
}

export interface StyleNote {
  text: string;
  source: "writer" | "mined";
  count: number;
}

export interface Criterion {
  name: string;
  description: string;
  weight: number;
}

export interface LedgerResponse {
  dial: number;
  locks: string[];
  hard_constraints: string[];
  style_notes: StyleNote[];
  active_style_notes: string[];
  criteria: Criterion[];
  rejected_directions: string[];
  learned: Record<string, unknown>;
}

export interface BranchesResponse {
  current: string;
  branches: Record<string, string>;
}

export interface MergeResponse {
  clean: boolean;
  conflicts: Array<Record<string, unknown>>;
  summary: string[];
  committed: string | null;
}

export interface AuthorshipResponse {
  overall: Record<string, number>;
  sentences: number;
  by_node: Record<string, Record<string, number>>;
}

export interface GallerySessionIndex {
  name: string;
  title: string;
  summary: string;
  steps: number;
}

export interface GalleryStep {
  index: number;
  title: string;
  note: string;
  snapshot_id: string | null;
  node_id: string | null;
  level: string;
  intent: string;
  shown: Array<{
    proposal_id: string;
    axis_label: string;
    delta_summary: string[];
    rationale: string;
    text: string;
    flags: Array<Record<string, unknown>>;
    base_quality: number;
    surprise: number;
    effective_quality: number;
    selected: boolean;
  }>;
  action: string;
  chosen: string | null;
  bible_diff: string[];
  marks: StaleMark[];
  flags: Flag[];
}

export interface GalleryReplayEntry {
  step: GalleryStep;
  tree: Array<{ id: string; type: string; title: string; status: string; stale_reason: string | null }>;
  facts: string[];
}

export interface GalleryResponse {
  session: {
    name: string;
    title: string;
    summary: string;
    branch: string;
    steps: GalleryStep[];
  };
  replay?: GalleryReplayEntry[];
}

export interface EvalPlot {
  name: string;
  url: string;
  caption: string;
}

export interface Health {
  ok: boolean;
  branch: string;
  nodes: number;
  facts: number;
  providers: string[];
  openrouter_enabled: boolean;
}

/** A failure from the backend, carrying its typed kind. */
export class ApiError extends Error {
  readonly kind: string;
  readonly status: number;
  readonly retryAfter: number | null;

  constructor(status: number, kind: string, detail: string, retryAfter: number | null) {
    super(detail);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
    this.retryAfter = retryAfter;
  }

  /** Whether waiting and trying again could plausibly work. */
  get retryable(): boolean {
    return this.status === 429 || this.status === 503;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      ...init,
      headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch (cause) {
    // The server is not there at all. Distinguished from a server that answered with an
    // error, because the writer's next action is different: start it, versus retry.
    throw new ApiError(0, "Unreachable", `Cannot reach the API: ${String(cause)}`, null);
  }

  if (!response.ok) {
    let kind = `HTTP${response.status}`;
    let detail = response.statusText;
    let retryAfter: number | null = null;
    try {
      const body = await response.json();
      kind = body.kind ?? kind;
      detail = body.detail ?? (typeof body === "string" ? body : JSON.stringify(body));
      retryAfter = typeof body.retry_after === "number" ? body.retry_after : null;
    } catch {
      /* a non-JSON error body is still an error; keep the status text */
    }
    throw new ApiError(response.status, kind, detail, retryAfter);
  }
  return (await response.json()) as T;
}

function post<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });
}

export const api = {
  health: () => request<Health>("/api/health"),
  tree: (branch?: string) => request<TreeResponse>(`/api/tree${branch ? `?branch=${branch}` : ""}`),
  node: (id: string) => request<NodeDetail>(`/api/node/${id}`),
  slice: (nodeId: string) => request<SliceResponse>(`/api/slice?node=${nodeId}`),
  threads: () => request<{ threads: Array<{ thread: Thread; beats_since_touched: number; opened_in: string }> }>("/api/threads"),
  flags: (audit = false) => request<{ flags: Flag[]; summary: string }>(`/api/flags?audit=${audit}`),
  ledger: () => request<LedgerResponse>("/api/ledger"),
  authorship: () => request<AuthorshipResponse>("/api/authorship"),
  history: () => request<{ history: Array<Record<string, unknown>> }>("/api/history"),

  propose: (nodeId: string | null, level: string, intent: string) =>
    post<ProposeResponse>("/api/propose", { node_id: nodeId, level, intent }),
  accept: (proposalId: string) => post<ActionResponse>("/api/action/accept", { proposal_id: proposalId }),
  reject: (proposalId: string, reason: string) =>
    post<{ ok: boolean }>("/api/action/reject", { proposal_id: proposalId, reason }),
  edit: (proposalId: string, text: string) =>
    post<ActionResponse>("/api/action/edit", { proposal_id: proposalId, text }),
  write: (nodeId: string, text: string) => post<ActionResponse>("/api/action/write", { node_id: nodeId, text }),

  lock: (nodeId: string) => post<{ ok: boolean }>(`/api/node/${nodeId}/lock`),
  unlock: (nodeId: string) => post<{ ok: boolean }>(`/api/node/${nodeId}/unlock`),
  dismissStale: (nodeId: string) => post<{ ok: boolean }>(`/api/node/${nodeId}/dismiss-stale`),
  regenerate: (nodeId: string, subtree: boolean) =>
    post<ProposeResponse>(`/api/node/${nodeId}/regenerate?subtree=${subtree}`),
  strikeFact: (factId: string) => post<ActionResponse>(`/api/fact/${factId}/strike`),

  setDial: (value: number) => post<{ ok: boolean }>("/api/ledger/dial", { value }),
  addStyleNote: (text: string) => post<{ ok: boolean }>("/api/ledger/style-note", { text }),
  addCriterion: (name: string, description: string) =>
    post<{ ok: boolean }>("/api/ledger/criterion", { name, description, weight: 1.0 }),
  removeStyleNote: (text: string) => post<{ ok: boolean }>("/api/ledger/style-note/remove", { text }),
  removeCriterion: (name: string) => post<{ ok: boolean }>("/api/ledger/criterion/remove", { name }),
  mineEdits: () => post<{ ok: boolean; style_notes: StyleNote[] }>("/api/ledger/mine-edits"),

  branches: () => request<BranchesResponse>("/api/branches"),
  createBranch: (name: string) => post<{ ok: boolean }>("/api/branch", { name }),
  switchBranch: (name: string) => post<{ current: string }>("/api/branch/switch", { name }),
  diffBranches: (a: string, b: string) =>
    request<{ op_count: number; summary: string[] }>(`/api/branch/diff?a=${a}&b=${b}`),
  mergeBranches: (ours: string, theirs: string, commit: boolean) =>
    post<MergeResponse>("/api/branch/merge", { ours, theirs, commit }),

  gallery: () => request<{ sessions: GallerySessionIndex[] }>("/api/gallery"),
  gallerySession: (name: string) => request<GalleryResponse>(`/api/gallery/${name}`),
  evalSummary: () => request<Record<string, unknown> & { available: boolean }>("/api/eval/summary"),
  evalPlots: () => request<{ plots: EvalPlot[] }>("/api/eval/plots"),
};
