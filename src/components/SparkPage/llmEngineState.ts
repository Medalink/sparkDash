import type { LlmEngineState, LlmMetrics } from "../../api/types";

export interface EngineStateLabel {
  text: string;
  /** Hover text; set when the value needs explaining. */
  title?: string;
  /** Render muted — the value is an absence, not a reading. */
  muted: boolean;
  /** Render as a problem (stalled or failed engine). */
  alert?: boolean;
}

/** Engine tile text for backends that report an explicit state (TensorFold). */
const ENGINE_STATE_LABEL: Record<LlmEngineState, string> = {
  ready: "Active · idle",
  busy: "Active · busy",
  stalled: "Stalled",
  error: "Error",
};
const ENGINE_STATE_ALERT: Record<LlmEngineState, boolean> = {
  ready: false,
  busy: false,
  stalled: true,
  error: true,
};

/**
 * The LLM panel's Engine tile. `gpuMemoryUtilization` carries the engine
 * state only where the probe fills it (vLLM's sleep gauge; SGLang and q27,
 * which are always resident, report Active). For the rest — TensorFold,
 * llama.cpp, … — a bare dash read like a failed probe, so the tile says the
 * backend does not report it. The dash is kept for "no LLM at all".
 */
export function engineStateLabel(llm: LlmMetrics | null | undefined): EngineStateLabel {
  if (!llm) return { text: "—", muted: false };
  if (llm.engineState) {
    return {
      text: ENGINE_STATE_LABEL[llm.engineState],
      muted: false,
      alert: ENGINE_STATE_ALERT[llm.engineState],
    };
  }
  // A successful Strata probe requires a loaded engine. Idle means the model
  // is ready for work, not asleep; it has no vLLM-style sleep gauge.
  if (llm.backend === "strata") {
    return llm.available
      ? { text: "Active", muted: false }
      : { text: "—", muted: true };
  }
  if (llm.gpuMemoryUtilization == null) {
    return {
      text: "n/a",
      title: "This backend does not report engine sleep state",
      muted: true,
    };
  }
  return { text: llm.gpuMemoryUtilization === 0 ? "Sleeping" : "Active", muted: false };
}
