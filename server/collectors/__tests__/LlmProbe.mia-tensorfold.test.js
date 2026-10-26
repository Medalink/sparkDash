/**
 * Mia's TensorFold builds polled through the public probe():
 * - the GLM CUDA engine (/health with lifetime totals whose wait gauge, latency
 *   histograms and preemptions live in /metrics), and
 * - the DeepSeek Zig tensorfold-native engine (bare-ok /health; counters,
 *   histograms under the e2e_ alias included, live in /metrics alone).
 * The lifetime totals always win over the /metrics twins, which cover finished
 * requests only.
 */
import { test } from "node:test";
import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { LlmProbe } from "../LlmProbe.js";

const FIXTURES = join(dirname(fileURLToPath(import.meta.url)), "fixtures", "mia-tensorfold");
const MIA_HEALTH = JSON.parse(readFileSync(join(FIXTURES, "health.json"), "utf8"));
const MIA_METRICS = readFileSync(join(FIXTURES, "metrics.txt"), "utf8");

const MODELS = { object: "list", data: [{ id: "GLM-5.3-Flash-EXL3", object: "model", owned_by: "tensorfold" }] };

function res(body, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => (typeof body === "string" ? JSON.parse(body) : body),
    text: async () => (typeof body === "string" ? body : JSON.stringify(body)),
  };
}

function server({ health = MIA_HEALTH, metrics = MIA_METRICS } = {}) {
  return async (url) => {
    const u = String(url);
    if (u.endsWith("/v1/models")) return res(MODELS);
    if (u.endsWith("/health")) return res(health);
    if (u.endsWith("/metrics")) return metrics == null ? res({ error: "not found" }, 404) : res(metrics);
    return res({ error: "not found" }, 404);
  };
}

// DeepSeek Zig tensorfold-native: bare-ok /health, everything else in /metrics.
// The E2E histogram is named e2e_request_latency_seconds; there is no cached-token
// counter (issue #281's spill v2 shape).
const DS_HEALTH = { ok: true, backend: "tensorfold", busy: false };
const DS_METRICS = [
  "tensorfold:requests_running 2",
  "tensorfold:requests_waiting 0",
  "tensorfold:num_requests_running 2",
  "tensorfold:prompt_tokens_total 90000",
  "tensorfold:generation_tokens_total 21000",
  "tensorfold:mtp_drafted_total 3000",
  "tensorfold:mtp_accepted_total 2250",
  "tensorfold:preemptions_total 7",
  "tensorfold:e2e_request_latency_seconds_sum 21",
  "tensorfold:e2e_request_latency_seconds_count 4",
  'tensorfold:e2e_request_latency_seconds_bucket{le="1"} 1',
  'tensorfold:e2e_request_latency_seconds_bucket{le="5"} 2',
  'tensorfold:e2e_request_latency_seconds_bucket{le="10"} 3',
  'tensorfold:e2e_request_latency_seconds_bucket{le="30"} 4',
  'tensorfold:e2e_request_latency_seconds_bucket{le="+Inf"} 4',
  "tensorfold:time_to_first_token_seconds_sum 1.4",
  "tensorfold:time_to_first_token_seconds_count 4",
  'tensorfold:time_to_first_token_seconds_bucket{le="0.1"} 1',
  'tensorfold:time_to_first_token_seconds_bucket{le="0.5"} 3',
  'tensorfold:time_to_first_token_seconds_bucket{le="1"} 4',
  'tensorfold:time_to_first_token_seconds_bucket{le="+Inf"} 4',
].join("\n");

test("mia GLM build: lifetime /health totals win; /metrics adds the wait gauge, p95s and MTP", async () => {
  const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8888);
  probe._fetch = server();
  const snap = await probe.probe();
  assert.equal(snap.available, true);
  // Lifetime totals from /health - not the finished-request counters of /metrics.
  assert.equal(snap.totalPromptTokens, 3120450);
  assert.equal(snap.totalOutputTokens, 288977);
  // Wait gauge, latency histograms: /metrics alone carries them.
  assert.equal(snap.requestsWaiting, 1);
  assert.equal(snap.ttftP95Seconds, 10);
  assert.equal(snap.e2eP95Seconds, 120);
  assert.equal(snap.mtpAcceptanceRate, 0.75); // 183600 / 244800
  // The cache split comes from /health's sticky cached_tokens_total ...
  assert.equal(snap.totalCachedTokens, 1870000);
  assert.equal(snap.prefixCacheHitRate, 0.5993); // 1870000 / 3120450
  assert.equal(snap.requestsRunning, 3); // streams: decoding 2 + filling 1
});

test("mia GLM build: a /metrics outage keeps every /health tile", async () => {
  const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8888);
  probe._fetch = server({ metrics: null });
  const snap = await probe.probe();
  assert.equal(snap.totalOutputTokens, 288977);
  assert.equal(snap.mtpAcceptanceRate, 0.75);
  assert.equal(snap.totalCachedTokens, 1870000);
  assert.equal(snap.prefixCacheHitRate, 0.5993);
  // Only the /metrics-sourced tiles go dark.
  assert.equal(snap.requestsWaiting, null);
  assert.equal(snap.ttftP95Seconds, null);
  assert.equal(snap.e2eP95Seconds, null);
});

test("mia DeepSeek build: bare-ok /health, counters and the e2e alias in /metrics", async () => {
  const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8889);
  probe._fetch = server({ health: DS_HEALTH, metrics: DS_METRICS });
  const snap = await probe.probe();
  assert.equal(snap.available, true);
  assert.equal(snap.totalPromptTokens, 90000);
  assert.equal(snap.totalOutputTokens, 21000);
  assert.equal(snap.requestsRunning, 2);
  // The spill v2 alias name is read (issue #281).
  assert.equal(snap.e2eP95Seconds, 26); // interpolated between le=10 and le=30
  assert.equal(snap.ttftP95Seconds, 0.9);
  assert.equal(snap.mtpAcceptanceRate, 0.75); // 2250 / 3000
  assert.equal(snap.preemptionsTotal, 7);
  // No cached-token series on this build: the ledger stays null.
  assert.equal(snap.totalCachedTokens, null);
  assert.equal(snap.prefixCacheHitRate, null);
});
