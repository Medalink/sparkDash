/**
 * TensorFold engine state (`/health`): the Engine tile reads Active · idle /
 * Active · busy / Stalled / Error, or "-" when /health did not answer with a
 * JSON object. The state is computed after every source of the running count
 * has run, `stalled` requires work still in flight, and a failure status with
 * an `ok: false` body is an Error even though the body is not applied.
 */
import { test } from "node:test";
import { strict as assert } from "node:assert";
import { LlmProbe } from "../LlmProbe.js";

const MODELS = {
  object: "list",
  data: [{ id: "GLM-5.3-Flash-EXL3", object: "model", owned_by: "tensorfold" }],
};

function res(body, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => (typeof body === "string" ? JSON.parse(body) : body),
    text: async () => (typeof body === "string" ? body : JSON.stringify(body)),
  };
}

/** _fetch stub; `health` may be a body, a [status, body] pair, or null for a refused connection. */
function server({ health = {} } = {}) {
  return async (url) => {
    const u = String(url);
    if (u.endsWith("/v1/models")) return res(MODELS);
    if (u.endsWith("/health")) {
      if (health === null) throw new Error("connect ECONNREFUSED");
      if (Array.isArray(health)) return res(health[1], health[0]);
      return res(health);
    }
    return res({ error: "not found" }, 404);
  };
}

function newProbe() {
  return new LlmProbe({ lanIp: "127.0.0.1" }, 8888);
}

const IDLE = { ok: true, backend: "tensorfold", busy: false, streams: { decoding: 0, filling: 0, paused: 0, max: 4 } };
const BUSY = { ...IDLE, busy: true, streams: { decoding: 2, filling: 1, paused: 0, max: 4 } };

test("engine state: busy from occupied lanes, ready when idle", async () => {
  let probe = newProbe();
  probe._fetch = server({ health: BUSY });
  let snap = await probe.probe();
  assert.equal(snap.engineState, "busy");

  probe = newProbe();
  probe._fetch = server({ health: IDLE });
  snap = await probe.probe();
  assert.equal(snap.engineState, "ready");
});

test("engine state: a truthy numeric busy flag reads busy (no streams needed)", async () => {
  const probe = newProbe();
  probe._fetch = server({ health: { ok: true, backend: "tensorfold", busy: 1 } });
  const snap = await probe.probe();
  assert.equal(snap.engineState, "busy");
});

test("engine state: ok:false reads Error", async () => {
  const probe = newProbe();
  probe._fetch = server({ health: { ...IDLE, ok: false } });
  const snap = await probe.probe();
  assert.equal(snap.engineState, "error");
});

test("engine state: a failure status with an ok:false body reads Error", async () => {
  const probe = newProbe();
  probe._fetch = server({ health: [503, { ok: false, error: "engine is shutting down" }] });
  const snap = await probe.probe();
  assert.equal(snap.engineState, "error");
});

test("engine state: a failure status with a non-JSON body reads null, not Error", async () => {
  const probe = newProbe();
  probe._fetch = server({ health: [503, "<html>gateway timeout</html>"] });
  const snap = await probe.probe();
  assert.equal(snap.engineState, null);
});

test("engine state: a step past the stalled threshold only counts while requests run", async () => {
  let probe = newProbe();
  probe._fetch = server({ health: { ...IDLE, iteration_s: 900 } });
  let snap = await probe.probe();
  assert.equal(snap.engineState, "ready"); // an idle engine is not stalled, however long its last step ran

  probe = newProbe();
  probe._fetch = server({ health: { ...BUSY, iteration_s: 299.9 } });
  snap = await probe.probe();
  assert.equal(snap.engineState, "busy");

  probe = newProbe();
  probe._fetch = server({ health: { ...BUSY, iteration_s: 300 } });
  snap = await probe.probe();
  assert.equal(snap.engineState, "stalled");
});

test("engine state: no /health answer reads null", async () => {
  const probe = newProbe();
  probe._fetch = server({ health: null });
  const snap = await probe.probe();
  assert.equal(snap.engineState, null);
});
