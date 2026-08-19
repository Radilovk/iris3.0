/**
 * worker.js — Cloudflare Worker
 * Iris Iridology Analysis Pipeline v11 — 3-Call Precision Architecture
 *
 * Architecture (3 AI calls for maximum precision, no Flask needed):
 *   CALL 1 (vision+image): Full Detection — geo calibration + structural + pigment + ANW collarette
 *   CALL 2 (vision+image): Verification — re-examine image with CALL1 findings, validate, refine, zone map
 *   CALL 3 (text only):    Report — synthesize into Bulgarian UI format with advice
 *
 * Why 3 calls:
 *   - CALL 1 gets max token budget for thorough initial detection with image
 *   - CALL 2 re-examines the image knowing what was found — catches misses, corrects false positives
 *   - CALL 3 has full context from verified findings — produces precise Bulgarian report
 *   - Each call has focused role → better precision than one overloaded prompt
 *   - 3 calls vs 8 original → 62% fewer API requests while maintaining quality
 *
 * Environment bindings (set in wrangler.toml / Cloudflare dashboard):
 *   iris_rag_kv     — KV namespace for caching results
 *   AI_API_KEY      — secret: API key (OpenAI or Google Gemini)
 *   AI_PROVIDER     — var: provider name ("openai", "gemini", "openai-compatible")
 *   AI_MODEL        — var: model name (e.g., "gemini-2.0-flash", "gpt-4o")
 *   AI_BASE_URL     — var: OpenAI API base URL
 *   GEMINI_API_URL  — var: Gemini API base URL
 *
 * Request format (POST /analyze, multipart/form-data):
 *   strip_image   — base64-encoded JPEG of the unwrapped iris strip
 *   side          — "R" or "L"
 *   image_hash    — optional unique ID; auto-generated if omitted
 *   questionnaire — optional JSON string with patient data
 */

// =====================================================================
// CORS HEADERS
// =====================================================================
const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

const ERR_MSG_LIMIT = 300;

// =====================================================================
// ENTRY POINT
// =====================================================================
export default {
  async fetch(request, env) {
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: CORS });
    }

    const url = new URL(request.url);

    try {
      if (request.method === 'POST' && url.pathname === '/analyze') {
        return await handleAnalyze(request, env);
      }
      if (request.method === 'GET' && url.pathname.startsWith('/result/')) {
        const key = url.pathname.slice('/result/'.length);
        return await handleGetResult(key, env);
      }
      if (request.method === 'GET' && url.pathname === '/models') {
        return handleGetModels(env);
      }
      if (request.method === 'GET' && url.pathname === '/health') {
        return handleHealthCheck(env);
      }
      if (url.pathname.startsWith('/admin')) {
        return await handleAdmin(request, env, url);
      }
      return jsonResp({ error: 'Not Found' }, 404);
    } catch (err) {
      const msg = err?.message || String(err);
      return jsonResp({ error: 'Internal server error', detail: msg.slice(0, ERR_MSG_LIMIT) }, 500);
    }
  },
};

// =====================================================================
// Available AI Models Configuration
// =====================================================================
const AVAILABLE_MODELS = {
  gemini: {
    name: 'Google Gemini',
    models: [
      { id: 'gemini-2.0-flash', name: 'Gemini 2.0 Flash', recommended: true, vision: true },
      { id: 'gemini-2.5-flash-preview-04-17', name: 'Gemini 2.5 Flash Preview', vision: true },
      { id: 'gemini-2.5-flash-latest', name: 'Gemini 2.5 Flash Latest', vision: true },
      { id: 'gemini-2.0-flash-lite', name: 'Gemini 2.0 Flash Lite', vision: true, costEffective: true },
      { id: 'gemini-1.5-pro-latest', name: 'Gemini 1.5 Pro Latest', vision: true },
      { id: 'gemini-1.5-flash-latest', name: 'Gemini 1.5 Flash Latest', vision: true },
    ]
  },
  openai: {
    name: 'OpenAI',
    models: [
      { id: 'gpt-4o', name: 'GPT-4o', recommended: true, vision: true },
      { id: 'gpt-4o-mini', name: 'GPT-4o Mini', vision: true, costEffective: true },
      { id: 'gpt-4-turbo', name: 'GPT-4 Turbo', vision: true },
    ]
  },
  'openai-compatible': {
    name: 'OpenAI-Compatible APIs',
    models: [
      { id: 'claude-3-opus-20240229', name: 'Claude 3 Opus', vision: true },
      { id: 'claude-3-sonnet-20240229', name: 'Claude 3 Sonnet', vision: true },
      { id: 'claude-3-haiku-20240307', name: 'Claude 3 Haiku', vision: true, costEffective: true },
    ]
  }
};

// =====================================================================
// ROUTE: GET /models
// =====================================================================
function handleGetModels(env) {
  const currentProvider = env.AI_PROVIDER || 'gemini';
  const currentModel = env.AI_MODEL || 'gemini-2.0-flash';

  return jsonResp({
    currentConfig: { provider: currentProvider, model: currentModel },
    availableProviders: AVAILABLE_MODELS,
    note: 'You can override the model per-request by passing ai_provider and ai_model in the /analyze request',
  });
}

// =====================================================================
// ROUTE: GET /health
// =====================================================================
function handleHealthCheck(env) {
  const hasApiKey = !!env.AI_API_KEY;
  const provider = env.AI_PROVIDER || 'gemini';
  const model = env.AI_MODEL || 'gemini-2.0-flash';

  return jsonResp({
    status: hasApiKey ? 'healthy' : 'degraded',
    version: 'v11.0-3call-precision',
    provider,
    model,
    apiKeyConfigured: hasApiKey,
    kvConfigured: !!env.iris_rag_kv,
    pipelineSteps: 3,
    pipelineDescription: 'CALL1(vision:detect) → CALL2(vision:verify+map) → CALL3(text:report)',
  });
}

// =====================================================================
// ROUTE: POST /analyze — 3-Call Precision Pipeline
// =====================================================================
async function handleAnalyze(request, env) {
  const form = await request.formData();
  const side         = (form.get('side') || 'R').toUpperCase();
  const stripB64     = form.get('strip_image');
  const imageHash    = form.get('image_hash') || genId();
  const qRaw         = form.get('questionnaire');
  const questionnaire = qRaw ? safeParseJSON(qRaw) : {};
  const cqRaw        = form.get('capture_quality');
  const captureQuality = cqRaw ? safeParseJSON(cqRaw) : null;

  const aiProvider   = form.get('ai_provider') || null;
  const aiModel      = form.get('ai_model') || null;

  if (!stripB64) {
    return jsonResp({ error: 'strip_image is required (base64 JPEG of the unwrapped iris strip)' }, 400);
  }
  if (side !== 'R' && side !== 'L') {
    return jsonResp({ error: 'side must be "R" or "L"' }, 400);
  }

  const kvConfig = await getKVConfig(env);
  const effectiveEnv = createEffectiveEnv(env, aiProvider, aiModel, kvConfig);
  const effectiveModel = effectiveEnv.AI_MODEL;
  const cacheKey = `result:${side}:${imageHash}:${effectiveModel}`;

  // Check cache
  let cached = null;
  try {
    cached = await env.iris_rag_kv.get(cacheKey, 'json');
  } catch (kvErr) {
    // KV read failure is non-fatal; proceed to run the pipeline
    console.error('KV get error:', kvErr?.message || kvErr);
  }
  if (cached) {
    return jsonResp({ cached: true, imageHash, side, model: effectiveModel, result: cached });
  }

  // Run 3-call pipeline
  const pipeline = new IrisPipeline(effectiveEnv, stripB64, side, imageHash, questionnaire, captureQuality);
  const result = await pipeline.run();

  // Store in KV with 24-hour TTL (even errors, to avoid hammering AI on bad images)
  await env.iris_rag_kv.put(cacheKey, JSON.stringify(result), { expirationTtl: 86400 }).catch(() => {});

  return jsonResp({ cached: false, imageHash, side, model: effectiveModel, result });
}

function createEffectiveEnv(env, aiProvider, aiModel, kvConfig) {
  return {
    iris_rag_kv: env.iris_rag_kv,
    AI_API_KEY: kvConfig?.apiKey || env.AI_API_KEY,
    AI_BASE_URL: kvConfig?.baseUrl || env.AI_BASE_URL,
    GEMINI_API_URL: kvConfig?.geminiApiUrl || env.GEMINI_API_URL,
    AI_PROVIDER: aiProvider || kvConfig?.provider || env.AI_PROVIDER || 'gemini',
    AI_MODEL: aiModel || kvConfig?.model || env.AI_MODEL || 'gemini-2.0-flash',
  };
}

// =====================================================================
// ROUTE: GET /result/:key
// =====================================================================
async function handleGetResult(key, env) {
  const cacheKey = `result:${key}`;
  const result = await env.iris_rag_kv.get(cacheKey, 'json').catch(() => null);
  if (!result) {
    return jsonResp({ error: 'Result not found or expired (24h TTL)' }, 404);
  }
  return jsonResp({ cached: true, result });
}

// =====================================================================
// 3-CALL PIPELINE CLASS
// =====================================================================
class IrisPipeline {
  constructor(env, stripB64, side, imageHash, questionnaire, captureQuality) {
    this.env          = env;
    this.imageB64     = stripB64;
    this.imageDataUrl = `data:image/jpeg;base64,${stripB64}`;
    this.side         = side;
    this.imageHash    = imageHash;
    this.questionnaire = questionnaire;
    // How good the capture actually was, measured by the client geometry stage.
    this.captureQuality = captureQuality || null;
  }

  async run() {
    // ── CALL 1: Full Detection (vision + image) ──────────────────────
    // Geo calibration + structural + pigment + ANW collarette — all in one vision call
    let call1 = await this.visionCall(promptCall1_Detect(this.side, this.imageHash));
    if (call1.error) {
      return { error: call1.error || call1, stage: 'CALL1_DETECT', imageHash: this.imageHash, side: this.side };
    }
    // The model only ever reads sector/ringGroup labels off the image; the
    // numeric minuteRange/ringRange used for zone matching is computed here,
    // deterministically, instead of trusting the model's own arithmetic.
    call1 = postProcessDetection(call1);

    // ── CALL 2: Verification & Zone Mapping (vision + image) ─────────
    // Re-examine image with CALL1's findings: validate, refine, consistency check, zone mapping
    let call2 = await this.visionCall(promptCall2_Verify(this.side, this.imageHash, call1));
    if (call2.error) {
      return { error: call2.error || call2, stage: 'CALL2_VERIFY', imageHash: this.imageHash, side: this.side };
    }
    call2 = postProcessDetection(call2);
    // Organ attribution happens here, in code, from the labels the model read —
    // never by asking the model to re-derive coordinates.
    call2 = attachZones(call2, this.side);

    // ── CALL 3: Report Generation (text only — no image needed) ──────
    // Synthesize verified findings into Bulgarian UI format with advice
    const call3 = await this.textCall(
      promptCall3_Report(this.side, this.imageHash, call1, call2, this.questionnaire, this.captureQuality)
    );

    return {
      imageHash: this.imageHash,
      side: this.side,
      ...call3,
      // Merge raw pipeline steps into call3's pipeline for debugging
      pipeline: {
        ...(call3.pipeline || {}),
        _call1: call1,
        _call2: call2,
      },
    };
  }

  async visionCall(prompt) {
    return aiCall(this.env, prompt, this.imageDataUrl);
  }

  async textCall(prompt) {
    return aiCall(this.env, prompt, null);
  }
}

// =====================================================================
// AI CALL — Multi-provider (OpenAI, Gemini)
// =====================================================================
async function aiCall(env, prompt, imageDataUrl) {
  const provider = (env.AI_PROVIDER || 'openai').toLowerCase();

  if (provider === 'gemini') {
    return await aiCallGemini(env, prompt, imageDataUrl);
  } else {
    return await aiCallOpenAI(env, prompt, imageDataUrl);
  }
}

// =====================================================================
// OpenAI-compatible API call
// =====================================================================
async function aiCallOpenAI(env, prompt, imageDataUrl) {
  const model   = env.AI_MODEL   || 'gpt-4o';
  const baseUrl = env.AI_BASE_URL || 'https://api.openai.com/v1';

  const userContent = imageDataUrl
    ? [
        { type: 'text', text: prompt },
        { type: 'image_url', image_url: { url: imageDataUrl, detail: 'high' } },
      ]
    : prompt;

  const body = {
    model,
    messages: [{ role: 'user', content: userContent }],
    temperature: 0.1,
    max_tokens: 16384,
    response_format: { type: 'json_object' },
  };

  let resp;
  try {
    resp = await fetch(`${baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${env.AI_API_KEY}`,
      },
      body: JSON.stringify(body),
    });
  } catch (err) {
    return { error: { code: 'NETWORK_ERROR', message: (err?.message || String(err)).slice(0, ERR_MSG_LIMIT) } };
  }

  if (!resp.ok) {
    const errText = await resp.text().catch(() => resp.statusText);
    return { error: { code: 'API_HTTP_ERROR', status: resp.status, message: errText.slice(0, ERR_MSG_LIMIT) } };
  }

  const data = await resp.json().catch(() => null);
  const content = data?.choices?.[0]?.message?.content;
  if (!content) {
    return { error: { code: 'EMPTY_RESPONSE', message: 'AI returned no content' } };
  }

  return safeParseJSON(content) || { error: { code: 'JSON_PARSE_ERROR', raw: content.slice(0, ERR_MSG_LIMIT) } };
}

// =====================================================================
// Google Gemini API call
// =====================================================================
async function aiCallGemini(env, prompt, imageDataUrl) {
  const model = env.AI_MODEL || 'gemini-2.0-flash';
  const baseUrl = env.GEMINI_API_URL || 'https://generativelanguage.googleapis.com/v1beta';

  /** @type {Array<{text: any} | {inline_data: {mime_type: any, data: any}}>} */
  const parts = [{ text: prompt }];

  if (imageDataUrl) {
    const matches = imageDataUrl.match(/^data:(image\/[^;]+);base64,(.+)$/);
    if (matches) {
      parts.push({
        inline_data: {
          mime_type: matches[1],
          data: matches[2]
        }
      });
    }
  }

  const body = {
    contents: [{ parts }],
    generationConfig: {
      temperature: 0.1,
      maxOutputTokens: 16384,
      responseMimeType: 'application/json'
    }
  };

  let resp;
  try {
    resp = await fetch(`${baseUrl}/models/${model}:generateContent?key=${env.AI_API_KEY}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch (err) {
    return { error: { code: 'NETWORK_ERROR', message: (err?.message || String(err)).slice(0, ERR_MSG_LIMIT) } };
  }

  if (!resp.ok) {
    const errText = await resp.text().catch(() => resp.statusText);
    return { error: { code: 'API_HTTP_ERROR', status: resp.status, message: errText.slice(0, ERR_MSG_LIMIT) } };
  }

  const data = await resp.json().catch(() => null);
  const content = data?.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!content) {
    return { error: { code: 'EMPTY_RESPONSE', message: 'Gemini returned no content' } };
  }

  return safeParseJSON(content) || { error: { code: 'JSON_PARSE_ERROR', raw: content.slice(0, ERR_MSG_LIMIT) } };
}

// =====================================================================
// IMAGE FORMAT PREAMBLE (shared across vision calls)
// =====================================================================
const IMAGE_FORMAT = `\
================================================================================
IMAGE_FORMAT: UNWRAPPED_IRIS_STRIP (READ THIS BEFORE ANYTHING ELSE)
================================================================================

CRITICAL NOTICE:
The image you are analyzing is NOT a raw circular iris photograph.
It is an UNWRAPPED (unrolled / linearized) iris strip produced by a
polar-to-rectangular transformation of the original circular iris image,
then RESHAPED so the metabolism/digestion/endocrine-relevant organ zone
gets much more vertical space than the rest.

LOCALIZATION METHOD — CLASSIFICATION, NOT MEASUREMENT:
This image has two label systems PRINTED DIRECTLY ON IT. Locating a finding
means READING which printed label it falls under — like picking a row/column
in a spreadsheet — NOT estimating a pixel coordinate or doing geometry.
DO NOT compute atan2, radial distance, or any pixel-offset formula. If you
find yourself estimating "this is about 42% of the way across", stop — look
again at which printed label the finding sits under/between.

1) SECTOR (horizontal position, printed as "S1".."S12" above the minute
   ticks, one per every 5-minute column):
     S1 = minutes 0-5 (12 o'clock)     S7  = minutes 30-35 (6 o'clock)
     S2 = minutes 5-10                 S8  = minutes 35-40
     S3 = minutes 10-15 (3 o'clock)    S9  = minutes 40-45
     S4 = minutes 15-20                S10 = minutes 45-50
     S5 = minutes 20-25                S11 = minutes 50-55
     S6 = minutes 25-30 (6 o'clock)    S12 = minutes 55-60 (12 o'clock)
   The strip also has faint alternating vertical shading (every other
   sector slightly tinted) so you can SEE the sector boundaries directly,
   not just infer them from the tick numbers.
   RIGHT EYE: TEMPORAL ~ S3 (3 o'clock), NASAL ~ S9 (9 o'clock)
   LEFT  EYE: NASAL ~ S3 (3 o'clock), TEMPORAL ~ S9 (9 o'clock)

2) RING GROUP (vertical/depth position, printed as a name in the left
   margin, with a bold horizontal line at every group boundary):
     IPB     — thin band, innermost, touching the pupil edge
     STOM    — thin band, just below IPB (stomach ring)
     ANW     — the wavy collarette band (autonomic nerve wreath)
     ORG_IN  — inner third of the organ zone (largest band on the strip)
     ORG_MID — middle third of the organ zone
     ORG_OUT — outer third of the organ zone
     LYM     — thin band near the outer edge (lymphatic)
     SCU     — outermost thin band, touching the limbus (skin/scurf)
   ORG_IN / ORG_MID / ORG_OUT together hold nearly all digestion,
   metabolism and endocrine organs and are drawn much taller than the
   other bands specifically so you can tell them apart reliably — use
   that space; do not default to "ORG_MID" out of uncertainty when the
   finding is visibly closer to one of the bold boundary lines.
   Faint thin gray lines and small "R4"/"R5" etc. labels inside ORG_IN/
   ORG_MID/ORG_OUT/ANW are an OPTIONAL finer reference only — the ring
   GROUP is what matters; do not strain for individual-ring precision.

WHEN A FINDING SPANS MORE THAN ONE SECTOR OR BAND:
  - Report sectorRange as [firstSector, lastSector] (e.g. [3,5]).
  - sectorRange must never wrap past S12→S1: if a finding straddles the
    12 o'clock line, split it into two findings, one ending at S12 and
    one starting at S1.
  - If a finding clearly crosses ring-group boundaries, set ringGroup to
    where it starts and ringGroupEnd to where it ends (omit ringGroupEnd
    if it stays within one band).

HOW FEATURES APPEAR:
  CONCENTRIC (rings, ANW, bands) = HORIZONTAL stripes left-to-right
  RADIAL (furrows, clefts)       = VERTICAL dark stripes top-to-bottom
  POINT/LOCAL (lacunae, crypts)  = Discrete patches
  EYELID-MASKED areas            = White/blank bands (DO NOT report findings there)
================================================================================`;

// =====================================================================
// SECTOR / RING-GROUP → CANONICAL NUMERIC RANGE (deterministic, no AI math)
// =====================================================================
const RING_GROUP_ORDER = ['IPB', 'STOM', 'ANW', 'ORG_IN', 'ORG_MID', 'ORG_OUT', 'LYM', 'SCU'];
const RING_GROUP_RANGE = {
  IPB: [0, 0], STOM: [1, 1], ANW: [2, 3],
  ORG_IN: [4, 5], ORG_MID: [6, 7], ORG_OUT: [8, 9],
  LYM: [10, 10], SCU: [11, 11],
};

function sectorRangeToMinuteRange(sectorRange) {
  if (!Array.isArray(sectorRange) || sectorRange.length !== 2) return [0, 5];
  const [s0, s1] = sectorRange;
  const a = Math.min(Math.max(1, s0 | 0), 12);
  const b = Math.min(Math.max(1, s1 | 0), 12);
  return [(Math.min(a, b) - 1) * 5, Math.max(a, b) * 5];
}

function ringGroupToRingRange(ringGroup, ringGroupEnd) {
  const start = RING_GROUP_RANGE[ringGroup] || RING_GROUP_RANGE.ORG_MID;
  const end = ringGroupEnd ? (RING_GROUP_RANGE[ringGroupEnd] || start) : start;
  return [Math.min(start[0], end[0]), Math.max(start[1], end[1])];
}

// Deterministically converts a detection's sectorRange/ringGroup labels
// (which the AI reads directly off the printed image) into the canonical
// minuteRange/ringRange used for zone matching downstream — keeps that
// arithmetic out of the model's hands entirely.
function canonicalizeFinding(f) {
  return {
    ...f,
    minuteRange: sectorRangeToMinuteRange(f.sectorRange),
    ringRange: ringGroupToRingRange(f.ringGroup, f.ringGroupEnd),
  };
}

function postProcessDetection(json) {
  if (!json || json.error) return json;
  const out = { ...json };
  if (Array.isArray(out.structural)) out.structural = out.structural.map(canonicalizeFinding);
  if (Array.isArray(out.pigment)) out.pigment = out.pigment.map(canonicalizeFinding);
  if (Array.isArray(out.verified_structural)) out.verified_structural = out.verified_structural.map(canonicalizeFinding);
  if (Array.isArray(out.verified_pigment)) out.verified_pigment = out.verified_pigment.map(canonicalizeFinding);
  if (out.collarette?.segments) {
    out.collarette = {
      ...out.collarette,
      segments: out.collarette.segments.map(s => ({ ...s, minuteRange: sectorRangeToMinuteRange([s.seg, s.seg]) })),
    };
  }
  if (out.collarette_verified?.segments) {
    out.collarette_verified = {
      ...out.collarette_verified,
      segments: out.collarette_verified.segments.map(s => ({ ...s, minuteRange: sectorRangeToMinuteRange([s.seg, s.seg]) })),
    };
  }
  return out;
}

// =====================================================================
// IRIDOLOGY ZONE MAP (v9)
// =====================================================================
const MAP_V9 = [
  { id: 'ANY-stomach',      side: 'ANY', mins: [0,  59], rings: [1,  1],  organ_bg: 'Стомах',                     system_bg: 'Храносмилателна' },
  { id: 'ANY-sm-intest',    side: 'ANY', mins: [0,  59], rings: [3,  4],  organ_bg: 'Тънко черво',                 system_bg: 'Храносмилателна' },
  { id: 'ANY-ANW',          side: 'ANY', mins: [0,  59], rings: [2,  3],  organ_bg: 'Автономна нервна система',   system_bg: 'Нервна' },
  { id: 'ANY-LYM',          side: 'ANY', mins: [0,  59], rings: [10, 10], organ_bg: 'Лимфна система',             system_bg: 'Имунна' },
  { id: 'ANY-SCU',          side: 'ANY', mins: [0,  59], rings: [11, 11], organ_bg: 'Кожа / Детоксикация',        system_bg: 'Детоксикация' },
  { id: 'ANY-spine-cerv-u', side: 'ANY', mins: [56, 59], rings: [8, 10],  organ_bg: 'Гръбначен стълб (шиен)',     system_bg: 'Опорно-двигателна' },
  { id: 'ANY-spine-cerv-l', side: 'ANY', mins: [0,  4],  rings: [8, 10],  organ_bg: 'Гръбначен стълб (шиен)',     system_bg: 'Опорно-двигателна' },
  { id: 'R-brain-motor',    side: 'R',   mins: [0,  3],  rings: [4,  9],  organ_bg: 'Мозък (моторни зони)',       system_bg: 'Нервна' },
  { id: 'R-brain-sens',     side: 'R',   mins: [57, 59], rings: [4,  9],  organ_bg: 'Мозък (сетивни зони)',       system_bg: 'Нервна' },
  { id: 'R-sinus',          side: 'R',   mins: [2,  6],  rings: [4,  7],  organ_bg: 'Синуси (десни)',             system_bg: 'Дихателна' },
  { id: 'R-larynx',         side: 'R',   mins: [5,  10], rings: [4,  5],  organ_bg: 'Ларинкс / Гърло',           system_bg: 'Дихателна' },
  { id: 'R-thyroid',        side: 'R',   mins: [7,  13], rings: [4,  6],  organ_bg: 'Щитовидна жлеза (дясна)',   system_bg: 'Ендокринна' },
  { id: 'R-eye-ear',        side: 'R',   mins: [10, 17], rings: [4,  7],  organ_bg: 'Ухо / Очи',                 system_bg: 'Нервна' },
  { id: 'R-bronchi',        side: 'R',   mins: [14, 19], rings: [4,  6],  organ_bg: 'Бронхи (десни)',             system_bg: 'Дихателна' },
  { id: 'R-shoulder',       side: 'R',   mins: [18, 22], rings: [4,  9],  organ_bg: 'Рамо (дясно)',               system_bg: 'Опорно-двигателна' },
  { id: 'R-lung',           side: 'R',   mins: [20, 30], rings: [4,  7],  organ_bg: 'Бял дроб (десен)',           system_bg: 'Дихателна' },
  { id: 'R-liver',          side: 'R',   mins: [23, 35], rings: [4,  9],  organ_bg: 'Черен дроб',                 system_bg: 'Детоксикация' },
  { id: 'R-bladder',        side: 'R',   mins: [25, 30], rings: [7,  9],  organ_bg: 'Пикочен мехур',             system_bg: 'Отделителна' },
  { id: 'R-gallbladder',    side: 'R',   mins: [27, 33], rings: [5,  8],  organ_bg: 'Жлъчен мехур',               system_bg: 'Детоксикация' },
  { id: 'R-urogen',         side: 'R',   mins: [28, 33], rings: [7,  9],  organ_bg: 'Простата / Матка',           system_bg: 'Урогенитална' },
  { id: 'R-colon-asc',      side: 'R',   mins: [28, 40], rings: [4,  9],  organ_bg: 'Дебело черво (възходящо)',   system_bg: 'Храносмилателна' },
  { id: 'R-pancreas',       side: 'R',   mins: [35, 42], rings: [4,  7],  organ_bg: 'Панкреас',                   system_bg: 'Храносмилателна' },
  { id: 'R-kidney',         side: 'R',   mins: [38, 48], rings: [4,  9],  organ_bg: 'Бъбрек (десен)',             system_bg: 'Отделителна' },
  { id: 'R-adrenal',        side: 'R',   mins: [40, 46], rings: [5,  8],  organ_bg: 'Надбъбречна жлеза (дясна)', system_bg: 'Ендокринна' },
  { id: 'R-hip',            side: 'R',   mins: [43, 50], rings: [4,  9],  organ_bg: 'Тазобедрена става (дясна)', system_bg: 'Опорно-двигателна' },
  { id: 'R-spine-thor',     side: 'R',   mins: [4,  10], rings: [8, 10],  organ_bg: 'Гръбначен стълб (гръден)',   system_bg: 'Опорно-двигателна' },
  { id: 'R-spine-lumb',     side: 'R',   mins: [25, 33], rings: [8, 10],  organ_bg: 'Гръбначен стълб (лумбален)', system_bg: 'Опорно-двигателна' },
  { id: 'L-brain-motor',    side: 'L',   mins: [57, 59], rings: [4,  9],  organ_bg: 'Мозък (моторни зони)',       system_bg: 'Нервна' },
  { id: 'L-brain-sens',     side: 'L',   mins: [0,  3],  rings: [4,  9],  organ_bg: 'Мозък (сетивни зони)',       system_bg: 'Нервна' },
  { id: 'L-sinus',          side: 'L',   mins: [54, 59], rings: [4,  7],  organ_bg: 'Синуси (леви)',              system_bg: 'Дихателна' },
  { id: 'L-larynx',         side: 'L',   mins: [50, 55], rings: [4,  5],  organ_bg: 'Ларинкс / Гърло',           system_bg: 'Дихателна' },
  { id: 'L-thyroid',        side: 'L',   mins: [47, 53], rings: [4,  6],  organ_bg: 'Щитовидна жлеза (лява)',    system_bg: 'Ендокринна' },
  { id: 'L-eye-ear',        side: 'L',   mins: [43, 50], rings: [4,  7],  organ_bg: 'Ухо / Очи',                 system_bg: 'Нервна' },
  { id: 'L-bronchi',        side: 'L',   mins: [41, 46], rings: [4,  6],  organ_bg: 'Бронхи (леви)',             system_bg: 'Дихателна' },
  { id: 'L-shoulder',       side: 'L',   mins: [38, 42], rings: [4,  9],  organ_bg: 'Рамо (ляво)',                system_bg: 'Опорно-двигателна' },
  { id: 'L-lung',           side: 'L',   mins: [20, 30], rings: [4,  7],  organ_bg: 'Бял дроб (ляв)',             system_bg: 'Дихателна' },
  { id: 'L-heart',          side: 'L',   mins: [23, 35], rings: [4,  8],  organ_bg: 'Сърце',                      system_bg: 'Сърдечно-съдова' },
  { id: 'L-bladder',        side: 'L',   mins: [30, 35], rings: [7,  9],  organ_bg: 'Пикочен мехур',             system_bg: 'Отделителна' },
  { id: 'L-spleen',         side: 'L',   mins: [28, 35], rings: [5,  9],  organ_bg: 'Далак',                      system_bg: 'Имунна' },
  { id: 'L-urogen',         side: 'L',   mins: [27, 32], rings: [7,  9],  organ_bg: 'Простата / Матка',           system_bg: 'Урогенитална' },
  { id: 'L-colon-desc',     side: 'L',   mins: [28, 40], rings: [4,  9],  organ_bg: 'Дебело черво (низходящо)',   system_bg: 'Храносмилателна' },
  { id: 'L-pancreas',       side: 'L',   mins: [18, 25], rings: [4,  7],  organ_bg: 'Панкреас',                   system_bg: 'Храносмилателна' },
  { id: 'L-kidney',         side: 'L',   mins: [12, 22], rings: [4,  9],  organ_bg: 'Бъбрек (ляв)',               system_bg: 'Отделителна' },
  { id: 'L-adrenal',        side: 'L',   mins: [14, 20], rings: [5,  8],  organ_bg: 'Надбъбречна жлеза (лява)',  system_bg: 'Ендокринна' },
  { id: 'L-hip',            side: 'L',   mins: [10, 17], rings: [4,  9],  organ_bg: 'Тазобедрена става (лява)',  system_bg: 'Опорно-двигателна' },
  { id: 'L-spine-thor',     side: 'L',   mins: [50, 56], rings: [8, 10],  organ_bg: 'Гръбначен стълб (гръден)',   system_bg: 'Опорно-двигателна' },
  { id: 'L-spine-lumb',     side: 'L',   mins: [27, 35], rings: [8, 10],  organ_bg: 'Гръбначен стълб (лумбален)', system_bg: 'Опорно-двигателна' },
];

// Zones with direct relevance to metabolism / endocrine function / digestion
// — the requested analytical focus. These get deeper scrutiny in detection
// and heavier weight in the report; everything else in MAP_V9 still exists
// and is still detected/reported, just without the extra emphasis.
const PRIORITY_ZONE_IDS = new Set([
  'ANY-stomach', 'ANY-sm-intest', 'ANY-ANW', 'ANY-LYM',
  'R-thyroid', 'R-liver', 'R-gallbladder', 'R-colon-asc', 'R-pancreas', 'R-kidney', 'R-adrenal',
  'L-thyroid', 'L-colon-desc', 'L-pancreas', 'L-kidney', 'L-adrenal',
]);
MAP_V9.forEach(z => { z.priority = PRIORITY_ZONE_IDS.has(z.id); });

// =====================================================================
// ZONE MATCHING (deterministic, in code — never asked of the model)
// =====================================================================
// Matching a finding to a zone is interval arithmetic, so it belongs here rather
// than in a prompt. Just as important: this map is heavily overlapping — for the
// right eye, 314 of 720 (minute x ring) cells are claimed by two or more zones and
// some by six. Any single "winning" organ is therefore partly an artefact of the
// tie-break rule, so matching reports the runners-up and flags ambiguity instead
// of presenting one confident organ name.

function rangeOverlap(a, b) {
  return Math.min(a[1], b[1]) - Math.max(a[0], b[0]);
}

function scoreZone(minuteRange, ringRange, zone) {
  const mOv = rangeOverlap(minuteRange, zone.mins);
  const rOv = rangeOverlap(ringRange, zone.rings);
  if (mOv < 0 || rOv < 0) return null;

  const zoneArea = (zone.mins[1] - zone.mins[0] + 1) * (zone.rings[1] - zone.rings[0] + 1);
  const overlapArea = (mOv + 1) * (rOv + 1);
  // Reward real overlap, prefer the more specific (smaller) zone, and prefer a
  // side-specific zone over an "ANY" catch-all covering the whole iris.
  const specificity = 1 / Math.sqrt(zoneArea);
  const sideBonus = zone.side === 'ANY' ? 0.85 : 1.0;
  return { score: overlapArea * specificity * sideBonus, overlapArea, zoneArea };
}

/**
 * Rank the zones a finding could belong to.
 * Returns { primary, alternates, ambiguous, systemLabel } — ambiguous when the
 * runner-up is nearly as good a fit, which on this map is common.
 */
function matchZones(minuteRange, ringRange, side) {
  const scored = [];
  for (const zone of MAP_V9) {
    if (zone.side !== side && zone.side !== 'ANY') continue;
    const s = scoreZone(minuteRange, ringRange, zone);
    if (s) scored.push({ zone, ...s });
  }
  if (!scored.length) {
    return { primary: null, alternates: [], ambiguous: false, systemLabel: null };
  }

  scored.sort((a, b) => (b.score - a.score) || (a.zoneArea - b.zoneArea));
  const top = scored[0];
  const runnerUp = scored[1];
  const ambiguous = !!runnerUp && runnerUp.score >= 0.75 * top.score;

  return {
    primary: {
      id: top.zone.id,
      organ_bg: top.zone.organ_bg,
      system_bg: top.zone.system_bg,
      priority: !!top.zone.priority,
    },
    alternates: scored.slice(1, 4).map(s => ({
      id: s.zone.id, organ_bg: s.zone.organ_bg, system_bg: s.zone.system_bg,
    })),
    ambiguous,
    // What we can state without overclaiming when several organs fit equally.
    systemLabel: top.zone.system_bg,
  };
}

/** Attach zone matches to every finding of a verified detection payload. */
function attachZones(json, side) {
  if (!json || json.error) return json;
  const withZone = (f) => {
    if (!f || !f.minuteRange || !f.ringRange) return f;
    const m = matchZones(f.minuteRange, f.ringRange, side);
    return { ...f, zone: m.primary, zoneAlternates: m.alternates, zoneAmbiguous: m.ambiguous };
  };
  const out = { ...json };
  if (Array.isArray(out.verified_structural)) out.verified_structural = out.verified_structural.map(withZone);
  if (Array.isArray(out.verified_pigment)) out.verified_pigment = out.verified_pigment.map(withZone);

  // Rebuild the zone summary from the matches rather than trusting the model's.
  const counts = new Map();
  for (const f of [...(out.verified_structural || []), ...(out.verified_pigment || [])]) {
    if (!f.zone) continue;
    const key = f.zone.id;
    const entry = counts.get(key) || {
      zoneId: key, organ_bg: f.zone.organ_bg, system_bg: f.zone.system_bg,
      priority: f.zone.priority, evidenceCount: 0, ambiguousCount: 0, types: {},
    };
    entry.evidenceCount++;
    if (f.zoneAmbiguous) entry.ambiguousCount++;
    entry.types[f.type] = (entry.types[f.type] || 0) + 1;
    counts.set(key, entry);
  }
  out.zoneSummary = Array.from(counts.values())
    .map(e => ({
      zoneId: e.zoneId, organ_bg: e.organ_bg, system_bg: e.system_bg,
      priority: e.priority, evidenceCount: e.evidenceCount,
      ambiguousCount: e.ambiguousCount,
      topTypes: Object.entries(e.types).map(([t, c]) => `${t}:${c}`),
    }))
    .sort((a, b) => b.evidenceCount - a.evidenceCount);

  return out;
}

// =====================================================================
// CALL 1: FULL DETECTION (vision + image)
// Combines: STEP1 geo + STEP2A structural + STEP2B pigment + STEP2B_ANW collarette
// =====================================================================
function priorityZoneHint(side) {
  const zones = MAP_V9.filter(z => z.priority && (z.side === side || z.side === 'ANY'));
  const toSectors = (mins) => [Math.floor(mins[0] / 5) + 1, Math.ceil(mins[1] / 5)];
  return zones.map(z => {
    const [s0, s1] = toSectors(z.mins);
    return `  - ${z.organ_bg} (${z.system_bg}): around S${s0}-S${s1}`;
  }).join('\n');
}

function promptCall1_Detect(side, imageHash) {
  return `${IMAGE_FORMAT}

================================================================================
CALL1: COMPREHENSIVE IRIS DETECTION — Geo + Structural + Pigment + Collarette
================================================================================

ROLE: expert_iridologist_detector_v11
MODE: image_parse_only
INPUT: unwrapped_iris_strip (rectangular image as described above)
SIDE: ${side}
IMG_ID: ${imageHash}

You are performing a THOROUGH first-pass detection of ALL iris features.
Take your time — precision is critical. This is the foundation for the entire analysis.

================================================================================
PART A: GEO CALIBRATION & QUALITY CHECK
================================================================================

1. Verify the strip is a valid unwrapped iris with visible grid labels.
2. Assess image quality:
   - focus: good | med | poor (is iris fiber texture clearly visible?)
   - glare: none | low | med | high (bright washed-out patches?)
   - occlusion: none | low | med | high (white/blank eyelid bands?)
3. Identify eyelid-masked (white/blank) regions → mark as invalidRegions.
4. Identify specular/glare patches → mark as invalidRegions.
5. Quality score 0-100 based on focus clarity, glare extent, occlusion area.
6. usableUpperIris: true if minutes [57-59] and [0-3] are NOT mostly white/blank.
7. refRay15Usable: true if minute-15 column is visible and free of masking.

QUALITY GATE — if ANY of these is true, return error:
- focus="poor" throughout
- >35% of strip is white/blank
- Strip is not a valid iris strip
→ return: {"error":{"stage":"CALL1","code":"LOW_QUALITY","message":"<reason>","canRetry":true}}

================================================================================
PRIORITY FOCUS: METABOLISM / ENDOCRINE / DIGESTION
================================================================================
This analysis feeds a nutrition plan. Give EXTRA scrutiny to sectors overlapping
these organs — look twice, use the full ORG_IN/ORG_MID/ORG_OUT height to tell
them apart, and don't merge distinct findings there for convenience:
${priorityZoneHint(side)}
All other zones on the strip are still detected and reported normally, just
without this extra pass.

================================================================================
PART B: STRUCTURAL DETECTION
================================================================================

Detect STRUCTURAL findings only (NO organ names, NO diagnosis):
  lacuna | crypt | giant_lacuna | atrophic_area | collarette_defect_lesion |
  radial_furrow | deep_radial_cleft | transversal_fiber | structural_asymmetry

For each: type, sectorRange [firstSector, lastSector] (1-12, read from the
          printed S1..S12 labels), ringGroup (read from the printed left-margin
          band name: IPB|STOM|ANW|ORG_IN|ORG_MID|ORG_OUT|LYM|SCU), ringGroupEnd
          (only if it visibly crosses into another band), size (xs/s/m/l),
          notes (<=60 chars), confidence (0.0-1.0).

IGNORE: White/blank eyelid-masked bands, glare/specular patches.

DEFINITIONS (how features appear in the UNWRAPPED STRIP):
- lacuna: horizontally elongated oval gap breaking fiber flow, lighter interior
- crypt: small deep dark triangular/rhomboid hole with sharp edges
- giant_lacuna: very large lacuna spanning 2+ sectors
- atrophic_area: absent/flattened fiber texture (NOT glare, NOT white band — dull, texture-free)
- collarette_defect_lesion: notch/break on the ANW band
- radial_furrow: narrow VERTICAL dark stripe running from ANW outward, within 1 sector wide
- deep_radial_cleft: wider VERTICAL dark stripe (spans part of 2+ sectors), deeper than furrow
- transversal_fiber: DIAGONAL line crossing radial fibers at an angle
- structural_asymmetry: visible fiber density/texture difference between strip halves

RANGE RULES:
- sectorRange NEVER wraps past S12→S1: split a straddling finding into two
  (one ending at S12, one starting at S1).
- Point-like findings still get a sectorRange of at least 1 sector wide (e.g. [4,4]).
- Findings spanning 4+ sectors: confidence -= 0.15; drop if <0.55 (likely overbroad).

================================================================================
PART C: PIGMENT & RING DETECTION
================================================================================

Detect pigment/ring features:
  pigment_spot (subtype: orange_rust|brown_black|yellow|other) |
  pigment_cloud | pigment_band | brushfield_like_spots |
  nerve_rings | lymphatic_rosary | scurf_rim | sodium_ring

DEFINITIONS:
- pigment_spot: bounded colored spot on fibers, no structural gap beneath
- pigment_cloud: diffuse haze with soft gradual edges, spanning multiple min/rings
- pigment_band: HORIZONTAL colored stripe running left-to-right, 1-2 ring rows
- brushfield_like_spots: scattered pale tiny dots in R9-R11
- nerve_rings: HORIZONTAL arc(s) running full strip width, structural stress lines
- lymphatic_rosary: chain of discrete pale nodules in R10 area (not a continuous band)
- scurf_rim: dark HORIZONTAL band at R11 (bottom edge of strip content)
- sodium_ring: pale/milky HORIZONTAL band near R9-R11

Assess GLOBAL TRIAD:
- constitution: LYM | HEM | BIL | unclear
- disposition: SILK | LINEN | BURLAP | unclear
- diathesis_tags: HAC | LRS | LIP | DYS (each with confidence 0-1)

================================================================================
PART D: ANW / COLLARETTE CONTOUR PROFILING
================================================================================

The collarette (ANW) is the wavy band with its own labeled row on the strip.
Divide into the same 12 sectors as everywhere else (seg = sector number, 1-12).

For EACH visible segment:
- position: high (near the ANW/ORG_IN boundary, contracted) |
            mid (centered in the ANW band, normal) |
            low (near the STOM/ANW boundary, expanded)
  (read this directly from where the band sits relative to its bold boundary
  lines — do not estimate a decimal ring number)
- shape: normal | expanded | contracted | broken | notched | ballooning | flattened
- thickness: thin | normal | thick (relative to the band's own printed height)
- integrity: sharp | fuzzy | absent
- If segment is masked (white/eyelid): visible=false

ANW_status overall: expanded | contracted | broken | normal | mixed | unclear

List ANW defects: breaks, notches, ballooning with sectorRange + notes
(ringGroup is always "ANW" for these).

contourSummary: expandedSegments[], contractedSegments[], brokenSegments[]
(seg numbers), overallIntegrity (good|moderate|poor).

================================================================================
OUTPUT — JSON ONLY — EXACT STRUCTURE:
================================================================================

{
  "imgId": "${imageHash}",
  "side": "${side}",
  "quality": {
    "ok": true,
    "score0_100": 0,
    "focus": "good|med|poor",
    "glare": "none|low|med|high",
    "occlusion": "none|low|med|high",
    "usableUpperIris": true,
    "refRay15Usable": true,
    "invalidRegions": [
      {"type":"specular|eyelid_band","sectorRange":[0,0],"ringGroup":"IPB|STOM|ANW|ORG_IN|ORG_MID|ORG_OUT|LYM|SCU"}
    ]
  },
  "structural": [
    {"type":"...","sectorRange":[0,0],"ringGroup":"...","ringGroupEnd":null,"size":"xs|s|m|l","notes":"<=60","confidence":0.0}
  ],
  "pigment": [
    {"type":"...","subtype":"...","sectorRange":[0,0],"ringGroup":"...","ringGroupEnd":null,"severity":"low|medium|high","notes":"<=60","confidence":0.0}
  ],
  "global": {
    "constitution": "LYM|HEM|BIL|unclear",
    "disposition": "SILK|LINEN|BURLAP|unclear",
    "diathesis": [{"code":"HAC|LRS|LIP|DYS","confidence":0.0}]
  },
  "collarette": {
    "ANW_status": "expanded|contracted|broken|normal|mixed|unclear",
    "confidence": 0.0,
    "segments": [
      {"seg":1,"visible":true,"position":"high|mid|low","shape":"normal|expanded|contracted|broken|notched|ballooning|flattened","thickness":"thin|normal|thick","integrity":"sharp|fuzzy|absent","confidence":0.0}
    ],
    "defects": [
      {"type":"break|notch|ballooning|lesion|pigment_on_ANW","sectorRange":[0,0],"notes":"<=60","confidence":0.0}
    ],
    "contourSummary": {
      "expandedSegments": [],
      "contractedSegments": [],
      "brokenSegments": [],
      "overallIntegrity": "good|moderate|poor"
    }
  }
}

FAILSAFE:
{"error":{"stage":"CALL1","code":"LOW_QUALITY|INVALID_STRIP|FORMAT_FAIL","message":"<reason>","canRetry":true}}`;
}

// =====================================================================
// CALL 2: VERIFICATION & ZONE MAPPING (vision + image)
// Combines: STEP2C consistency + STEP3 zone mapper + STEP4 profile builder
// Re-examines image to verify CALL1 findings
// =====================================================================
function promptCall2_Verify(side, imageHash, call1) {
  return `${IMAGE_FORMAT}

================================================================================
CALL2: VERIFICATION, CONSISTENCY & ZONE MAPPING — Re-examine with Known Findings
================================================================================

ROLE: expert_iridologist_verifier_v11
MODE: image_parse_only + data_integration
INPUT: unwrapped_iris_strip (same image as CALL1) + CALL1 detection results
SIDE: ${side}
IMG_ID: ${imageHash}

CALL1_RESULTS: ${JSON.stringify(call1)}

You have the INITIAL DETECTION results from CALL1. Now RE-EXAMINE the actual image
to VERIFY, REFINE, and CORRECT. This is your chance to catch false positives,
find missed features, and ensure precision.

================================================================================
PART A: VERIFICATION — Re-examine image against CALL1 findings
================================================================================

Note: CALL1_RESULTS below already has canonical minuteRange/ringRange filled
in (computed deterministically from CALL1's sector/ringGroup picks) — use
those for zone-matching in Part C. For your own re-classification here, keep
reading sectorRange/ringGroup from the printed labels, not pixel positions.

For EACH structural finding from CALL1:
1. Look at the stated sector(s) + ring group in the actual image.
2. CONFIRM the finding exists (keep with same or adjusted confidence).
3. CORRECT sectorRange/ringGroup if they seem off after careful re-examination.
4. REJECT if the area is actually white/blank (eyelid mask), glare, or normal tissue.
   Move rejected findings to "dropped" with reason.

For EACH pigment finding from CALL1:
1. Re-examine the actual image area.
2. CONFIRM, CORRECT sectorRange/ringGroup, or REJECT with reason.

CHECK FOR MISSED FINDINGS:
- Carefully scan the entire strip for features that CALL1 may have missed.
- Give the priority sectors listed in CALL1's prompt (metabolism/endocrine/
  digestion organs) a second dedicated look before finalizing.
- Add any newly detected findings to the verified lists.

================================================================================
PART B: CONSISTENCY RULES (apply to verified findings)
================================================================================

CONTRADICTION RULES:
1) scurf_rim vs sodium_ring: same area → keep sodium_ring if light/milky; keep scurf_rim if dark.
2) pigment_spot vs lacuna/crypt: overlapping → keep structural; drop pigment.
3) lymphatic_rosary vs brushfield_like_spots: same area → keep rosary if chain/arc of discrete nodules.
4) Specular contamination: drop findings overlapping invalidRegions >25%.
5) collarette_defect_lesion vs ANW defects: same sector(s) → merge, keep ANW defect detail.

DEDUP/MERGE: Same type + same sector(s) + same ringGroup → merge (union sectorRange, max confidence).

RANGE NORMALIZATION:
- Clamp sectors 1..12, ringGroup to one of the 8 named bands.
- sectorRange NEVER wraps past S12→S1: split if it would.
- usableUpperIris=false: findings in S12 or S1 → confidence -= 0.10.

COLLARETTE:
- collarette ringGroup is always "ANW". Cross-check ANW status: if structural
  defects contradict, note the discrepancy.

================================================================================
PART C: (NOT YOUR JOB) — ZONE MAPPING IS DONE IN CODE
================================================================================

Do NOT map findings to organs, and do NOT output any organ or zone name here.
Anatomical mapping is pure interval arithmetic and is performed deterministically
after this call, from the sector/ring-group labels you report. Your only job is
to describe WHAT you see and WHERE it sits on the printed grid, accurately.

Reporting an organ name here would mean re-deriving numeric coordinates by eye,
which is exactly the error this pipeline is designed to avoid.

================================================================================
PART D: PROFILE BUILD — Derive health axes and channels
================================================================================

From the verified findings, compute (WITHOUT naming organs — see Part C):
1. Constitution/disposition/diathesis from global traits.
2. ANW profile from collarette segments.
3. Ring-group load: for each ring group with findings, how loaded it looks
   (normal | attention | concern) plus the finding ids that justify it.
4. Axes: stress (0-100), digestive (0-100), immune (0-100).

================================================================================
OUTPUT — JSON ONLY — EXACT STRUCTURE:
================================================================================

{
  "imgId": "${imageHash}",
  "side": "${side}",
  "verified_structural": [
    {"fid":"S1","type":"...","sectorRange":[0,0],"ringGroup":"...","ringGroupEnd":null,"size":"xs|s|m|l","notes":"<=60","confidence":0.0,"status":"confirmed|corrected|new"}
  ],
  "verified_pigment": [
    {"fid":"P1","type":"...","subtype":"...","sectorRange":[0,0],"ringGroup":"...","ringGroupEnd":null,"severity":"low|medium|high","notes":"<=60","confidence":0.0,"status":"confirmed|corrected|new"}
  ],
  "collarette_verified": {
    "ANW_status": "expanded|contracted|broken|normal|mixed|unclear",
    "confidence": 0.0,
    "segments": [
      {"seg":1,"visible":true,"position":"high|mid|low","shape":"normal|expanded|contracted|broken|notched|ballooning|flattened","thickness":"thin|normal|thick","integrity":"sharp|fuzzy|absent","confidence":0.0}
    ],
    "defects": [
      {"type":"break|notch|ballooning|lesion","sectorRange":[0,0],"notes":"<=60","confidence":0.0}
    ],
    "contourSummary": {
      "expandedSegments":[],"contractedSegments":[],"brokenSegments":[],
      "overallIntegrity":"good|moderate|poor"
    }
  },
  "global_verified": {
    "constitution": "LYM|HEM|BIL|unclear",
    "disposition": "SILK|LINEN|BURLAP|unclear",
    "diathesis": [{"code":"HAC|LRS|LIP|DYS","confidence":0.0}],
    "ANW_status": "expanded|contracted|broken|normal|mixed|unclear"
  },
  "profile": {
    "axesScore": {"stress0_100":0,"digestive0_100":0,"immune0_100":0},
    "ringGroupLoad": [
      {"ringGroup":"STOM|ANW|ORG_IN|ORG_MID|ORG_OUT|LYM|SCU","status":"normal|attention|concern","evidence":[{"fid":"S1"}]}
    ],
    "ANW_profile": {
      "overallIntegrity":"good|moderate|poor",
      "expandedSectors":"...",
      "contractedSectors":"...",
      "brokenSectors":"...",
      "clinicalNote":"<=120 chars"
    }
  },
  "dropped": [
    {"type":"...","sectorRange":[0,0],"reason":"contradiction|specular|eyelid_band|too_wide|low_confidence|duplicate|false_positive"}
  ],
  "warnings": ["<=60 chars"]
}

FAILSAFE:
{"error":{"stage":"CALL2","code":"PREREQ_FAIL|FORMAT_FAIL","message":"<reason>","canRetry":true}}`;
}

// =====================================================================
// CALL 3: REPORT GENERATION (text only — no image)
// Combines: STEP5 Bulgarian report
// =====================================================================
function promptCall3_Report(side, imageHash, call1, call2, questionnaire, captureQuality) {
  const q = questionnaire || {};
  return `IRIS PIPELINE — CALL3: Bulgarian Report Generation (v11)

ROLE: iris_frontend_report_generator_bg_v11
MODE: strict_json_only (NO image — text synthesis only)

INPUTS:
  DETECTION = ${JSON.stringify(call1)}
  VERIFIED  = ${JSON.stringify(call2)}
  QUESTIONNAIRE = ${JSON.stringify(q)}
  CAPTURE_QUALITY = ${JSON.stringify(captureQuality || {})}
  SIDE = ${side}
  IMG_ID = ${imageHash}

PREREQ: If VERIFIED.error exists → return error JSON.

You have VERIFIED detection results (CALL2 output). Synthesize them into the
final Bulgarian-language UI report. This is a TEXT-ONLY call — no image analysis.

================================================================================
NOT MEDICAL — MANDATORY FRAMING:
- This is a wellness-style, non-diagnostic iris reading used only to bias a
  nutrition plan. It is NOT a medical diagnosis and iridology organ-mapping
  is not scientifically validated. Every output MUST include the disclaimer
  field below verbatim (translated meaning, Bulgarian wording may vary
  slightly but must state clearly: not a medical diagnosis, informational/
  wellness use only, consult a doctor for real symptoms).

ROLE OF THE IRIS IN THIS REPORT (read carefully — this drives everything):
- The iris findings are a SUPPORTING signal, not the basis of the advice. The
  nutrition plan is built primarily from the QUESTIONNAIRE, which is real
  self-reported data. Iris findings only shift emphasis between options that
  are already appropriate for that person.
- Never invent a restriction, a deficiency, or a condition from the iris alone.
- If QUESTIONNAIRE is sparse, say so in dataQuality and keep the plan
  correspondingly general — do NOT compensate by leaning harder on the iris.

CORE TRUTH RULE:
- ORGAN and SYSTEM names MUST come ONLY from the zone fields already attached
  to each finding (zone.organ_bg / zone.system_bg). These were computed in code.
  Do NOT infer an organ from coordinates yourself and do NOT introduce any organ
  name that is not present in VERIFIED.
- The 12 UI zones are DISPLAY BUCKETS ONLY and correspond 1:1 to sectors
  S1..S12 already used throughout detection (Zone N = Sector N).

AMBIGUITY IS EXPLICIT — RESPECT IT:
- Each finding carries zoneAmbiguous and zoneAlternates. The underlying map
  overlaps heavily: many positions are claimed by several organs at once, so a
  single organ name is often partly an artefact of the tie-break.
- If zoneAmbiguous is true → use the general system label (zone.system_bg), NOT
  the specific organ name. You may mention the alternatives together in the
  findings text (e.g. "черен дроб или жлъчен мехур"), never as a single verdict.
- If zoneAmbiguous is false AND zone.priority is true (metabolism / endocrine /
  digestion — the requested focus) → use the SPECIFIC organ name.
- If zoneAmbiguous is false but zone.priority is false → prefer the system
  label; a specific organ is allowed only when evidence is strong (confidence
  >= 0.75).

CAPTURE QUALITY GOVERNS CONFIDENCE:
- CAPTURE_QUALITY carries how good the photo actually was (score 0-100,
  visibleFraction, and rollCorrected — whether an anatomical rotation reference
  could be established from the eye corners).
- If rollCorrected is false, the angular position of every finding may be off by
  up to a sector. In that case: prefer system labels over organ names throughout,
  and say so in dataQuality.
- If score < 55 or visibleFraction < 0.7, keep zone statuses conservative
  (avoid "concern" unless the questionnaire independently supports it).

VALIDATION PRIORITY (cross-reference with QUESTIONNAIRE):
  1) ВИСОК — потвърдено от въпросника (highest weight)
  2) СРЕДЕН — не е споменато (preventive, medium weight)
  3) НИСЪК — противоречи (flag it, do not emphasize)

MINUTE-TO-ZONE CONVERSION:
  Zone  1 ("12-1ч"):  minutes  0-5   → degrees 0-30
  Zone  2 ("1-2ч"):   minutes  5-10  → degrees 30-60
  Zone  3 ("2-3ч"):   minutes 10-15  → degrees 60-90
  Zone  4 ("3-4ч"):   minutes 15-20  → degrees 90-120
  Zone  5 ("4-5ч"):   minutes 20-25  → degrees 120-150
  Zone  6 ("5-6ч"):   minutes 25-30  → degrees 150-180
  Zone  7 ("6-7ч"):   minutes 30-35  → degrees 180-210
  Zone  8 ("7-8ч"):   minutes 35-40  → degrees 210-240
  Zone  9 ("8-9ч"):   minutes 40-45  → degrees 240-270
  Zone 10 ("9-10ч"):  minutes 45-50  → degrees 270-300
  Zone 11 ("10-11ч"): minutes 50-55  → degrees 300-330
  Zone 12 ("11-12ч"): minutes 55-60  → degrees 330-360

HOW TO FILL EACH ZONE:
- Collect VERIFIED findings whose center minute falls in the zone's range.
- Determine dominant organ/system by: weight = confidence + severity_bonus(lacuna=+0.1, crypt=+0.15, cleft=+0.15)
- status: concern (strong evidence + HIGH questionnaire) | attention (medium evidence) | normal (little/no evidence)
- findings: brief Bulgarian summary ≤ 60 chars

ARTIFACTS (2-5 strongest from VERIFIED):
  Prioritize: lacuna, crypt, radial_furrow, deep_radial_cleft, nerve_rings,
              pigment_spot, sodium_ring, scurf_rim, lymphatic_rosary, ANW defects
  location format: clock string (minute 0→"12:00", 5→"1:00", 10→"2:00", 15→"3:00", etc.)

COLLARETTE PROFILE (from VERIFIED.collarette_verified + VERIFIED.profile.ANW_profile):
  - status: Bulgarian label
  - integrity: Bulgarian
  - 12 segments with shape/position in Bulgarian
  - defects with clock location
  - clinicalNote ≤ 120 chars Bulgarian

SYSTEM SCORES (always exactly 6):
  Храносмилателна | Имунна | Нервна | Сърдечно-съдова | Детоксикация | Ендокринна
  score 0-100 based on VERIFIED.profile.axesScore + zone evidence
  description ≤ 60 chars Bulgarian
  Give Храносмилателна/Детоксикация/Ендокринна the most detailed, evidence-
  specific descriptions since those map to the priority focus area; the
  other three can stay more general.

NUTRITION PLAN — THIS IS THE MAIN DELIVERABLE:
  Build a concrete, usable daily plan, not a list of slogans.
  - Source of truth order: (1) QUESTIONNAIRE — goals, complaints, dietary
    habits, allergies, medications, activity, sleep, age/sex/BMI; (2) general
    sound nutrition practice; (3) iris priority-zone findings, for emphasis only.
  - dayPlan: 4-5 entries (Закуска, Обяд, Следобедна закуска, Вечеря, по избор
    Преди сън). Each: meal (Bulgarian label), suggestion (concrete foods, a real
    portion idea, ≤140 chars), rationale (≤100 chars, WHY for THIS person).
  - emphasize / reduce: 3-6 each. Each has item (≤40) and reason (≤100). The
    reason must reference the questionnaire where possible; only cite an iris
    finding when it genuinely adds something, and then name the SYSTEM, not a
    specific organ, unless that finding was unambiguous and priority.
  - hydration: one concrete line (≤120).
  - weeklyHabits: 2-4 small, checkable habits (≤100 each).
  - cautions: allergies, medications, and anything the questionnaire flags that
    should override a generic suggestion. If allergies/medications are present in
    QUESTIONNAIRE they MUST be reflected here. Empty array only if truly nothing.
  - notForYou: 1-3 explicit statements of what this plan does NOT address and
    when to see a professional (≤120 each).

DATA QUALITY (honest self-assessment, shown to the user):
  - questionnaireCompleteness: "пълен|частичен|минимален"
  - irisContribution: "съществен|поддържащ|ограничен" — how much the iris
    actually shaped the plan. With poor capture quality or an unreliable
    rotation reference this must be "ограничен".
  - limitations: 1-3 short Bulgarian sentences naming the real limits of this
    particular analysis (sparse questionnaire, occluded iris, no roll reference,
    ambiguous zone attribution).

ADVICE (Bulgarian, all values ≤ 120 chars):
  priorities: 3-6 bullets | lifestyle.sleep: 2-4 | lifestyle.stress: 2-4
  lifestyle.activity: 2-4 | followUp: 2-5 bullets

================================================================================
OUTPUT — JSON ONLY — EXACT STRUCTURE:
================================================================================

{
  "disclaimer": "Този анализ не е медицинска диагноза - ориентировъчно, уелнес приложение с цел хранителен план. При реални оплаквания се консултирайте с лекар.",
  "analysis": {
    "zones": [
      {"id":1,"name":"12-1ч","organ":"<БГ>","status":"normal|attention|concern","findings":"<=60 БГ","angle":[0,30]},
      {"id":2,"name":"1-2ч","organ":"...","status":"...","findings":"<=60","angle":[30,60]},
      {"id":3,"name":"2-3ч","organ":"...","status":"...","findings":"<=60","angle":[60,90]},
      {"id":4,"name":"3-4ч","organ":"...","status":"...","findings":"<=60","angle":[90,120]},
      {"id":5,"name":"4-5ч","organ":"...","status":"...","findings":"<=60","angle":[120,150]},
      {"id":6,"name":"5-6ч","organ":"...","status":"...","findings":"<=60","angle":[150,180]},
      {"id":7,"name":"6-7ч","organ":"...","status":"...","findings":"<=60","angle":[180,210]},
      {"id":8,"name":"7-8ч","organ":"...","status":"...","findings":"<=60","angle":[210,240]},
      {"id":9,"name":"8-9ч","organ":"...","status":"...","findings":"<=60","angle":[240,270]},
      {"id":10,"name":"9-10ч","organ":"...","status":"...","findings":"<=60","angle":[270,300]},
      {"id":11,"name":"10-11ч","organ":"...","status":"...","findings":"<=60","angle":[300,330]},
      {"id":12,"name":"11-12ч","organ":"...","status":"...","findings":"<=60","angle":[330,360]}
    ],
    "artifacts": [
      {"type":"тип_БГ","location":"3:00-4:00","description":"<=60 БГ","severity":"low|medium|high"}
    ],
    "collaretteProfile": {
      "status": "разширена|свита|прекъсната|нормална|смесена|неясна",
      "integrity": "добра|умерена|слаба",
      "segments": [
        {"seg":1,"clock":"12-1ч","shape":"нормална|разширена|свита|прекъсната|вдлъбната|балониране|заличена","position":"висока|средна|ниска","visible":true}
      ],
      "defects": [
        {"type":"прекъсване|вдлъбнатина|балониране|лезия","location":"3:00-4:00","description":"<=60"}
      ],
      "clinicalNote": "<=120 chars кратка интерпретация"
    },
    "overallHealth": 75,
    "systemScores": [
      {"system":"Храносмилателна","score":80,"description":"<=60"},
      {"system":"Имунна","score":80,"description":"<=60"},
      {"system":"Нервна","score":80,"description":"<=60"},
      {"system":"Сърдечно-съдова","score":80,"description":"<=60"},
      {"system":"Детоксикация","score":80,"description":"<=60"},
      {"system":"Ендокринна","score":80,"description":"<=60"}
    ]
  },
  "nutritionPlan": {
    "summary": "<=200 chars БГ - какъв е подходът за този човек и защо",
    "dayPlan": [
      {"meal":"Закуска","suggestion":"<=140 конкретни храни и порция","rationale":"<=100 защо за този човек"}
    ],
    "emphasize": [{"item":"<=40","reason":"<=100"}],
    "reduce": [{"item":"<=40","reason":"<=100"}],
    "hydration": "<=120",
    "weeklyHabits": ["<=100"],
    "cautions": ["<=120 - алергии, лекарства, важни ограничения"],
    "notForYou": ["<=120 - какво този план НЕ покрива"]
  },
  "dataQuality": {
    "questionnaireCompleteness": "пълен|частичен|минимален",
    "irisContribution": "съществен|поддържащ|ограничен",
    "limitations": ["<=140 БГ"]
  },
  "advice": {
    "priorities": ["<=120 chars БГ"],
    "lifestyle": {"sleep": ["<=120"], "stress": ["<=120"], "activity": ["<=120"]},
    "followUp": ["<=120"]
  },
  "pipeline": {
    "quality": {
      "score0_100": 0,
      "focus": "good|med|poor",
      "glare": "none|low|med|high",
      "occlusion": "none|low|med|high"
    },
    "global": {
      "constitution": "LYM|HEM|BIL|unclear",
      "disposition": "SILK|LINEN|BURLAP|unclear",
      "diathesis": [{"code":"HAC|LRS|LIP|DYS","confidence":0.0}],
      "ANW_status": "expanded|contracted|broken|normal|mixed|unclear"
    },
    "structuralFindings": [
      {"fid":"S1","type":"...","sectorRange":[0,0],"ringGroup":"...","size":"xs|s|m|l","zone":"zone_id","organ_bg":"...","ambiguous":false,"confidence":0.0}
    ],
    "pigmentFindings": [
      {"fid":"P1","type":"...","subtype":"...","sectorRange":[0,0],"ringGroup":"...","severity":"low|medium|high","zone":"zone_id","organ_bg":"...","ambiguous":false,"confidence":0.0}
    ]
  }
}

RULES:
- JSON ONLY output, no markdown, no extra text
- All UI text in BULGARIAN
- No double quotes inside string values
- findings ≤ 60 chars, descriptions ≤ 60 chars, advice ≤ 120 chars
- severity: low|medium|high
- Always output exactly 12 zones, exactly 6 systemScores
- 2-5 artifacts (strongest findings)
- organ names MUST come from the zone MAP
- IGNORE white/blank eyelid-masked areas
- "disclaimer" field is REQUIRED and must state this is not a medical diagnosis
- "nutritionPlan" is the main deliverable and must be concrete and usable
- "dataQuality" must be honest, including when the analysis is weak
- Never state a specific organ for a finding whose zoneAmbiguous is true

FAILSAFE:
{"error":{"stage":"CALL3","code":"PREREQ_FAIL|FORMAT_FAIL","message":"<reason>","canRetry":true}}`;
}

// =====================================================================
// ADMIN PANEL ROUTES
// =====================================================================

function checkAdminAuth(request, env) {
  if (!env.ADMIN_SECRET) return true; // open if no secret configured
  const auth = request.headers.get('Authorization') || '';
  return auth === `Bearer ${env.ADMIN_SECRET}`;
}

async function handleAdmin(request, env, url) {
  if (!checkAdminAuth(request, env)) {
    return jsonResp({ error: 'Unauthorized — provide correct Authorization: Bearer <ADMIN_SECRET>' }, 401);
  }

  const path = url.pathname;
  const method = request.method;

  // GET /admin/health — extended worker health + config
  if (method === 'GET' && path === '/admin/health') {
    const kvConfig = await getKVConfig(env);
    const effectiveApiKey = kvConfig?.apiKey || env.AI_API_KEY;
    const effectiveProvider = kvConfig?.provider || env.AI_PROVIDER || 'gemini';
    const effectiveModel = kvConfig?.model || env.AI_MODEL || 'gemini-2.0-flash';
    return jsonResp({
      status: effectiveApiKey ? 'healthy' : 'degraded',
      version: 'v11.0-3call-precision',
      provider: effectiveProvider,
      model: effectiveModel,
      apiKeyConfigured: !!effectiveApiKey,
      apiKeySource: kvConfig?.apiKey ? 'kv' : (env.AI_API_KEY ? 'env' : 'none'),
      configSource: kvConfig ? 'kv' : 'env',
      adminSecretConfigured: !!env.ADMIN_SECRET,
      kvConfigured: !!env.iris_rag_kv,
      aiBaseUrl: kvConfig?.baseUrl || env.AI_BASE_URL || 'https://api.openai.com/v1',
      geminiApiUrl: kvConfig?.geminiApiUrl || env.GEMINI_API_URL || 'https://generativelanguage.googleapis.com/v1beta',
      pipelineSteps: 3,
      pipelineDescription: 'CALL1(vision:detect) → CALL2(vision:verify+map) → CALL3(text:report)',
      timestamp: new Date().toISOString(),
    });
  }

  // GET /admin/cache/list?prefix=&cursor= — list KV keys (paginated, max 100)
  if (method === 'GET' && path === '/admin/cache/list') {
    if (!env.iris_rag_kv) return jsonResp({ error: 'KV not configured' }, 503);
    const prefix = url.searchParams.get('prefix') || '';
    const cursor = url.searchParams.get('cursor') || undefined;
    const limit  = Math.min(Number(url.searchParams.get('limit') || 50), 100);
    const listOpts = { prefix, limit };
    if (cursor) listOpts.cursor = cursor;
    const list = await env.iris_rag_kv.list(listOpts).catch(err => ({ error: err?.message }));
    return jsonResp(list);
  }

  // GET /admin/cache/entry?key= — read one KV entry
  if (method === 'GET' && path === '/admin/cache/entry') {
    if (!env.iris_rag_kv) return jsonResp({ error: 'KV not configured' }, 503);
    const key = url.searchParams.get('key');
    if (!key) return jsonResp({ error: 'key query param required' }, 400);
    const value = await env.iris_rag_kv.get(key, 'json').catch(() => null);
    if (value === null) return jsonResp({ error: 'Entry not found' }, 404);
    return jsonResp({ key, value });
  }

  // DELETE /admin/cache/entry?key= — delete one KV entry
  if (method === 'DELETE' && path === '/admin/cache/entry') {
    if (!env.iris_rag_kv) return jsonResp({ error: 'KV not configured' }, 503);
    const key = url.searchParams.get('key');
    if (!key) return jsonResp({ error: 'key query param required' }, 400);
    await env.iris_rag_kv.delete(key).catch(err => { throw err; });
    return jsonResp({ deleted: true, key });
  }

  // DELETE /admin/cache/flush — delete all KV entries (iterative)
  if (method === 'DELETE' && path === '/admin/cache/flush') {
    if (!env.iris_rag_kv) return jsonResp({ error: 'KV not configured' }, 503);
    let cursor;
    let total = 0;
    do {
      const opts = { limit: 100 };
      if (cursor) opts.cursor = cursor;
      const list = await env.iris_rag_kv.list(opts);
      const keys = list.keys.map(k => k.name);
      await Promise.all(keys.map(k => env.iris_rag_kv.delete(k)));
      total += keys.length;
      cursor = list.list_complete ? undefined : list.cursor;
    } while (cursor);
    return jsonResp({ flushed: true, deletedCount: total });
  }

  // POST /admin/test-ai — test the AI provider with a simple text prompt
  if (method === 'POST' && path === '/admin/test-ai') {
    const body = await request.json().catch(() => ({}));
    const aiProvider = body.ai_provider || null;
    const aiModel    = body.ai_model || null;
    const prompt     = body.prompt || 'Reply with exactly: {"status":"ok","message":"AI is working"}';
    const kvConfig = await getKVConfig(env);
    const effectiveEnv = createEffectiveEnv(env, aiProvider, aiModel, kvConfig);
    const t0 = Date.now();
    try {
      const raw = await aiCall(effectiveEnv, prompt, null);
      return jsonResp({
        ok: true,
        provider: effectiveEnv.AI_PROVIDER,
        model: effectiveEnv.AI_MODEL,
        durationMs: Date.now() - t0,
        response: raw,
      });
    } catch (err) {
      return jsonResp({
        ok: false,
        provider: effectiveEnv.AI_PROVIDER,
        model: effectiveEnv.AI_MODEL,
        durationMs: Date.now() - t0,
        error: err?.message || String(err),
      }, 502);
    }
  }

  // GET /admin/models — list all available models (same as public /models)
  if (method === 'GET' && path === '/admin/models') {
    return handleGetModels(env);
  }

  // GET /admin/config — read effective AI configuration (KV override or env)
  if (method === 'GET' && path === '/admin/config') {
    const kvConfig = await getKVConfig(env);
    const hasKV = kvConfig !== null;
    return jsonResp({
      source: hasKV ? 'kv' : 'env',
      provider: kvConfig?.provider || env.AI_PROVIDER || 'gemini',
      model: kvConfig?.model || env.AI_MODEL || 'gemini-2.0-flash',
      apiKeyConfigured: !!(kvConfig?.apiKey || env.AI_API_KEY),
      apiKeySource: kvConfig?.apiKey ? 'kv' : (env.AI_API_KEY ? 'env' : 'none'),
      baseUrl: kvConfig?.baseUrl || env.AI_BASE_URL || 'https://api.openai.com/v1',
      geminiApiUrl: kvConfig?.geminiApiUrl || env.GEMINI_API_URL || 'https://generativelanguage.googleapis.com/v1beta',
    });
  }

  // POST /admin/config — save AI configuration to KV
  if (method === 'POST' && path === '/admin/config') {
    if (!env.iris_rag_kv) return jsonResp({ error: 'KV not configured — cannot store settings' }, 503);
    const body = await request.json().catch(() => ({}));
    const existing = await getKVConfig(env) || {};
    const newConfig = {
      provider: body.provider || existing.provider || env.AI_PROVIDER || 'gemini',
      model: body.model || existing.model || env.AI_MODEL || 'gemini-2.0-flash',
      baseUrl: body.baseUrl !== undefined ? body.baseUrl : (existing.baseUrl || env.AI_BASE_URL || 'https://api.openai.com/v1'),
      geminiApiUrl: body.geminiApiUrl !== undefined ? body.geminiApiUrl : (existing.geminiApiUrl || env.GEMINI_API_URL || 'https://generativelanguage.googleapis.com/v1beta'),
    };
    if (body.apiKey && body.apiKey.trim()) {
      newConfig.apiKey = body.apiKey.trim();
    } else if (existing.apiKey) {
      newConfig.apiKey = existing.apiKey;
    }
    await env.iris_rag_kv.put(CONFIG_KV_KEY, JSON.stringify(newConfig));
    return jsonResp({
      saved: true,
      provider: newConfig.provider,
      model: newConfig.model,
      apiKeyConfigured: !!newConfig.apiKey,
      baseUrl: newConfig.baseUrl,
      geminiApiUrl: newConfig.geminiApiUrl,
    });
  }

  // DELETE /admin/config — remove KV config override (revert to env vars)
  if (method === 'DELETE' && path === '/admin/config') {
    if (!env.iris_rag_kv) return jsonResp({ error: 'KV not configured' }, 503);
    await env.iris_rag_kv.delete(CONFIG_KV_KEY);
    return jsonResp({ deleted: true, message: 'KV config cleared — using env vars now' });
  }

  return jsonResp({ error: 'Admin route not found', path }, 404);
}

// =====================================================================
// KV CONFIG HELPERS
// =====================================================================
const CONFIG_KV_KEY = 'config:ai';

async function getKVConfig(env) {
  if (!env.iris_rag_kv) return null;
  try {
    return await env.iris_rag_kv.get(CONFIG_KV_KEY, 'json');
  } catch {
    return null;
  }
}

// =====================================================================
// HELPERS
// =====================================================================
function genId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
}

function safeParseJSON(str) {
  try { return JSON.parse(str); } catch { return null; }
}

function jsonResp(data, status = 200) {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: { ...CORS, 'Content-Type': 'application/json' },
  });
}
