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
 *   IRIS_KV         — KV namespace for caching results
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
    kvConfigured: !!env.IRIS_KV,
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
    cached = await env.IRIS_KV.get(cacheKey, 'json');
  } catch (kvErr) {
    // KV read failure is non-fatal; proceed to run the pipeline
    console.error('KV get error:', kvErr?.message || kvErr);
  }
  if (cached) {
    return jsonResp({ cached: true, imageHash, side, model: effectiveModel, result: cached });
  }

  // Run 3-call pipeline
  const pipeline = new IrisPipeline(effectiveEnv, stripB64, side, imageHash, questionnaire);
  const result = await pipeline.run();

  // Store in KV with 24-hour TTL (even errors, to avoid hammering AI on bad images)
  await env.IRIS_KV.put(cacheKey, JSON.stringify(result), { expirationTtl: 86400 }).catch(() => {});

  return jsonResp({ cached: false, imageHash, side, model: effectiveModel, result });
}

function createEffectiveEnv(env, aiProvider, aiModel, kvConfig) {
  return {
    IRIS_KV: env.IRIS_KV,
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
  const result = await env.IRIS_KV.get(cacheKey, 'json').catch(() => null);
  if (!result) {
    return jsonResp({ error: 'Result not found or expired (24h TTL)' }, 404);
  }
  return jsonResp({ cached: true, result });
}

// =====================================================================
// 3-CALL PIPELINE CLASS
// =====================================================================
class IrisPipeline {
  constructor(env, stripB64, side, imageHash, questionnaire) {
    this.env          = env;
    this.imageB64     = stripB64;
    this.imageDataUrl = `data:image/jpeg;base64,${stripB64}`;
    this.side         = side;
    this.imageHash    = imageHash;
    this.questionnaire = questionnaire;
  }

  async run() {
    // ── CALL 1: Full Detection (vision + image) ──────────────────────
    // Geo calibration + structural + pigment + ANW collarette — all in one vision call
    const call1 = await this.visionCall(promptCall1_Detect(this.side, this.imageHash));
    if (call1.error) {
      return { error: call1.error || call1, stage: 'CALL1_DETECT', imageHash: this.imageHash, side: this.side };
    }

    // ── CALL 2: Verification & Zone Mapping (vision + image) ─────────
    // Re-examine image with CALL1's findings: validate, refine, consistency check, zone mapping
    const call2 = await this.visionCall(promptCall2_Verify(this.side, this.imageHash, call1));
    if (call2.error) {
      return { error: call2.error || call2, stage: 'CALL2_VERIFY', imageHash: this.imageHash, side: this.side };
    }

    // ── CALL 3: Report Generation (text only — no image needed) ──────
    // Synthesize verified findings into Bulgarian UI format with advice
    const call3 = await this.textCall(
      promptCall3_Report(this.side, this.imageHash, call1, call2, this.questionnaire)
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
polar-to-rectangular transformation of the original circular iris image.

STRIP DIMENSIONS:
  Full canvas : ~1280 x 390 px  (includes labeled margins)
  Iris content: ~1200 x 300 px  (inside margins)
  Left margin (~60 px) : ring labels R0, R1, ..., R11
  Top margin  (~50 px) : minute tick marks 0, 5, 10, ..., 60

COORDINATE AXES:
  X AXIS (horizontal) = ANGULAR POSITION IN MINUTES:
    Left edge  = minute  0 = 12 o'clock
    1/4 width  = minute 15 = 3 o'clock
    1/2 width  = minute 30 = 6 o'clock
    3/4 width  = minute 45 = 9 o'clock
    Right edge = minute 60 = 12 o'clock again
    Formula: minute = (x_pixel - 60) / 1200 * 60

  Y AXIS (vertical) = RADIAL DEPTH IN RINGS:
    Top    = Ring 0  (R0) = innermost, adjacent to pupil
    Bottom = Ring 11 (R11) = outermost, adjacent to limbus
    Formula: ring = (y_pixel - 50) / 300 * 12

RING GROUP ZONES:
  R0        = IPB  (iris pupillary border)
  R1        = STOM (stomach ring)
  R2-R3     = ANW  (autonomic nerve wreath / collarette)
  R4-R9     = ORG  (organ zone)
  R10       = LYM  (lymphatic zone)
  R11       = SCU  (scurf rim / skin zone)

NASAL / TEMPORAL ORIENTATION:
  RIGHT EYE: TEMPORAL ~ minute 15 (3 o'clock), NASAL ~ minute 45 (9 o'clock)
  LEFT  EYE: NASAL ~ minute 15 (3 o'clock), TEMPORAL ~ minute 45 (9 o'clock)

HOW FEATURES APPEAR:
  CONCENTRIC (rings, ANW, bands) = HORIZONTAL stripes left-to-right
  RADIAL (furrows, clefts)       = VERTICAL dark stripes top-to-bottom
  POINT/LOCAL (lacunae, crypts)  = Discrete patches
  EYELID-MASKED areas            = White/blank bands (DO NOT report findings there)

CRITICAL: Read minute from TOP TICK LABELS, ring from LEFT R-LABELS.
DO NOT use atan2, radial distance, or any circular-geometry formula.
================================================================================`;

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

// =====================================================================
// CALL 1: FULL DETECTION (vision + image)
// Combines: STEP1 geo + STEP2A structural + STEP2B pigment + STEP2B_ANW collarette
// =====================================================================
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
PART B: STRUCTURAL DETECTION
================================================================================

Detect STRUCTURAL findings only (NO organ names, NO diagnosis):
  lacuna | crypt | giant_lacuna | atrophic_area | collarette_defect_lesion |
  radial_furrow | deep_radial_cleft | transversal_fiber | structural_asymmetry

For each: type, minuteRange [start, end], ringRange [start, end], size (xs/s/m/l),
          notes (<=60 chars), confidence (0.0-1.0).

IGNORE: White/blank eyelid-masked bands, glare/specular patches.

DEFINITIONS (how features appear in the UNWRAPPED STRIP):
- lacuna: horizontally elongated oval gap breaking fiber flow, lighter interior
- crypt: small deep dark triangular/rhomboid hole with sharp edges
- giant_lacuna: very large lacuna ≥8 minutes wide
- atrophic_area: absent/flattened fiber texture (NOT glare, NOT white band — dull, texture-free)
- collarette_defect_lesion: notch/break on ANW (rings R2-R3)
- radial_furrow: narrow VERTICAL dark stripe (1-3 min wide) from ANW outward
- deep_radial_cleft: wider VERTICAL dark stripe (4-8 min wide), deeper than furrow
- transversal_fiber: DIAGONAL line crossing radial fibers at an angle
- structural_asymmetry: visible fiber density/texture difference between strip halves

RANGE RULES:
- minuteRange NEVER wraps. Split [57,3] into [57,59] + [0,3] as two findings.
- ringRange: start ≤ end always.
- Point-like findings: width ≥ ±1 minute (min width = 2).
- Very wide (>20 min OR >3 rings): confidence -= 0.15; drop if <0.55.

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

The collarette (ANW) is the wavy band visible in rows R2-R3 (~50-100px from strip top).
Divide into 12 segments of 5 minutes each:
  Seg 1: min 0-5, Seg 2: min 5-10, ..., Seg 12: min 55-60

For EACH visible segment:
- position: high (near R2, contracted) | mid (R2-R3, normal) | low (near R3+, expanded)
- ringCenter: approximate ring position as decimal (e.g., 2.3, 2.8, 3.1)
- shape: normal | expanded | contracted | broken | notched | ballooning | flattened
- thickness: thin (<15px/<0.6 rings) | normal (15-30px) | thick (>30px/>1.2 rings)
- integrity: sharp | fuzzy | absent
- If segment is masked (white/eyelid): visible=false

ANW_status overall: expanded | contracted | broken | normal | mixed | unclear

List ANW defects: breaks, notches, ballooning with minuteRange + ringRange + notes.

contourSummary: avgRingCenter, minRingCenter, maxRingCenter, expandedSegments[],
contractedSegments[], brokenSegments[], overallIntegrity (good|moderate|poor).

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
      {"type":"specular|eyelid_band","minuteRange":[0,0],"ringRange":[0,0]}
    ]
  },
  "structural": [
    {"type":"...","minuteRange":[0,0],"ringRange":[0,0],"size":"xs|s|m|l","notes":"<=60","confidence":0.0}
  ],
  "pigment": [
    {"type":"...","subtype":"...","minuteRange":[0,0],"ringRange":[0,0],"severity":"low|medium|high","notes":"<=60","confidence":0.0}
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
      {"seg":1,"minuteRange":[0,5],"visible":true,"position":"high|mid|low","ringCenter":2.5,"shape":"normal|expanded|contracted|broken|notched|ballooning|flattened","thickness":"thin|normal|thick","integrity":"sharp|fuzzy|absent","confidence":0.0}
    ],
    "defects": [
      {"type":"break|notch|ballooning|lesion|pigment_on_ANW","minuteRange":[0,0],"ringRange":[2,3],"notes":"<=60","confidence":0.0}
    ],
    "contourSummary": {
      "avgRingCenter": 2.5,
      "minRingCenter": 2.0,
      "maxRingCenter": 3.0,
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

For EACH structural finding from CALL1:
1. Look at the stated minuteRange + ringRange in the actual image.
2. CONFIRM the finding exists (keep with same or adjusted confidence).
3. CORRECT minuteRange/ringRange if they seem off after careful re-examination.
4. REJECT if the area is actually white/blank (eyelid mask), glare, or normal tissue.
   Move rejected findings to "dropped" with reason.

For EACH pigment finding from CALL1:
1. Re-examine the actual image area.
2. CONFIRM, CORRECT ranges, or REJECT with reason.

CHECK FOR MISSED FINDINGS:
- Carefully scan the entire strip for features that CALL1 may have missed.
- Add any newly detected findings to the verified lists.

================================================================================
PART B: CONSISTENCY RULES (apply to verified findings)
================================================================================

CONTRADICTION RULES:
1) scurf_rim vs sodium_ring: same area → keep sodium_ring if light/milky; keep scurf_rim if dark.
2) pigment_spot vs lacuna/crypt: overlapping → keep structural; drop pigment.
3) lymphatic_rosary vs brushfield_like_spots: same area → keep rosary if chain/arc of discrete nodules.
4) Specular contamination: drop findings overlapping invalidRegions >25%.
5) collarette_defect_lesion vs ANW defects: same minuteRange → merge, keep ANW defect detail.

DEDUP/MERGE: Same type + minute overlap >60% AND ring overlap >60% → merge (union ranges, max confidence).

RANGE NORMALIZATION:
- Clamp minutes 0..59, rings 0..11.
- minuteRange NEVER wraps: split if start > end.
- Very wide (>20 min OR >3 rings) → confidence -= 0.15; drop if <0.55.
- usableUpperIris=false: findings in [57..59] or [0..3] → confidence -= 0.10.

COLLARETTE:
- collarette ringRange MUST be [2, 3]. Clamp if inconsistent, confidence -= 0.20.
- Cross-check ANW status: if structural defects contradict, note the discrepancy.

================================================================================
PART C: ZONE MAPPING — Match findings to anatomical zones
================================================================================

Match each VERIFIED finding to the most specific zone from MAP_V9 below using
(side + minuteRange + ringRange overlap):

${JSON.stringify(MAP_V9.filter(z => z.side === side || z.side === 'ANY'))}

MATCH RULE: zone matches if:
- zone.side == "${side}" or zone.side == "ANY"
- finding.minuteRange overlaps zone.mins (max(starts) ≤ min(ends))
- finding.ringRange overlaps zone.rings

TIE-BREAK: prefer side-specific over ANY, prefer smaller zone area.

================================================================================
PART D: PROFILE BUILD — Derive health axes and channels
================================================================================

From the verified and mapped findings, compute:
1. Constitution/disposition/diathesis from global traits.
2. ANW profile from collarette segments.
3. Elimination channels: gut_ANW → kidney → lymph → skin (status + evidence).
4. Axes: stress (0-100), digestive (0-100), immune (0-100).
5. Hypotheses: preventive health claims citing specific findings + zones.

================================================================================
OUTPUT — JSON ONLY — EXACT STRUCTURE:
================================================================================

{
  "imgId": "${imageHash}",
  "side": "${side}",
  "verified_structural": [
    {"fid":"S1","type":"...","minuteRange":[0,0],"ringRange":[0,0],"size":"xs|s|m|l","notes":"<=60","confidence":0.0,"zone":{"id":"...","organ_bg":"...","system_bg":"..."},"status":"confirmed|corrected|new"}
  ],
  "verified_pigment": [
    {"fid":"P1","type":"...","subtype":"...","minuteRange":[0,0],"ringRange":[0,0],"severity":"low|medium|high","notes":"<=60","confidence":0.0,"zone":{"id":"...","organ_bg":"...","system_bg":"..."},"status":"confirmed|corrected|new"}
  ],
  "collarette_verified": {
    "ANW_status": "expanded|contracted|broken|normal|mixed|unclear",
    "confidence": 0.0,
    "segments": [
      {"seg":1,"minuteRange":[0,5],"visible":true,"position":"high|mid|low","ringCenter":2.5,"shape":"normal|expanded|contracted|broken|notched|ballooning|flattened","thickness":"thin|normal|thick","integrity":"sharp|fuzzy|absent","confidence":0.0}
    ],
    "defects": [
      {"type":"break|notch|ballooning|lesion","minuteRange":[0,0],"ringRange":[2,3],"notes":"<=60","confidence":0.0}
    ],
    "contourSummary": {
      "avgRingCenter":2.5,"minRingCenter":2.0,"maxRingCenter":3.0,
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
  "zoneSummary": [
    {"zoneId":"...","organ_bg":"...","system_bg":"...","evidenceCount":0,"topTypes":["type:count"]}
  ],
  "profile": {
    "axesScore": {"stress0_100":0,"digestive0_100":0,"immune0_100":0},
    "elimChannels": [
      {"channel":"gut_ANW","status":"normal|attention|concern","evidence":[{"fid":"S1","zoneId":"..."}]},
      {"channel":"kidney","status":"normal|attention|concern","evidence":[]},
      {"channel":"lymph","status":"normal|attention|concern","evidence":[]},
      {"channel":"skin_scu","status":"normal|attention|concern","evidence":[]}
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
    {"type":"...","minuteRange":[0,0],"reason":"contradiction|specular|eyelid_band|too_wide|low_confidence|duplicate|false_positive"}
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
function promptCall3_Report(side, imageHash, call1, call2, questionnaire) {
  const q = questionnaire || {};
  return `IRIS PIPELINE — CALL3: Bulgarian Report Generation (v11)

ROLE: iris_frontend_report_generator_bg_v11
MODE: strict_json_only (NO image — text synthesis only)

INPUTS:
  DETECTION = ${JSON.stringify(call1)}
  VERIFIED  = ${JSON.stringify(call2)}
  QUESTIONNAIRE = ${JSON.stringify(q)}
  SIDE = ${side}
  IMG_ID = ${imageHash}

PREREQ: If VERIFIED.error exists → return error JSON.

You have VERIFIED detection results (CALL2 output). Synthesize them into the
final Bulgarian-language UI report. This is a TEXT-ONLY call — no image analysis.

================================================================================
CORE TRUTH RULE:
- ORGAN and SYSTEM names MUST come from VERIFIED.zoneSummary and VERIFIED.verified_structural / verified_pigment zone fields.
- The 12 UI zones below are DISPLAY BUCKETS ONLY.

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

ADVICE (Bulgarian, all values ≤ 120 chars):
  priorities: 3-6 bullets | nutrition.focus: 3-6 | nutrition.limit: 3-6
  lifestyle.sleep: 2-4 | lifestyle.stress: 2-4 | lifestyle.activity: 2-4
  followUp: 2-5 bullets

================================================================================
OUTPUT — JSON ONLY — EXACT STRUCTURE:
================================================================================

{
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
  "advice": {
    "priorities": ["<=120 chars БГ"],
    "nutrition": {"focus": ["<=120"], "limit": ["<=120"]},
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
      {"fid":"S1","type":"...","minuteRange":[0,0],"ringRange":[0,0],"size":"xs|s|m|l","zone":"zone_id","organ_bg":"...","confidence":0.0}
    ],
    "pigmentFindings": [
      {"fid":"P1","type":"...","subtype":"...","minuteRange":[0,0],"ringRange":[0,0],"severity":"low|medium|high","zone":"zone_id","organ_bg":"...","confidence":0.0}
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
      kvConfigured: !!env.IRIS_KV,
      aiBaseUrl: kvConfig?.baseUrl || env.AI_BASE_URL || 'https://api.openai.com/v1',
      geminiApiUrl: kvConfig?.geminiApiUrl || env.GEMINI_API_URL || 'https://generativelanguage.googleapis.com/v1beta',
      pipelineSteps: 3,
      pipelineDescription: 'CALL1(vision:detect) → CALL2(vision:verify+map) → CALL3(text:report)',
      timestamp: new Date().toISOString(),
    });
  }

  // GET /admin/cache/list?prefix=&cursor= — list KV keys (paginated, max 100)
  if (method === 'GET' && path === '/admin/cache/list') {
    if (!env.IRIS_KV) return jsonResp({ error: 'KV not configured' }, 503);
    const prefix = url.searchParams.get('prefix') || '';
    const cursor = url.searchParams.get('cursor') || undefined;
    const limit  = Math.min(Number(url.searchParams.get('limit') || 50), 100);
    const listOpts = { prefix, limit };
    if (cursor) listOpts.cursor = cursor;
    const list = await env.IRIS_KV.list(listOpts).catch(err => ({ error: err?.message }));
    return jsonResp(list);
  }

  // GET /admin/cache/entry?key= — read one KV entry
  if (method === 'GET' && path === '/admin/cache/entry') {
    if (!env.IRIS_KV) return jsonResp({ error: 'KV not configured' }, 503);
    const key = url.searchParams.get('key');
    if (!key) return jsonResp({ error: 'key query param required' }, 400);
    const value = await env.IRIS_KV.get(key, 'json').catch(() => null);
    if (value === null) return jsonResp({ error: 'Entry not found' }, 404);
    return jsonResp({ key, value });
  }

  // DELETE /admin/cache/entry?key= — delete one KV entry
  if (method === 'DELETE' && path === '/admin/cache/entry') {
    if (!env.IRIS_KV) return jsonResp({ error: 'KV not configured' }, 503);
    const key = url.searchParams.get('key');
    if (!key) return jsonResp({ error: 'key query param required' }, 400);
    await env.IRIS_KV.delete(key).catch(err => { throw err; });
    return jsonResp({ deleted: true, key });
  }

  // DELETE /admin/cache/flush — delete all KV entries (iterative)
  if (method === 'DELETE' && path === '/admin/cache/flush') {
    if (!env.IRIS_KV) return jsonResp({ error: 'KV not configured' }, 503);
    let cursor;
    let total = 0;
    do {
      const opts = { limit: 100 };
      if (cursor) opts.cursor = cursor;
      const list = await env.IRIS_KV.list(opts);
      const keys = list.keys.map(k => k.name);
      await Promise.all(keys.map(k => env.IRIS_KV.delete(k)));
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
    if (!env.IRIS_KV) return jsonResp({ error: 'KV not configured — cannot store settings' }, 503);
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
    await env.IRIS_KV.put(CONFIG_KV_KEY, JSON.stringify(newConfig));
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
    if (!env.IRIS_KV) return jsonResp({ error: 'KV not configured' }, 503);
    await env.IRIS_KV.delete(CONFIG_KV_KEY);
    return jsonResp({ deleted: true, message: 'KV config cleared — using env vars now' });
  }

  return jsonResp({ error: 'Admin route not found', path }, 404);
}

// =====================================================================
// KV CONFIG HELPERS
// =====================================================================
const CONFIG_KV_KEY = 'config:ai';

async function getKVConfig(env) {
  if (!env.IRIS_KV) return null;
  try {
    return await env.IRIS_KV.get(CONFIG_KV_KEY, 'json');
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
