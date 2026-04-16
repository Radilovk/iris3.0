/**
 * worker.js — Cloudflare Worker
 * Iris Iridology Analysis Pipeline v10 — Single AI Call Architecture
 *
 * Architecture:
 *   1. POST /analyze  — receive unwrapped iris strip + metadata, run 1 AI call
 *   2. GET  /result/:side/:hash — retrieve a cached analysis result
 *
 * Innovation: All 8 previous pipeline steps (STEP1–STEP5) are consolidated into
 * a single comprehensive vision prompt. This reduces Cloudflare/AI API usage from
 * 8 requests per analysis to just 1, while producing identical output format.
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
    version: 'v10.0-single-call',
    provider,
    model,
    apiKeyConfigured: hasApiKey,
    kvConfigured: !!env.IRIS_KV,
  });
}

// =====================================================================
// ROUTE: POST /analyze — Single AI Call Pipeline
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

  const effectiveEnv = createEffectiveEnv(env, aiProvider, aiModel);
  const effectiveModel = effectiveEnv.AI_MODEL;
  const cacheKey = `result:${side}:${imageHash}:${effectiveModel}`;

  // Check cache
  let cached = null;
  try {
    cached = await env.IRIS_KV.get(cacheKey, 'json');
  } catch (kvErr) {
    console.error('KV get error:', kvErr?.message || kvErr);
  }
  if (cached) {
    return jsonResp({ cached: true, imageHash, side, model: effectiveModel, result: cached });
  }

  // Build mega prompt and make single AI call
  const imageDataUrl = `data:image/jpeg;base64,${stripB64}`;
  const prompt = buildMegaPrompt(side, imageHash, questionnaire);
  const result = await aiCall(effectiveEnv, prompt, imageDataUrl);

  // If AI returned an error, wrap it
  if (result.error) {
    return jsonResp({
      cached: false,
      imageHash,
      side,
      model: effectiveModel,
      result: { error: result.error, stage: 'ANALYSIS', imageHash, side }
    });
  }

  // Store in KV with 24-hour TTL
  await env.IRIS_KV.put(cacheKey, JSON.stringify(result), { expirationTtl: 86400 }).catch(() => {});

  return jsonResp({ cached: false, imageHash, side, model: effectiveModel, result });
}

function createEffectiveEnv(env, aiProvider, aiModel) {
  return {
    IRIS_KV: env.IRIS_KV,
    AI_API_KEY: env.AI_API_KEY,
    AI_BASE_URL: env.AI_BASE_URL,
    GEMINI_API_URL: env.GEMINI_API_URL,
    AI_PROVIDER: aiProvider || env.AI_PROVIDER || 'gemini',
    AI_MODEL: aiModel || env.AI_MODEL || 'gemini-2.0-flash',
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
    max_tokens: 16384, // v10: single comprehensive prompt needs larger output budget
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
      maxOutputTokens: 16384, // v10: single comprehensive prompt needs larger output budget
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
// IMAGE FORMAT PREAMBLE
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
// MEGA PROMPT — Single comprehensive vision analysis
// =====================================================================
function buildMegaPrompt(side, imageHash, questionnaire) {
  const q = questionnaire || {};
  return `${IMAGE_FORMAT}

================================================================================
IRIS_COMPREHENSIVE_ANALYSIS — Single-Pass Full Report (v10)
================================================================================

ROLE: expert_iridologist_v10
MODE: image_parse_only + full_report
INPUT: unwrapped_iris_strip (rectangular image as described in IMAGE_FORMAT above)
SIDE: ${side}
IMG_ID: ${imageHash}
QUESTIONNAIRE: ${JSON.stringify(q)}

You are an expert iridologist performing a COMPLETE analysis of this unwrapped iris
strip image in a SINGLE pass. You must perform ALL of the following tasks and produce
ONE unified JSON output.

================================================================================
TASK 1: GEO CALIBRATION & QUALITY CHECK
================================================================================

- Verify the strip is a valid unwrapped iris with visible grid labels.
- Assess image quality: focus (good/med/poor), glare (none/low/med/high),
  occlusion (none/low/med/high).
- Identify eyelid-masked (white/blank) regions and specular/glare patches.
- If the strip is unusable (>35% white, poor focus throughout, invalid strip),
  return ONLY: {"error":{"stage":"ANALYSIS","code":"LOW_QUALITY","message":"<reason>","canRetry":true}}

================================================================================
TASK 2: STRUCTURAL DETECTION (what was STEP2A)
================================================================================

Detect STRUCTURAL findings only (no organ names, no diagnosis):
  lacuna | crypt | giant_lacuna | atrophic_area | collarette_defect_lesion |
  radial_furrow | deep_radial_cleft | transversal_fiber | structural_asymmetry

For each finding: type, minuteRange, ringRange, size (xs/s/m/l), confidence (0-1).
IGNORE white/blank eyelid bands and glare patches.

Definitions:
- lacuna: oval gap breaking fiber flow, horizontally elongated in strip
- crypt: small deep dark triangular hole with sharp edges
- radial_furrow: narrow VERTICAL dark stripe from ANW outward
- deep_radial_cleft: wider VERTICAL dark stripe, broader than furrow
- transversal_fiber: DIAGONAL line crossing radial fibers
- collarette_defect_lesion: notch/break on ANW (rings R2-R3)

================================================================================
TASK 3: PIGMENT & RING DETECTION (what was STEP2B)
================================================================================

Detect:
  pigment_spot | pigment_cloud | pigment_band | brushfield_like_spots |
  nerve_rings | lymphatic_rosary | scurf_rim | sodium_ring

Assess global traits:
- constitution: LYM | HEM | BIL | unclear
- disposition: SILK | LINEN | BURLAP | unclear
- diathesis_tags: HAC | LRS | LIP | DYS (with confidence)

Assess ANW/collarette:
- ANW_status: expanded | contracted | broken | normal | mixed | unclear
- Divide the collarette into 12 segments of 5 minutes each (seg 1: min 0-5, ..., seg 12: min 55-60)
- For each visible segment: position (high/mid/low), shape, thickness, integrity, ringCenter

================================================================================
TASK 4: ZONE MAPPING (what was STEP3)
================================================================================

Match each finding to the most specific zone from this MAP using side + minuteRange + ringRange overlap:

${JSON.stringify(MAP_V9.filter(z => z.side === side || z.side === 'ANY'))}

Tie-break: prefer side-specific over ANY, prefer smaller zone area.

================================================================================
TASK 5: REPORT GENERATION (what was STEP5 — Bulgarian)
================================================================================

Generate the final report in BULGARIAN for the frontend UI.

MINUTE-TO-ZONE CONVERSION (for 12 UI zones):
  Zone  1 ("12-1ч"):  minutes  0-5   -> degrees   0-30
  Zone  2 ("1-2ч"):   minutes  5-10  -> degrees  30-60
  Zone  3 ("2-3ч"):   minutes 10-15  -> degrees  60-90
  Zone  4 ("3-4ч"):   minutes 15-20  -> degrees  90-120
  Zone  5 ("4-5ч"):   minutes 20-25  -> degrees 120-150
  Zone  6 ("5-6ч"):   minutes 25-30  -> degrees 150-180
  Zone  7 ("6-7ч"):   minutes 30-35  -> degrees 180-210
  Zone  8 ("7-8ч"):   minutes 35-40  -> degrees 210-240
  Zone  9 ("8-9ч"):   minutes 40-45  -> degrees 240-270
  Zone 10 ("9-10ч"):  minutes 45-50  -> degrees 270-300
  Zone 11 ("10-11ч"): minutes 50-55  -> degrees 300-330
  Zone 12 ("11-12ч"): minutes 55-60  -> degrees 330-360

VALIDATION PRIORITY (cross-reference with QUESTIONNAIRE):
  1) HIGH   — confirmed by questionnaire
  2) MEDIUM — not mentioned (preventive)
  3) LOW    — contradicts questionnaire

================================================================================
OUTPUT — JSON ONLY — EXACT STRUCTURE BELOW
================================================================================

{
  "analysis": {
    "zones": [
      {"id":1,"name":"12-1ч","organ":"<БГ орган от MAP>","status":"normal|attention|concern","findings":"<=60 chars БГ","angle":[0,30]},
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
    "priorities": ["<=120 chars БГ bullet 1", "..."],
    "nutrition": {"focus": ["<=120 chars"], "limit": ["<=120 chars"]},
    "lifestyle": {"sleep": ["<=120"], "stress": ["<=120"], "activity": ["<=120"]},
    "followUp": ["<=120 chars"]
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
      {"type":"...","minuteRange":[0,0],"ringRange":[0,0],"size":"xs|s|m|l","zone":"zone_id","organ_bg":"...","confidence":0.0}
    ],
    "pigmentFindings": [
      {"type":"...","subtype":"...","minuteRange":[0,0],"ringRange":[0,0],"severity":"low|medium|high","zone":"zone_id","organ_bg":"...","confidence":0.0}
    ]
  }
}

RULES:
- JSON ONLY output, no markdown, no extra text
- All UI text in BULGARIAN
- No double quotes inside string values
- findings text <= 60 chars, descriptions <= 60 chars, advice <= 120 chars
- severity: low|medium|high
- Always output exactly 12 zones, exactly 6 systemScores
- 2-5 artifacts (strongest findings)
- organ names MUST come from the zone MAP provided above
- minuteRange NEVER wraps: split [57,3] into two findings
- IGNORE white/blank eyelid-masked areas — do NOT report findings there

FAILSAFE — if image is unusable, return ONLY:
{"error":{"stage":"ANALYSIS","code":"LOW_QUALITY|INVALID_STRIP","message":"<short reason>","canRetry":true}}`;
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
