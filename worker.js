/**
 * worker.js — Cloudflare Worker
 * Iris Iridology Analysis Pipeline v9.1
 *
 * Architecture:
 *   1. POST /analyze  — receive unwrapped iris strip + metadata, run 6-step AI pipeline
 *   2. GET  /result/:side/:hash — retrieve a cached analysis result
 *
 * Environment bindings (set in wrangler.toml / Cloudflare dashboard):
 *   IRIS_KV         — KV namespace for caching results
 *   AI_API_KEY      — secret: API key (OpenAI or Google Gemini)
 *   AI_PROVIDER     — var: provider name ("openai", "gemini", "openai-compatible")
 *   AI_MODEL        — var: model name (e.g., "gemini-2.0-flash", "gpt-4o", "gpt-4o-mini")
 *   AI_BASE_URL     — var: OpenAI API base URL (default: "https://api.openai.com/v1")
 *   GEMINI_API_URL  — var: Gemini API base URL (default: "https://generativelanguage.googleapis.com/v1beta")
 *
 * Request format (POST /analyze, multipart/form-data):
 *   strip_image   — base64-encoded JPEG of the unwrapped iris strip (from app.py)
 *   side          — "R" or "L"
 *   image_hash    — optional unique ID; auto-generated if omitted
 *   questionnaire — optional JSON string with patient data
 *
 * The unwrapped strip is produced by app.py's unwrap_iris_fast() + draw_ai_grid_map_expanded():
 *   Canvas:  ~1280 × 390 px
 *   Content: 1200 × 300 px iris strip (inside 60px left margin + 50px top margin)
 *   X-axis:  minutes 0–60 (left=0=12 o'clock, right=60=12 o'clock)
 *   Y-axis:  rings R0–R11 (top=R0=innermost, bottom=R11=outermost)
 *   Labels:  minute ticks at top, R0–R11 at left, NASAL/TEMPORAL at bottom
 *
 * Pipeline steps:
 *   STEP1      — geo calibration (vision)
 *   STEP2A     — structural detector (vision, parallel)
 *   STEP2B     — pigment/rings detector (vision, parallel)
 *   STEP2B_ANW — ANW collarette contour profiler (vision, parallel)
 *   STEP2C     — consistency validator (text)
 *   STEP3      — zone mapper (text)
 *   STEP4      — profile builder (text)
 *   STEP5      — Bulgarian report (text)
 */

// =====================================================================
// CORS HEADERS
// =====================================================================
const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

// Maximum bytes of raw AI/HTTP error text to include in error responses.
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
        const key = url.pathname.slice('/result/'.length); // "R:abc123"
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
      { id: 'gpt-4-vision-preview', name: 'GPT-4 Vision Preview', vision: true },
      { id: 'o1-preview', name: 'O1 Preview', vision: false, reasoning: true },
    ]
  },
  'openai-compatible': {
    name: 'OpenAI-Compatible APIs',
    models: [
      { id: 'claude-3-opus-20240229', name: 'Claude 3 Opus', vision: true },
      { id: 'claude-3-sonnet-20240229', name: 'Claude 3 Sonnet', vision: true },
      { id: 'claude-3-haiku-20240307', name: 'Claude 3 Haiku', vision: true, costEffective: true },
      { id: 'llava-v1.6-34b', name: 'LLaVA v1.6 34B', vision: true, local: true },
      { id: 'mistral-large-latest', name: 'Mistral Large', vision: false },
    ]
  }
};

// =====================================================================
// ROUTE: GET /models - List available AI models
// =====================================================================
function handleGetModels(env) {
  const currentProvider = env.AI_PROVIDER || 'gemini';
  const currentModel = env.AI_MODEL || 'gemini-2.0-flash';
  
  return jsonResp({
    currentConfig: {
      provider: currentProvider,
      model: currentModel,
    },
    availableProviders: AVAILABLE_MODELS,
    note: 'You can override the model per-request by passing ai_provider and ai_model in the /analyze request',
  });
}

// =====================================================================
// ROUTE: GET /health - Health check endpoint
// =====================================================================
function handleHealthCheck(env) {
  const hasApiKey = !!env.AI_API_KEY;
  const provider = env.AI_PROVIDER || 'gemini';
  const model = env.AI_MODEL || 'gemini-2.0-flash';
  
  return jsonResp({
    status: hasApiKey ? 'healthy' : 'degraded',
    version: 'v9.1',
    provider,
    model,
    apiKeyConfigured: hasApiKey,
    kvConfigured: !!env.IRIS_KV,
  });
}

// =====================================================================
// ROUTE: POST /analyze
// =====================================================================
async function handleAnalyze(request, env) {
  const form = await request.formData();
  const side         = (form.get('side') || 'R').toUpperCase();
  const stripB64     = form.get('strip_image');
  const imageHash    = form.get('image_hash') || genId();
  const qRaw         = form.get('questionnaire');
  const questionnaire = qRaw ? safeParseJSON(qRaw) : {};
  
  // Dynamic AI model configuration from request (overrides env defaults)
  const aiProvider   = form.get('ai_provider') || null;
  const aiModel      = form.get('ai_model') || null;

  if (!stripB64) {
    return jsonResp({ error: 'strip_image is required (base64 JPEG of the unwrapped iris strip from app.py)' }, 400);
  }
  if (side !== 'R' && side !== 'L') {
    return jsonResp({ error: 'side must be "R" or "L"' }, 400);
  }

  // Create effective env with request-level overrides
  const effectiveEnv = createEffectiveEnv(env, aiProvider, aiModel);
  
  // Always include effective model in cache key for consistency
  // This ensures identical images analyzed with the same model (explicit or default) hit the same cache
  const effectiveModel = effectiveEnv.AI_MODEL;
  const cacheKey = `result:${side}:${imageHash}:${effectiveModel}`;
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

  // Run pipeline with effective env
  const pipeline = new IrisPipeline(effectiveEnv, stripB64, side, imageHash, questionnaire);
  const result = await pipeline.run();

  // Store in KV with 24-hour TTL
  await env.IRIS_KV.put(cacheKey, JSON.stringify(result), { expirationTtl: 86400 }).catch(() => {});

  return jsonResp({ cached: false, imageHash, side, model: effectiveModel, result });
}

// Create effective environment with request-level AI config overrides
function createEffectiveEnv(env, aiProvider, aiModel) {
  // Return a proxy-like object that overrides AI settings if provided
  return {
    // Pass through all original env properties
    IRIS_KV: env.IRIS_KV,
    AI_API_KEY: env.AI_API_KEY,
    AI_BASE_URL: env.AI_BASE_URL,
    GEMINI_API_URL: env.GEMINI_API_URL,
    // Override AI_PROVIDER if specified in request
    AI_PROVIDER: aiProvider || env.AI_PROVIDER || 'gemini',
    // Override AI_MODEL if specified in request
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
// PIPELINE
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
    // STEP 1 — geo calibration (needs image)
    const step1 = await this.visionCall(promptStep1(this.side, this.imageHash));
    if (step1.error) {
      return { error: step1, stage: 'STEP1', imageHash: this.imageHash, side: this.side };
    }

    // STEP 2A + 2B + 2B_ANW — parallel: structural + pigment + ANW contour (all need image)
    const [step2a, step2b, step2anw] = await Promise.all([
      this.visionCall(promptStep2A(this.side, step1)),
      this.visionCall(promptStep2B(this.side, step1)),
      this.visionCall(promptStep2B_ANW(this.side, step1)),
    ]);

    // STEP 2C — consistency validator (text only, now includes ANW contour)
    const step2c = await this.textCall(promptStep2C(this.side, step1, step2a, step2b, step2anw));

    // STEP 3 — mapper (text only)
    const step3 = await this.textCall(promptStep3(this.side, step1, step2c));

    // STEP 4 — profile builder (text only, now includes ANW contour)
    const step4 = await this.textCall(promptStep4(this.side, step3, step2c));

    // STEP 5 — final Bulgarian report (text only)
    const step5 = await this.textCall(
      promptStep5(this.side, step1, step2c, step3, step4, this.questionnaire)
    );

    return {
      imageHash: this.imageHash,
      side: this.side,
      pipeline: { step1, step2a, step2b, step2anw, step2c, step3, step4 },
      report: step5,
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
// AI CALL — Multi-provider vision/text endpoint (OpenAI, Gemini)
// =====================================================================
async function aiCall(env, prompt, imageDataUrl) {
  const provider = (env.AI_PROVIDER || 'openai').toLowerCase();
  
  if (provider === 'gemini') {
    return await aiCallGemini(env, prompt, imageDataUrl);
  } else {
    // Default to OpenAI-compatible API
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
    max_tokens: 4096,
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
  
  // Prepare content parts
  const parts = [{ text: prompt }];
  
  // If image provided, convert data URL to inline data format for Gemini
  if (imageDataUrl) {
    // Extract base64 data and MIME type from data URL
    const matches = imageDataUrl.match(/^data:(image\/[^;]+);base64,(.+)$/);
    if (matches) {
      const mimeType = matches[1];
      const base64Data = matches[2];
      parts.push({
        inline_data: {
          mime_type: mimeType,
          data: base64Data
        }
      });
    }
  }
  
  const body = {
    contents: [{
      parts: parts
    }],
    generationConfig: {
      temperature: 0.1,
      maxOutputTokens: 4096,
      responseMimeType: 'application/json'
    }
  };

  let resp;
  try {
    resp = await fetch(`${baseUrl}/models/${model}:generateContent?key=${env.AI_API_KEY}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
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
  const content = data?.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!content) {
    return { error: { code: 'EMPTY_RESPONSE', message: 'Gemini returned no content' } };
  }

  return safeParseJSON(content) || { error: { code: 'JSON_PARSE_ERROR', raw: content.slice(0, ERR_MSG_LIMIT) } };
}

// =====================================================================
// IMAGE FORMAT PREAMBLE (prepended to every image-analysis prompt)
// Pixel positions match app.py draw_ai_grid_map_expanded():
//   pt=50 (top margin), pl=60 (left margin), pb=40, pr_pad=20
//   Iris content: 1200 × 300 px; rings: 12 equal bands of 25 px each
// =====================================================================
const IMAGE_FORMAT = `\
================================================================================
IMAGE_FORMAT: UNWRAPPED_IRIS_STRIP (READ THIS BEFORE ANYTHING ELSE)
================================================================================

CRITICAL NOTICE:
The image you are analyzing is NOT a raw circular iris photograph.
It is an UNWRAPPED (unrolled / linearized) iris strip produced by a
polar-to-rectangular transformation of the original circular iris image.
The circular ring of the iris has been "straightened" into a horizontal rectangle.

STRIP DIMENSIONS:
  Full canvas : ~1280 × 390 px  (includes labeled margins)
  Iris content: ~1200 × 300 px  (inside margins)
  Left margin (~60 px) : ring labels R0, R1, ..., R11 at each ring boundary
  Top margin  (~50 px) : minute tick marks labeled 0, 5, 10, 15, ..., 60 (every 5 min)
  Bottom area          : eye side label ("RIGHT EYE" / "LEFT EYE"), "^ NASAL", "^ TEMPORAL"

COORDINATE AXES:
  X AXIS (horizontal, left → right) = ANGULAR POSITION IN MINUTES:
    Left edge  → minute  0  (= 12 o'clock, top of the original circular iris)
    1/4 width  → minute 15  (= 3 o'clock)
    1/2 width  → minute 30  (= 6 o'clock, bottom of the original circular iris)
    3/4 width  → minute 45  (= 9 o'clock)
    Right edge → minute 60  (= 12 o'clock again — same angular position as minute 0)
    Formula: minute = (x_pixel − 60) / 1200 × 60
    Use the printed tick labels at the TOP of the image to read minute directly.

  Y AXIS (vertical, top → bottom) = RADIAL DEPTH IN RINGS:
    Top of strip    → Ring 0  (R0) — innermost, adjacent to pupil edge
    Bottom of strip → Ring 11 (R11) — outermost, adjacent to iris/limbus edge
    Formula: ring = (y_pixel − 50) / 300 × 12
    Use the printed R0–R11 labels in the LEFT MARGIN to read ring directly.

RING GROUP ZONES (with vertical pixel positions in the 300 px iris content):
  R0        → IPB  (iris pupillary border)   — top  0– 25 px
  R1        → STOM (stomach ring)            — top 25– 50 px
  R2–R3     → ANW  (autonomic nerve wreath)  — top 50–100 px  (upper 1/6 of strip)
  R4–R9     → ORG  (organ zone)             — top 100–250 px  (middle 1/2 of strip)
  R10       → LYM  (lymphatic zone)         — top 250–275 px
  R11       → SCU  (scurf rim / skin zone)  — top 275–300 px  (bottom 1/12 of strip)

NASAL / TEMPORAL ORIENTATION (visible at the bottom of the image):
  RIGHT EYE: "^ TEMPORAL" ≈ minute 15 (3 o'clock),  "^ NASAL" ≈ minute 45 (9 o'clock)
  LEFT  EYE: "^ NASAL"    ≈ minute 15 (3 o'clock),  "^ TEMPORAL" ≈ minute 45 (9 o'clock)

HOW CIRCULAR FEATURES APPEAR IN THE RECTANGULAR STRIP:
  CONCENTRIC features (rings, rosary arcs, color bands, ANW collarette):
    → HORIZONTAL stripes / wavy bands running LEFT-TO-RIGHT across the strip.
    → Examples: nerve_rings, lymphatic_rosary, scurf_rim, sodium_ring, pigment_band, ANW.

  RADIAL features (furrows and clefts running from pupil outward toward limbus):
    → VERTICAL dark stripes running TOP-TO-BOTTOM in the strip.
    → Examples: radial_furrow, deep_radial_cleft.

  POINT / LOCAL features (lacunae, crypts, pigment spots):
    → Discrete patches. Lacunae may be WIDER than tall (horizontally elongated)
      especially at minutes 0–10 and 50–60 (12 o'clock poles) and minute 30 (6 o'clock).
    → Crypts remain compact and nearly square or triangular.

  DIAGONAL / CROSSING features (transversal fibers):
    → Lines at a DIAGONAL ANGLE crossing the strip; not purely vertical or horizontal.

  EYELID-MASKED areas (white/blank regions):
    → Upper eyelid: white band at LEFT edge (minutes 0–10) AND RIGHT edge (minutes 50–60).
      Both edges are the same 12 o'clock region because the strip starts and ends there.
    → Lower eyelid: white band near strip CENTER (minutes 25–35, the 6 o'clock region).
    → These bands span the FULL HEIGHT of the strip (R0 through R11).
    → DO NOT report iris findings in white/blank areas — they are masked, not iris tissue.

CRITICAL — DO NOT:
  - Detect or describe a limbus ellipse, pupil circle, or any circular boundary.
  - Use atan2, cos, sin, or radial distance formulas to locate features.
  - Compute any position relative to an image center point.
  Instead: READ minute from the TOP TICK LABELS, READ ring from the LEFT R-LABELS.
  This is the ONLY correct localization method for this unwrapped strip image.
================================================================================`;

// =====================================================================
// PROMPT BUILDERS
// =====================================================================

function promptStep1(side, imageHash) {
  return `${IMAGE_FORMAT}

================================================================================
STEP1_geo_calibration
================================================================================

ROLE: iris_geo_calibrator_v9
MODE: image_parse_only
INPUT: unwrapped_iris_strip — rectangular image as described in IMAGE_FORMAT above
SIDE: ${side}
IMG_ID: ${imageHash}

GOAL:
- Decide if the unwrapped strip is usable for iris analysis.
- Confirm the v9 coordinate grid printed on the image margins.

COORDINATE READING (CRITICAL):
- DO NOT detect limbus ellipse, pupil circle, or apply any circular geometry.
- Minute: read from X position using the TOP tick labels (each interval = 5 min; interpolate).
- Ring: read from Y position using the LEFT R0–R11 labels (each row = one ring; top = R0).
- Quarter reference: 1/4 strip width ≈ min 15, 1/2 ≈ min 30, 3/4 ≈ min 45.

SPECULAR / GLARE HANDLING:
- Detect bright white or washed-out patches without iris texture.
- Locate their minuteRange + ringRange by reading the grid axes.
- Mark as invalidRegions.

EYELID OCCLUSION:
- Upper eyelid: white band at LEFT edge (min 0–10) AND RIGHT edge (min 50–60).
  If it covers >1/3 of strip height in those columns: usableUpperIris=false.
- Lower eyelid: white band near center (min 25–35, 6 o'clock).

QUALITY GATE — set ok=false if ANY:
- focus="poor" (iris fiber texture blurry or unrecognizable throughout the strip)
- >35% of strip area is white/blank (masked or glare)
- Specular/glare covers the majority of the ANW rows (R2–R3, top 50–100 px)
- Strip is invalid (all white/black, not an iris strip)

refRay15Usable: Is the minute-15 column (3 o'clock — temporal for RIGHT, nasal for LEFT)
  visible and free of white masking or glare? Set false if predominantly white/blank.

OUTPUT_JSON ONLY:
{
  "imgId": "${imageHash}",
  "side": "${side}",
  "ok": true,
  "rejectReasons": [],
  "quality": {
    "score0_100": 0,
    "focus": "good|med|poor",
    "glare": "none|low|med|high",
    "occlusion": "none|low|med|high",
    "issues": []
  },
  "geo": {
    "mins": 60,
    "rings": 12,
    "degPerMin": 6,
    "refMinute": 15,
    "ringGroups": {
      "IPB": [0, 0],
      "STOM": [1, 1],
      "ANW": [2, 3],
      "ORG": [4, 9],
      "LYM": [10, 10],
      "SCU": [11, 11]
    }
  },
  "refRay15Usable": true,
  "usableUpperIris": true,
  "invalidRegions": [
    { "type": "specular", "minuteRange": [0, 0], "ringRange": [0, 0], "confidence": 0.0 }
  ]
}

FAILSAFE — if cannot comply, return ONLY:
{"error":{"stage":"STEP1","code":"FORMAT_FAIL|LOW_QUALITY|INVALID_STRIP|STRIP_MOSTLY_OCCLUDED","message":"<short reason>","canRetry":true}}`;
}

function promptStep2A(side, step1) {
  return `${IMAGE_FORMAT}

================================================================================
STEP2A_structural_detector
================================================================================

ROLE: iris_detector_struct_v9
MODE: image_parse_only
INPUT: unwrapped_iris_strip — rectangular image as described in IMAGE_FORMAT above
SIDE: ${side}
GEO: ${JSON.stringify(step1)}

PREREQ: If GEO.ok != true → return error JSON immediately.

TARGET: IRIS STRUCTURE ONLY — NO meaning, NO organ names, NO diagnosis.

IGNORE: White/blank eyelid-masked bands | glare/specular patches | GEO.invalidRegions

LOCALIZATION — CRITICAL (UNWRAPPED STRIP):
- Minute: observe X position → read from TOP tick labels (0 at left edge = 12 o'clock,
  60 at right edge = 12 o'clock again). Interpolate between ticks.
- Ring: observe Y position → read from LEFT R-labels (R0 at top = innermost,
  R11 at bottom = outermost).
- Output minuteRange + ringRange ALWAYS.
- Point-like findings: minuteRange width must be at least ±1 minute (min width = 2).
- Prefer ranges over artificial precision.
- DO NOT use atan2, radial distance, or any circular-geometry formula.

RANGE NORMALIZATION (MANDATORY):
- minuteRange NEVER wraps. If it would cross minute 59→0, SPLIT into two findings:
  one [start, 59] and one [0, end].
- ringRange must be contiguous (start ≤ end).
- minute width > 20 OR ring width > 3 → confidence −= 0.15; drop if < 0.55.
- Single-minute or single-ring only if confidence ≥ 0.85 AND focus = "good".

UPPER IRIS PENALTY:
- If GEO.usableUpperIris = false: findings centered in minutes [57..59] or [0..3]
  → reduce confidence by 0.15.

DETECT (STRUCTURAL ONLY):
  lacuna | crypt | giant_lacuna | atrophic_area | collarette_defect_lesion |
  radial_furrow | deep_radial_cleft | transversal_fiber | structural_asymmetry

DEFINITIONS (with strip appearance):
- lacuna: oval/leaf gap breaking fiber flow; not pure black; shallow-to-mid depth.
  IN STRIP → horizontally elongated oval or lens-shaped darker patch with slightly
  lighter interior; wider than tall, especially near minutes 0–10, 50–60 (12 o'clock
  poles) and minute 30 (6 o'clock). More oval-shaped away from those positions.

- crypt: small deep dark rhomboid/triangular hole; sharp edges; fibers abruptly stop.
  IN STRIP → compact near-black slot or small triangle with sharply defined edges;
  roughly as wide as tall or slightly taller than wide.

- giant_lacuna: very large lacuna dominating a full sector (≥ ~8 minutes wide).
  IN STRIP → a conspicuously large dark oval spanning at least 8 minute columns
  (≈ 1/7 of the full strip width); clearly larger than any ordinary lacuna.

- atrophic_area: locally absent or flattened fiber texture (NOT glare, NOT a white band).
  IN STRIP → a region where the horizontal fiber striation pattern is absent or
  greatly reduced; the area is NOT bright white — it has a dull, texture-free look.

- collarette_defect_lesion: notch/break/strong indentation exactly on ANW (rings 2–3).
  IN STRIP → a gap, indentation, or interruption in the roughly horizontal wavy band
  of the collarette (rows R2–R3, approximately 50–100 px from the top of the strip content).

- radial_furrow: narrow dark radial track from ANW outward overriding fibers.
  IN STRIP → a narrow VERTICAL dark stripe (1–3 minute columns wide) running
  TOP-TO-BOTTOM from the ANW row area (top ~50–100 px) downward; darker than fibers.

- deep_radial_cleft: wider/deeper radial channel than radial_furrow.
  IN STRIP → a wider VERTICAL dark stripe (4–8 minute columns wide) running
  top-to-bottom; distinctly broader and deeper than a radial_furrow.

- transversal_fiber: non-radial crossing line(s) cutting across radial fibers.
  IN STRIP → a DIAGONAL line or narrow band crossing the strip at a clear angle
  that differs from vertical; it cuts across several ring rows diagonally.

- structural_asymmetry: strong structural difference between sectors not due to glare.
  IN STRIP → visible difference in fiber density, texture, or color between clearly
  separated columns or halves of the strip; not caused by white eyelid/glare bands.

OUTPUT_JSON ONLY:
{
  "imgId": "${step1.imgId || ''}",
  "side": "${side}",
  "findings": [
    {
      "type": "lacuna|crypt|giant_lacuna|atrophic_area|collarette_defect_lesion|radial_furrow|deep_radial_cleft|transversal_fiber|structural_asymmetry",
      "minuteRange": [0, 0],
      "ringRange": [0, 0],
      "size": "xs|s|m|l",
      "notes": "<=60 chars describing appearance in the strip",
      "confidence": 0.0
    }
  ],
  "excluded": [
    { "type": "specular|eyelid_band", "minuteRange": [0, 0], "ringRange": [0, 0] }
  ]
}

FAILSAFE:
If GEO.ok != true: {"error":{"stage":"STEP2A","code":"PREREQ_FAIL","message":"geo not ok","canRetry":true}}
If cannot comply: {"error":{"stage":"STEP2A","code":"FORMAT_FAIL","message":"<reason>","canRetry":true}}`;
}

function promptStep2B(side, step1) {
  return `${IMAGE_FORMAT}

================================================================================
STEP2B_pigment_rings_detector
================================================================================

ROLE: iris_detector_pigment_rings_v9
MODE: image_parse_only
INPUT: unwrapped_iris_strip — rectangular image as described in IMAGE_FORMAT above
SIDE: ${side}
GEO: ${JSON.stringify(step1)}

PREREQ: If GEO.ok != true → return error JSON immediately.

TARGET: IRIS STRUCTURE ONLY — NO meaning, NO organ names, NO diagnosis.

IGNORE: White/blank eyelid-masked bands | glare/specular patches | GEO.invalidRegions

LOCALIZATION — CRITICAL (UNWRAPPED STRIP):
- Minute: observe X position → read from TOP tick labels.
- Ring: observe Y position → read from LEFT R-labels.
- Always minuteRange + ringRange. Point-like: width ≥ ±1 minute (min width = 2).
- DO NOT use atan2, radial distance, or any circular-geometry formula.

RANGE NORMALIZATION (MANDATORY):
- minuteRange NEVER wraps; split into two findings if it would cross 59→0.
- ringRange contiguous (start ≤ end).
- minute width > 20 OR ring width > 3 → confidence −= 0.15; drop if < 0.55.
- Single min/ring only if confidence ≥ 0.85 AND focus = "good".

UPPER IRIS PENALTY:
- GEO.usableUpperIris = false → findings in minutes [57..59] or [0..3]: confidence −= 0.15.

DETECT (PIGMENT / PERIPHERY / RINGS):
  pigment_spot | pigment_cloud | pigment_band | brushfield_like_spots |
  nerve_rings | lymphatic_rosary | scurf_rim | sodium_ring

COLLARETTE (ANW):
- ANW_status: expanded | contracted | broken | normal | unclear
- collarette ringRange MUST intersect [2, 3].
- IN STRIP: the ANW/collarette appears as a roughly horizontal, slightly wavy or undulating
  band in the UPPER PORTION of the strip (rows R2–R3, approximately 50–100 px from the
  top edge of the iris content). An expanded ANW pushes that band downward toward R3;
  a contracted ANW sits higher near R2. A broken ANW shows clear gaps in the band.

GLOBAL TRIAD TAGS (only if visually supported):
- constitution: LYM | HEM | BIL | unclear
- disposition: SILK | LINEN | BURLAP | unclear
- diathesis_tags: HAC | LRS | LIP | DYS

DEFINITIONS (with strip appearance):
- pigment_spot: bounded colored spot on fibers, NO structural gap beneath it.
  IN STRIP → small colored dot or smudge; may be slightly wider than tall near the
  12 o'clock poles (left/right edges) but remains compact elsewhere.
  subtype: orange_rust | brown_black | yellow | other

- pigment_cloud: diffuse haze field with soft, gradual edges.
  IN STRIP → soft-edged, diffusely colored area spanning several minute columns
  and ring rows with no sharp boundary; fades gradually.

- pigment_band: belt or arc of color following the concentric direction around the iris.
  IN STRIP → a HORIZONTAL colored stripe running LEFT-TO-RIGHT across the strip,
  confined to 1–2 ring rows. Distinguished from nerve_rings by being color-based, not
  a structural arc.

- brushfield_like_spots: many tiny light specks near the periphery; not a continuous band.
  IN STRIP → scattered pale or bright tiny dots concentrated in the LOWER ROWS
  (approximately R9–R11, bottom ~75 px of the strip content). They do not merge.

- nerve_rings: concentric stress arcs/rings (structural, not just color).
  IN STRIP → one or more HORIZONTAL arc(s) or faint bands running LEFT-TO-RIGHT
  across the full strip width, parallel to the top/bottom edges; they appear as subtle
  darker or lighter lines within the fiber texture at specific ring depths.

- lymphatic_rosary: chain/arc of pale nodules near the outer zone (discrete, not merged).
  IN STRIP → a HORIZONTAL chain of discrete rounded pale nodules in the R10 row area
  (approximately 250–275 px from the top); individual bead-like structures with small
  visible gaps between them — not a continuous pale band.

- scurf_rim: dark peripheral rim at ring 11 (outermost ring).
  IN STRIP → a dark HORIZONTAL band running along the BOTTOM EDGE of the strip content
  (the R11 row, approximately the lowest 25 px of the 300 px iris content).

- sodium_ring: pale/milky peripheral ring or band (NOT dark); may obscure iris detail.
  IN STRIP → a pale, milky, or off-white HORIZONTAL band near the BOTTOM ROWS
  (R9–R11, approximately the lowest 75 px of the strip content); lighter than surrounding
  iris texture; may partially hide fiber structure beneath it.

OUTPUT_JSON ONLY:
{
  "imgId": "${step1.imgId || ''}",
  "side": "${side}",
  "global": {
    "constitution": "LYM|HEM|BIL|unclear",
    "disposition": "SILK|LINEN|BURLAP|unclear",
    "diathesis": [{ "code": "HAC|LRS|LIP|DYS", "confidence": 0.0 }]
  },
  "collarette": {
    "ANW_status": "expanded|contracted|broken|normal|unclear",
    "minuteRange": [0, 59],
    "ringRange": [2, 3],
    "confidence": 0.0
  },
  "findings": [
    {
      "type": "pigment_spot|pigment_cloud|pigment_band|brushfield_like_spots|nerve_rings|lymphatic_rosary|scurf_rim|sodium_ring",
      "subtype": "(pigment_spot only) orange_rust|brown_black|yellow|other",
      "minuteRange": [0, 0],
      "ringRange": [0, 0],
      "severity": "low|medium|high",
      "notes": "<=60 chars describing appearance in the strip",
      "confidence": 0.0
    }
  ],
  "excluded": [
    { "type": "specular|eyelid_band", "minuteRange": [0, 0], "ringRange": [0, 0] }
  ]
}

FAILSAFE:
If GEO.ok != true: {"error":{"stage":"STEP2B","code":"PREREQ_FAIL","message":"geo not ok","canRetry":true}}
If cannot comply: {"error":{"stage":"STEP2B","code":"FORMAT_FAIL","message":"<reason>","canRetry":true}}`;
}

function promptStep2B_ANW(side, step1) {
  return `${IMAGE_FORMAT}

================================================================================
STEP2B_ANW_contour_profiler
================================================================================

ROLE: iris_ANW_contour_profiler_v9
MODE: image_parse_only
INPUT: unwrapped_iris_strip — rectangular image as described in IMAGE_FORMAT above
SIDE: ${side}
GEO: ${JSON.stringify(step1)}

PREREQ: If GEO.ok != true → return error JSON immediately.

TARGET: DETAILED ANW / COLLARETTE CONTOUR PROFILING — segment by segment.
This step ONLY examines the collarette (autonomic nerve wreath) — the wavy band
visible in rows R2–R3 (approximately 50–100 px from the top of the 300 px iris content).

IGNORE: White/blank eyelid-masked bands | glare/specular patches | GEO.invalidRegions
DO NOT report findings outside the R1–R4 range. Focus ONLY on the collarette region.

WHY THIS MATTERS:
In iridology the collarette is the single most important landmark. Its shape varies
LOCALLY — it may be expanded (pushed toward R3–R4) in one sector and contracted
(pulled toward R1) in another. Breaks, notches, and ballooning are significant per-sector.
A single global "expanded | contracted | broken | normal" label is insufficient.

SEGMENTATION — divide the strip into 12 equal segments of 5 minutes each:
  Segment  1: minutes  0– 5   (12:00–1:00)
  Segment  2: minutes  5–10   (1:00–2:00)
  Segment  3: minutes 10–15   (2:00–3:00)
  Segment  4: minutes 15–20   (3:00–4:00)
  Segment  5: minutes 20–25   (4:00–5:00)
  Segment  6: minutes 25–30   (5:00–6:00)
  Segment  7: minutes 30–35   (6:00–7:00)
  Segment  8: minutes 35–40   (7:00–8:00)
  Segment  9: minutes 40–45   (8:00–9:00)
  Segment 10: minutes 45–50   (9:00–10:00)
  Segment 11: minutes 50–55   (10:00–11:00)
  Segment 12: minutes 55–60   (11:00–12:00)

For EACH visible segment (not masked by white/eyelid):

1. POSITION — Where is the center of the collarette band in this segment?
   - "high" = band center near R2 (contracted toward pupil)
   - "mid"  = band center between R2 and R3 (normal position)
   - "low"  = band center near R3 or beyond (expanded toward periphery)
   Report the approximate ring position as a decimal (e.g., 2.3, 2.8, 3.1).

2. SHAPE — What is the local shape of the collarette?
   - "normal"     = smooth, continuous, well-defined wavy band
   - "expanded"   = band pushed outward (toward R3–R4), wider than average
   - "contracted" = band pulled inward (toward R1–R2), thinner than average
   - "broken"     = clear gap or discontinuity in the band
   - "notched"    = V-shaped indentation cutting into the band
   - "ballooning" = localized outward bulge (mushroom-shaped protrusion)
   - "flattened"  = band is present but indistinct, texture reduced

3. THICKNESS — Estimated band thickness:
   - "thin"   = less than ~15 px (< 0.6 rings)
   - "normal" = ~15–30 px (0.6–1.2 rings)
   - "thick"  = > 30 px (> 1.2 rings)

4. INTEGRITY — Is the collarette border sharp and well-defined?
   - "sharp"  = clear boundary between collarette and surrounding tissue
   - "fuzzy"  = gradual, blurred transition
   - "absent" = cannot distinguish collarette in this segment

5. If segment is fully masked (white/eyelid/glare) → set visible=false, skip other fields.

LOCALIZATION (CRITICAL):
- Read minute position from TOP tick labels.
- Read ring position from LEFT R-labels (R0 at top).
- DO NOT use atan2, radial distance, or circular geometry.

OUTPUT_JSON ONLY:
{
  "imgId": "${step1.imgId || ''}",
  "side": "${side}",
  "ANW_global_status": "expanded|contracted|broken|normal|mixed|unclear",
  "ANW_global_confidence": 0.0,
  "segments": [
    {
      "seg": 1,
      "minuteRange": [0, 5],
      "visible": true,
      "position": "high|mid|low",
      "ringCenter": 2.5,
      "shape": "normal|expanded|contracted|broken|notched|ballooning|flattened",
      "thickness": "thin|normal|thick",
      "integrity": "sharp|fuzzy|absent",
      "defects": [],
      "notes": "<=60 chars",
      "confidence": 0.0
    }
  ],
  "defects": [
    {
      "type": "break|notch|ballooning|lesion|pigment_on_ANW",
      "minuteRange": [0, 0],
      "ringRange": [2, 3],
      "notes": "<=60 chars",
      "confidence": 0.0
    }
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

FAILSAFE:
If GEO.ok != true: {"error":{"stage":"STEP2B_ANW","code":"PREREQ_FAIL","message":"geo not ok","canRetry":true}}
If cannot comply: {"error":{"stage":"STEP2B_ANW","code":"FORMAT_FAIL","message":"<reason>","canRetry":true}}`;
}

function promptStep2C(side, step1, step2a, step2b, step2anw) {
  return `IRIS PIPELINE — STEP2C_consistency_validator

ROLE: iris_consistency_validator_v9
MODE: strict_json_only

INPUTS:
  GEO    = ${JSON.stringify(step1)}
  STRUCT = ${JSON.stringify(step2a)}
  PIG    = ${JSON.stringify(step2b)}
  ANW_CONTOUR = ${JSON.stringify(step2anw)}

PREREQ:
- If GEO.ok != true → return error JSON.
- If STRUCT.error or PIG.error exists → return error JSON.
- ANW_CONTOUR.error is non-fatal: if present, use PIG.collarette as fallback for collarette data.

CONTEXT — COORDINATE SYSTEM:
All minuteRange values are X-axis positions in the UNWRAPPED STRIP (0=left edge=12 o'clock,
30=strip center=6 o'clock, 60=right edge=12 o'clock again). ringRange is R0(top)–R11(bottom).
There are no circles, no limbus, no radial distances in this coordinate system.

PURPOSE:
- Remove contradictions, duplicates, and range errors from STRUCT and PIG findings.
- Normalize all ranges (no wrap), clamp values, merge duplicates.
- Integrate detailed ANW contour data from ANW_CONTOUR into collarette_clean.
- Do NOT add new findings. Only keep / merge / split / drop existing ones.

ANW CONTOUR INTEGRATION:
- If ANW_CONTOUR is valid (no error), use its per-segment data to build collarette_clean:
  - ANW_status: use ANW_CONTOUR.ANW_global_status (prefer over PIG.collarette.ANW_status).
  - segments: pass through ANW_CONTOUR.segments (only visible ones).
  - defects: pass through ANW_CONTOUR.defects after range normalization.
  - contourSummary: pass through ANW_CONTOUR.contourSummary.
  - Cross-check: if PIG.collarette.ANW_status contradicts ANW_CONTOUR.ANW_global_status,
    prefer ANW_CONTOUR (it has per-segment detail) and add a warning.
- If ANW_CONTOUR has error: fall back to PIG.collarette for basic ANW_status only.
  Set segments=[], defects=[], contourSummary=null.

LIMITS:
- maxFindingsOut = 60. If > 60 after cleanup: keep highest confidence first.
- maxSplitsPerFinding = 2.

CONTRADICTION RULES:
1) scurf_rim vs sodium_ring: if minute overlap > 50% AND ring overlap > 50%:
   keep sodium_ring if notes imply light/milky; keep scurf_rim if notes imply dark rim.
   If undecidable: drop both unless one confidence ≥ 0.80.
2) pigment_spot vs lacuna/crypt: if they overlap → keep structural gap; drop pigment_spot.
3) lymphatic_rosary vs brushfield_like_spots: same area → keep rosary only if clearly
   a chain/arc of discrete nodules; otherwise keep brushfield.
4) Specular contamination: drop any finding that overlaps GEO.invalidRegions > 25%.
5) collarette_defect_lesion (from STRUCT) vs ANW_CONTOUR.defects: if same minuteRange,
   merge — keep the ANW_CONTOUR defect detail, combine notes.

DEDUP / MERGE:
- Same type + minute overlap > 60% AND ring overlap > 60% → merge ranges (union),
  confidence = max, notes = shorter.

RANGE NORMALIZATION:
- Clamp minutes 0..59, rings 0..11.
- minuteRange NEVER wraps: if start > end → split into [start, 59] + [0, end].
- ringRange: start ≤ end (swap if reversed).
- Point-like (width = 0): expand to ±1 unless confidence ≥ 0.85 AND focus = "good".
- Very wide (> 20 min OR > 3 rings) → confidence −= 0.15; drop if < 0.55.

GLOBAL CONSISTENCY:
- collarette_clean.ringRange MUST be [2, 3]. If inconsistent: clamp and confidence −= 0.20.
- GEO.usableUpperIris = false AND finding centered in minutes [57..59] or [0..3]:
  confidence −= 0.10.

OUTPUT_JSON ONLY:
{
  "imgId": "${step1.imgId || ''}",
  "side": "${side}",
  "findings_struct_clean": [
    { "type": "...", "minuteRange": [0, 0], "ringRange": [0, 0], "size": "xs|s|m|l", "notes": "<=60 chars", "confidence": 0.0 }
  ],
  "findings_pigment_clean": [
    { "type": "...", "subtype": "...", "minuteRange": [0, 0], "ringRange": [0, 0], "severity": "low|medium|high", "notes": "<=60 chars", "confidence": 0.0 }
  ],
  "collarette_clean": {
    "ANW_status": "expanded|contracted|broken|normal|mixed|unclear",
    "minuteRange": [0, 59],
    "ringRange": [2, 3],
    "confidence": 0.0,
    "segments": [
      {
        "seg": 1,
        "minuteRange": [0, 5],
        "visible": true,
        "position": "high|mid|low",
        "ringCenter": 2.5,
        "shape": "normal|expanded|contracted|broken|notched|ballooning|flattened",
        "thickness": "thin|normal|thick",
        "integrity": "sharp|fuzzy|absent",
        "confidence": 0.0
      }
    ],
    "defects": [
      {
        "type": "break|notch|ballooning|lesion|pigment_on_ANW",
        "minuteRange": [0, 0],
        "ringRange": [2, 3],
        "notes": "<=60 chars",
        "confidence": 0.0
      }
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
  },
  "global_clean": {
    "constitution": "LYM|HEM|BIL|unclear",
    "disposition": "SILK|LINEN|BURLAP|unclear",
    "diathesis": [{ "code": "HAC|LRS|LIP|DYS", "confidence": 0.0 }]
  },
  "dropped": [
    { "type": "...", "reason": "contradiction|specular|eyelid_band|too_wide|low_confidence|duplicate" }
  ],
  "warnings": ["<=60 chars"]
}

FAILSAFE:
{"error":{"stage":"STEP2C","code":"PREREQ_FAIL|FORMAT_FAIL","message":"<reason>","canRetry":true}}`;
}

// Minimal v9 iridology coordinate map — used by STEP3 for zone matching.
// Zone fields: id, side ("R"|"L"|"ANY"), mins [start, end], rings [start, end],
//              organ_bg (Bulgarian), system_bg (Bulgarian).
const MAP_V9 = [
  // ── Bilateral / ANY side ──────────────────────────────────────────
  { id: 'ANY-stomach',      side: 'ANY', mins: [0,  59], rings: [1,  1],  organ_bg: 'Стомах',                     system_bg: 'Храносмилателна' },
  { id: 'ANY-sm-intest',    side: 'ANY', mins: [0,  59], rings: [3,  4],  organ_bg: 'Тънко черво',                 system_bg: 'Храносмилателна' },
  { id: 'ANY-ANW',          side: 'ANY', mins: [0,  59], rings: [2,  3],  organ_bg: 'Автономна нервна система',   system_bg: 'Нервна' },
  { id: 'ANY-LYM',          side: 'ANY', mins: [0,  59], rings: [10, 10], organ_bg: 'Лимфна система',             system_bg: 'Имунна' },
  { id: 'ANY-SCU',          side: 'ANY', mins: [0,  59], rings: [11, 11], organ_bg: 'Кожа / Детоксикация',        system_bg: 'Детоксикация' },
  { id: 'ANY-spine-cerv-u', side: 'ANY', mins: [56, 59], rings: [8, 10],  organ_bg: 'Гръбначен стълб (шиен)',     system_bg: 'Опорно-двигателна' },
  { id: 'ANY-spine-cerv-l', side: 'ANY', mins: [0,  4],  rings: [8, 10],  organ_bg: 'Гръбначен стълб (шиен)',     system_bg: 'Опорно-двигателна' },
  // ── RIGHT EYE ─────────────────────────────────────────────────────
  { id: 'R-brain-motor',    side: 'R',   mins: [0,  3],  rings: [4,  9],  organ_bg: 'Мозък (моторни зони)',       system_bg: 'Нервна' },
  { id: 'R-brain-sens',     side: 'R',   mins: [57, 59], rings: [4,  9],  organ_bg: 'Мозък (сетивни зони)',       system_bg: 'Нервна' },
  { id: 'R-sinus',          side: 'R',   mins: [2,  6],  rings: [4,  7],  organ_bg: 'Синуси (десни)',             system_bg: 'Дихателна' },
  { id: 'R-larynx',         side: 'R',   mins: [5,  10], rings: [4,  5],  organ_bg: 'Ларинкс / Гърло',           system_bg: 'Дихателна' },
  { id: 'R-thyroid',        side: 'R',   mins: [7,  13], rings: [4,  6],  organ_bg: 'Щитовидна жлеза (дясна)',   system_bg: 'Ендокринна' },
  { id: 'R-eye-ear',        side: 'R',   mins: [10, 17], rings: [4,  7],  organ_bg: 'Ухо / Очи',                 system_bg: 'Нервна' },
  { id: 'R-bronchi',         side: 'R',   mins: [14, 19], rings: [4,  6],  organ_bg: 'Бронхи (десни)',             system_bg: 'Дихателна' },
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
  // ── LEFT EYE ──────────────────────────────────────────────────────
  { id: 'L-brain-motor',    side: 'L',   mins: [57, 59], rings: [4,  9],  organ_bg: 'Мозък (моторни зони)',       system_bg: 'Нервна' },
  { id: 'L-brain-sens',     side: 'L',   mins: [0,  3],  rings: [4,  9],  organ_bg: 'Мозък (сетивни зони)',       system_bg: 'Нервна' },
  { id: 'L-sinus',          side: 'L',   mins: [54, 59], rings: [4,  7],  organ_bg: 'Синуси (леви)',              system_bg: 'Дихателна' },
  { id: 'L-larynx',         side: 'L',   mins: [50, 55], rings: [4,  5],  organ_bg: 'Ларинкс / Гърло',           system_bg: 'Дихателна' },
  { id: 'L-thyroid',        side: 'L',   mins: [47, 53], rings: [4,  6],  organ_bg: 'Щитовидна жлеза (лява)',    system_bg: 'Ендокринна' },
  { id: 'L-eye-ear',        side: 'L',   mins: [43, 50], rings: [4,  7],  organ_bg: 'Ухо / Очи',                 system_bg: 'Нервна' },
  { id: 'L-bronchi',         side: 'L',   mins: [41, 46], rings: [4,  6],  organ_bg: 'Бронхи (леви)',             system_bg: 'Дихателна' },
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

function promptStep3(side, step1, step2c) {
  return `IRIS PIPELINE — STEP3_mapper_v9

ROLE: iris_mapper_v9
MODE: strict_json_only

INPUTS:
  GEO   = ${JSON.stringify(step1)}
  CLEAN = ${JSON.stringify(step2c)}
  MAP_V9 = ${JSON.stringify(MAP_V9)}
  SIDE  = ${side}

PREREQ: If GEO.ok != true OR CLEAN.error exists → return error JSON.

GOAL: Attach each finding from CLEAN to the most specific matching MAP_V9 zone
using (side + minuteRange + ringRange).

COORDINATE REMINDER:
All minuteRange / ringRange values in CLEAN refer to the UNWRAPPED STRIP coordinate system:
  minute = X-axis 0–59 (left=0=12 o'clock, right≈60=12 o'clock again)
  ring   = Y-axis R0–R11 (top=innermost, bottom=outermost)
Zone matching uses numeric range overlap only — no circular geometry.

MATCH RULE: A zone matches a finding if:
- zone.side == SIDE or zone.side == "ANY"
- finding.minuteRange overlaps zone.mins  (i.e., max(starts) ≤ min(ends))
- finding.ringRange overlaps zone.rings

TIE-BREAK (when multiple zones match):
1. Prefer zones with explicit rings over zones with wide ring ranges.
2. Prefer side-specific (R or L) over ANY.
3. Prefer smallest zone area (mins span × rings span).

UNMAPPED WARNING: If > 60% of findings have no matching zone → add warning
"possible side or geo mismatch — check strip orientation".

OUTPUT_JSON ONLY:
{
  "imgId": "${step1.imgId || ''}",
  "side": "${side}",
  "mappedFindings": [
    {
      "fid": "F1",
      "type": "...",
      "minuteRange": [0, 0],
      "ringRange": [0, 0],
      "confidence": 0.0,
      "zone": { "id": "...", "organ_bg": "...", "system_bg": "..." },
      "mapped": true
    }
  ],
  "zoneSummary": [
    { "zoneId": "...", "organ_bg": "...", "system_bg": "...", "evidenceCount": 0, "topTypes": ["type:count"] }
  ],
  "unmapped": [
    { "fid": "F9", "type": "...", "reason": "no MAP_V9 match by min+ring" }
  ],
  "warnings": ["<=60 chars"]
}

FAILSAFE:
{"error":{"stage":"STEP3","code":"PREREQ_FAIL|FORMAT_FAIL","message":"<reason>","canRetry":true}}`;
}

function promptStep4(side, step3, step2c) {
  return `IRIS PIPELINE — STEP4_profile_builder

ROLE: iris_rag_profile_builder_v2
MODE: strict_json_only

INPUTS:
  MAPPED = ${JSON.stringify(step3)}
  CLEAN  = ${JSON.stringify(step2c)}

PREREQ: If MAPPED.error or CLEAN.error exists → return error JSON.

SCOPE: Preventive health profile only — NOT a medical diagnosis. No organ/disease names
beyond what is in MAPPED.zoneSummary.

BUILD LOGIC:
1) Base constitution/disposition/diathesis from CLEAN.global_clean.
2) ANW_status from CLEAN.collarette_clean.ANW_status.
3) ANW CONTOUR DETAIL: If CLEAN.collarette_clean.segments exists and is non-empty:
   - Use per-segment data to assess gut_ANW channel more precisely.
   - Expanded segments near organ zones → flag those organs for digestive involvement.
   - Contracted segments → may indicate spastic tendency in corresponding sector.
   - Broken/notched segments → flag as potential weak points in autonomic regulation.
   - Use contourSummary.expandedSegments / contractedSegments / brokenSegments.
   - Factor contourSummary.overallIntegrity into digestive axis score.
4) Elimination channel priority: gut_ANW → kidney_6 → lymph → skin_scu.
5) Axes: stress (from nerve/brain zones), digestive (ANW + gut zones), immune (LYM + SCU zones).
6) Hypotheses: each must cite at least one fid from MAPPED.mappedFindings + its zoneId.
   ANW contour defects can also serve as evidence (cite segment number).

OUTPUT_JSON ONLY:
{
  "imgId": "${step3.imgId || ''}",
  "side": "${side}",
  "base": {
    "constitution": "LYM|HEM|BIL|unclear",
    "disposition": "SILK|LINEN|BURLAP|unclear",
    "diathesis": ["HAC", "LRS", "LIP", "DYS"],
    "ANW_status": "expanded|contracted|broken|normal|mixed|unclear"
  },
  "ANW_profile": {
    "overallIntegrity": "good|moderate|poor",
    "expandedSectors": "list of clock positions where ANW is expanded",
    "contractedSectors": "list of clock positions where ANW is contracted",
    "brokenSectors": "list of clock positions where ANW has breaks",
    "clinicalNote": "<=120 chars: brief interpretation of ANW contour pattern"
  },
  "axesScore": { "stress0_100": 0, "digestive0_100": 0, "immune0_100": 0 },
  "elimChannels": [
    { "channel": "gut_ANW",  "status": "normal|attention|concern", "evidence": [{ "fid": "F1", "zoneId": "..." }] },
    { "channel": "kidney_6", "status": "normal|attention|concern", "evidence": [] },
    { "channel": "lymph",    "status": "normal|attention|concern", "evidence": [] },
    { "channel": "skin_scu", "status": "normal|attention|concern", "evidence": [] }
  ],
  "hypotheses": [
    {
      "title": "<=60 chars",
      "claim": "short preventive claim",
      "evidence": [{ "fid": "F1", "zoneId": "..." }],
      "confidence0_1": 0.0,
      "applicability": "when/if condition"
    }
  ],
  "verificationQuestions": ["short question 1", "short question 2"]
}

FAILSAFE:
{"error":{"stage":"STEP4","code":"PREREQ_FAIL|FORMAT_FAIL","message":"<reason>","canRetry":true}}`;
}

function promptStep5(side, step1, step2c, step3, step4, questionnaire) {
  const q = questionnaire || {};
  return `IRIS PIPELINE — STEP5_frontend_report_bg

ROLE: iris_frontend_report_generator_bg_v9
MODE: strict_json_only

INPUTS:
  GEO          = ${JSON.stringify(step1)}
  CLEAN        = ${JSON.stringify(step2c)}
  MAPPED       = ${JSON.stringify(step3)}
  PROFILE      = ${JSON.stringify(step4)}
  QUESTIONNAIRE = ${JSON.stringify(q)}

PREREQ: If any input contains error OR GEO.ok != true → return error JSON.

CORE TRUTH RULE (NO DRIFT):
- ORGAN and SYSTEM MUST come ONLY from MAPPED.zoneSummary and MAPPED.mappedFindings.
- The 12 UI zones below are DISPLAY BUCKETS ONLY, not a topography source.

VALIDATION PRIORITY (cross-reference with QUESTIONNAIRE):
  1) ВИСОК — потвърдено от въпросника (highest weight)
  2) СРЕДЕН — не е споменато (preventive, medium weight)
  3) НИСЪК — противоречи (flag it, do not emphasize)

OUTPUT CONSTRAINTS:
  - JSON ONLY | UI texts in BULGARIAN | No markdown | No newlines inside strings
  - No double quotes inside string values | zones[].findings ≤ 60 chars
  - artifacts[].description ≤ 60 chars | systemScores[].description ≤ 60 chars
  - advice bullets ≤ 120 chars | severity: low|medium|high | angle always 0..360

MINUTE-TO-ZONE CONVERSION (for mapping findings to the 12 UI zones):
  Findings carry minuteRange (strip X-axis, 0–59). UI zones use degrees (0–360).
  Conversion: degrees = minute × 6. Zone assigned by center minute of finding.

  Zone  1 ("12-1ч"):  minutes  0– 5  → degrees   0°– 30°   (12:00–1:00)
  Zone  2 ("1-2ч"):   minutes  5–10  → degrees  30°– 60°   (1:00–2:00)
  Zone  3 ("2-3ч"):   minutes 10–15  → degrees  60°– 90°   (2:00–3:00)
  Zone  4 ("3-4ч"):   minutes 15–20  → degrees  90°–120°   (3:00–4:00)
  Zone  5 ("4-5ч"):   minutes 20–25  → degrees 120°–150°   (4:00–5:00)
  Zone  6 ("5-6ч"):   minutes 25–30  → degrees 150°–180°   (5:00–6:00)
  Zone  7 ("6-7ч"):   minutes 30–35  → degrees 180°–210°   (6:00–7:00)
  Zone  8 ("7-8ч"):   minutes 35–40  → degrees 210°–240°   (7:00–8:00)
  Zone  9 ("8-9ч"):   minutes 40–45  → degrees 240°–270°   (8:00–9:00)
  Zone 10 ("9-10ч"):  minutes 45–50  → degrees 270°–300°   (9:00–10:00)
  Zone 11 ("10-11ч"): minutes 50–55  → degrees 300°–330°  (10:00–11:00)
  Zone 12 ("11-12ч"): minutes 55–60  → degrees 330°–360°  (11:00–12:00)

  NOTE: minutes 0 and 60 are the same 12 o'clock position. A finding spanning
  [57, 3] must be split: [57,59] → Zone 12, [0,3] → Zone 1.

FIXED 12 UI ZONES (always output all 12):
  1:[0,30]  2:[30,60]  3:[60,90]  4:[90,120]  5:[120,150]  6:[150,180]
  7:[180,210] 8:[210,240] 9:[240,270] 10:[270,300] 11:[300,330] 12:[330,360]

ZONE NAMES (Bulgarian, stable):
  1"12-1ч" 2"1-2ч" 3"2-3ч" 4"3-4ч" 5"4-5ч" 6"5-6ч"
  7"6-7ч"  8"7-8ч" 9"8-9ч" 10"9-10ч" 11"10-11ч" 12"11-12ч"

HOW TO FILL EACH ZONE:
- Collect MAPPED findings whose center minute (= (minuteRange[0]+minuteRange[1])/2)
  falls within the zone's minute range (from the table above).
- Convert center minute to degrees (× 6) to confirm the angle range.
- Determine dominant organ/system by:
  weight = confidence + structural_severity_bonus(lacuna=+0.1, crypt=+0.15, atrophic=+0.1, cleft=+0.15)
- organ: short Bulgarian name from organ_bg (or "орган1; орган2" if two equally dominant)
- status:
    concern   = strong evidence + HIGH questionnaire validation OR severe structural signs
    attention = medium evidence OR preventive only
    normal    = little or no evidence for this zone
- findings: brief Bulgarian summary ≤ 60 chars

ARTIFACTS (2–5 strongest from CLEAN + MAPPED):
  Types to prioritize: lacuna, crypt, radial_furrow, deep_radial_cleft, nerve_rings,
                       pigment_spot, sodium_ring, scurf_rim, lymphatic_rosary,
                       ANW contour defects (break, notch, ballooning)
  location format: clock string using 5-min = 1-hour mapping:
    minute 0→"12:00", 5→"1:00", 10→"2:00", 15→"3:00", 20→"4:00", 25→"5:00",
    30→"6:00", 35→"7:00", 40→"8:00", 45→"9:00", 50→"10:00", 55→"11:00"
    Example: minuteRange [15,20] → "3:00-4:00"
  description: brief Bulgarian ≤ 60 chars, correlated with questionnaire if possible

COLLARETTE PROFILE (from CLEAN.collarette_clean + PROFILE.ANW_profile):
  If CLEAN.collarette_clean.segments is available and non-empty, build collaretteProfile:
  - status: CLEAN.collarette_clean.ANW_status (Bulgarian label)
  - integrity: PROFILE.ANW_profile.overallIntegrity or CLEAN.collarette_clean.contourSummary.overallIntegrity
  - For each of the 12 segments: report shape and position in Bulgarian
  - defects: list notable ANW defects (breaks, notches, ballooning) with clock location
  - clinicalNote: brief Bulgarian interpretation ≤ 120 chars
  If segments data is not available, output collaretteProfile with just status and a note.

SYSTEM SCORES (always output exactly 6):
  Храносмилателна | Имунна | Нервна | Сърдечно-съдова | Детоксикация | Ендокринна
  score 0–100 based on PROFILE.axesScore + weighted evidence from MAPPED zones
  description ≤ 60 chars

ADVICE (Bulgarian, all values ≤ 120 chars, no newlines, no double quotes):
  priorities: 3–6 bullets | nutrition.focus: 3–6 | nutrition.limit: 3–6
  lifestyle.sleep: 2–4 | lifestyle.stress: 2–4 | lifestyle.activity: 2–4
  followUp: 2–5 bullets

OUTPUT_JSON ONLY:
{
  "analysis": {
    "zones": [
      {"id":1,"name":"12-1ч","organ":"...","status":"normal|attention|concern","findings":"<=60","angle":[0,30]},
      {"id":2,"name":"1-2ч","organ":"...","status":"normal|attention|concern","findings":"<=60","angle":[30,60]},
      {"id":3,"name":"2-3ч","organ":"...","status":"normal|attention|concern","findings":"<=60","angle":[60,90]},
      {"id":4,"name":"3-4ч","organ":"...","status":"normal|attention|concern","findings":"<=60","angle":[90,120]},
      {"id":5,"name":"4-5ч","organ":"...","status":"normal|attention|concern","findings":"<=60","angle":[120,150]},
      {"id":6,"name":"5-6ч","organ":"...","status":"normal|attention|concern","findings":"<=60","angle":[150,180]},
      {"id":7,"name":"6-7ч","organ":"...","status":"normal|attention|concern","findings":"<=60","angle":[180,210]},
      {"id":8,"name":"7-8ч","organ":"...","status":"normal|attention|concern","findings":"<=60","angle":[210,240]},
      {"id":9,"name":"8-9ч","organ":"...","status":"normal|attention|concern","findings":"<=60","angle":[240,270]},
      {"id":10,"name":"9-10ч","organ":"...","status":"normal|attention|concern","findings":"<=60","angle":[270,300]},
      {"id":11,"name":"10-11ч","organ":"...","status":"normal|attention|concern","findings":"<=60","angle":[300,330]},
      {"id":12,"name":"11-12ч","organ":"...","status":"normal|attention|concern","findings":"<=60","angle":[330,360]}
    ],
    "artifacts": [
      {"type":"тип_БГ","location":"3:00-4:00","description":"<=60","severity":"low|medium|high"}
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
      "clinicalNote": "<=120 chars кратка интерпретация на контура на колартата"
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
    "priorities": ["..."],
    "nutrition": {"focus": ["..."], "limit": ["..."]},
    "lifestyle": {"sleep": ["..."], "stress": ["..."], "activity": ["..."]},
    "followUp": ["..."]
  }
}

FAILSAFE:
{"error":{"stage":"STEP5","code":"PREREQ_FAIL|FORMAT_FAIL","message":"<reason>","canRetry":true}}`;
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
