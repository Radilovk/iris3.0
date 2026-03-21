/**
 * worker.js — Cloudflare Worker
 * Iris Iridology Analysis Pipeline v9
 *
 * Architecture:
 *   1. POST /analyze  — receive unwrapped iris strip + metadata, run 5-step AI pipeline
 *   2. GET  /result/:side/:hash — retrieve a cached analysis result
 *
 * Environment bindings (set in wrangler.toml / Cloudflare dashboard):
 *   IRIS_KV      — KV namespace for caching results
 *   AI_API_KEY   — secret: OpenAI-compatible API key
 *   AI_MODEL     — var: model name, e.g. "gpt-4o"  (default: "gpt-4o")
 *   AI_BASE_URL  — var: API base URL (default: "https://api.openai.com/v1")
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
      return jsonResp({ error: 'Not Found' }, 404);
    } catch (err) {
      const msg = err?.message || String(err);
      return jsonResp({ error: 'Internal server error', detail: msg.slice(0, ERR_MSG_LIMIT) }, 500);
    }
  },
};

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

  if (!stripB64) {
    return jsonResp({ error: 'strip_image is required (base64 JPEG of the unwrapped iris strip from app.py)' }, 400);
  }
  if (side !== 'R' && side !== 'L') {
    return jsonResp({ error: 'side must be "R" or "L"' }, 400);
  }

  // KV cache lookup
  const cacheKey = `result:${side}:${imageHash}`;
  let cached = null;
  try {
    cached = await env.IRIS_KV.get(cacheKey, 'json');
  } catch (kvErr) {
    // KV read failure is non-fatal; proceed to run the pipeline
    console.error('KV get error:', kvErr?.message || kvErr);
  }
  if (cached) {
    return jsonResp({ cached: true, imageHash, side, result: cached });
  }

  // Run pipeline
  const pipeline = new IrisPipeline(env, stripB64, side, imageHash, questionnaire);
  const result = await pipeline.run();

  // Store in KV with 24-hour TTL
  await env.IRIS_KV.put(cacheKey, JSON.stringify(result), { expirationTtl: 86400 }).catch(() => {});

  return jsonResp({ cached: false, imageHash, side, result });
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

    // STEP 2A + 2B — parallel structural + pigment detection (both need image)
    const [step2a, step2b] = await Promise.all([
      this.visionCall(promptStep2A(this.side, step1)),
      this.visionCall(promptStep2B(this.side, step1)),
    ]);

    // STEP 2C — consistency validator (text only)
    const step2c = await this.textCall(promptStep2C(this.side, step1, step2a, step2b));

    // STEP 3 — mapper (text only)
    const step3 = await this.textCall(promptStep3(this.side, step1, step2c));

    // STEP 4 — profile builder (text only)
    const step4 = await this.textCall(promptStep4(this.side, step3, step2c));

    // STEP 5 — final Bulgarian report (text only)
    const step5 = await this.textCall(
      promptStep5(this.side, step1, step2c, step3, step4, this.questionnaire)
    );

    return {
      imageHash: this.imageHash,
      side: this.side,
      pipeline: { step1, step2a, step2b, step2c, step3, step4 },
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
// AI CALL — OpenAI-compatible vision/text endpoint
// =====================================================================
async function aiCall(env, prompt, imageDataUrl) {
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

function promptStep2C(side, step1, step2a, step2b) {
  return `IRIS PIPELINE — STEP2C_consistency_validator

ROLE: iris_consistency_validator_v9
MODE: strict_json_only

INPUTS:
  GEO    = ${JSON.stringify(step1)}
  STRUCT = ${JSON.stringify(step2a)}
  PIG    = ${JSON.stringify(step2b)}

PREREQ:
- If GEO.ok != true → return error JSON.
- If STRUCT.error or PIG.error exists → return error JSON.

CONTEXT — COORDINATE SYSTEM:
All minuteRange values are X-axis positions in the UNWRAPPED STRIP (0=left edge=12 o'clock,
30=strip center=6 o'clock, 60=right edge=12 o'clock again). ringRange is R0(top)–R11(bottom).
There are no circles, no limbus, no radial distances in this coordinate system.

PURPOSE:
- Remove contradictions, duplicates, and range errors from STRUCT and PIG findings.
- Normalize all ranges (no wrap), clamp values, merge duplicates.
- Do NOT add new findings. Only keep / merge / split / drop existing ones.

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
    "ANW_status": "expanded|contracted|broken|normal|unclear",
    "minuteRange": [0, 59],
    "ringRange": [2, 3],
    "confidence": 0.0
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
  { id: 'ANY-stomach',   side: 'ANY', mins: [0,  59], rings: [1,  1],  organ_bg: 'Стомах',                     system_bg: 'Храносмилателна' },
  { id: 'ANY-ANW',       side: 'ANY', mins: [0,  59], rings: [2,  3],  organ_bg: 'Автономна нервна система',   system_bg: 'Нервна' },
  { id: 'ANY-LYM',       side: 'ANY', mins: [0,  59], rings: [10, 10], organ_bg: 'Лимфна система',             system_bg: 'Имунна' },
  { id: 'ANY-SCU',       side: 'ANY', mins: [0,  59], rings: [11, 11], organ_bg: 'Кожа / Детоксикация',        system_bg: 'Детоксикация' },
  // ── RIGHT EYE ─────────────────────────────────────────────────────
  { id: 'R-brain-motor', side: 'R',   mins: [0,  3],  rings: [4,  9],  organ_bg: 'Мозък (моторни зони)',       system_bg: 'Нервна' },
  { id: 'R-brain-sens',  side: 'R',   mins: [57, 59], rings: [4,  9],  organ_bg: 'Мозък (сетивни зони)',       system_bg: 'Нервна' },
  { id: 'R-sinus',       side: 'R',   mins: [2,  6],  rings: [4,  7],  organ_bg: 'Синуси (десни)',             system_bg: 'Дихателна' },
  { id: 'R-eye-ear',     side: 'R',   mins: [10, 17], rings: [4,  7],  organ_bg: 'Ухо / Очи',                 system_bg: 'Нервна' },
  { id: 'R-shoulder',    side: 'R',   mins: [18, 22], rings: [4,  9],  organ_bg: 'Рамо (дясно)',               system_bg: 'Опорно-двигателна' },
  { id: 'R-lung',        side: 'R',   mins: [20, 30], rings: [4,  7],  organ_bg: 'Бял дроб (десен)',           system_bg: 'Дихателна' },
  { id: 'R-liver',       side: 'R',   mins: [23, 35], rings: [4,  9],  organ_bg: 'Черен дроб',                 system_bg: 'Детоксикация' },
  { id: 'R-gallbladder', side: 'R',   mins: [27, 33], rings: [5,  8],  organ_bg: 'Жлъчен мехур',               system_bg: 'Детоксикация' },
  { id: 'R-colon-asc',   side: 'R',   mins: [28, 40], rings: [4,  9],  organ_bg: 'Дебело черво (възходящо)',   system_bg: 'Храносмилателна' },
  { id: 'R-kidney',      side: 'R',   mins: [38, 48], rings: [4,  9],  organ_bg: 'Бъбрек (десен)',             system_bg: 'Отделителна' },
  { id: 'R-adrenal',     side: 'R',   mins: [40, 46], rings: [5,  8],  organ_bg: 'Надбъбречна жлеза (дясна)', system_bg: 'Ендокринна' },
  { id: 'R-hip',         side: 'R',   mins: [43, 50], rings: [4,  9],  organ_bg: 'Тазобедрена става (дясна)', system_bg: 'Опорно-двигателна' },
  // ── LEFT EYE ──────────────────────────────────────────────────────
  { id: 'L-brain-motor', side: 'L',   mins: [57, 59], rings: [4,  9],  organ_bg: 'Мозък (моторни зони)',       system_bg: 'Нервна' },
  { id: 'L-brain-sens',  side: 'L',   mins: [0,  3],  rings: [4,  9],  organ_bg: 'Мозък (сетивни зони)',       system_bg: 'Нервна' },
  { id: 'L-sinus',       side: 'L',   mins: [54, 59], rings: [4,  7],  organ_bg: 'Синуси (леви)',              system_bg: 'Дихателна' },
  { id: 'L-eye-ear',     side: 'L',   mins: [43, 50], rings: [4,  7],  organ_bg: 'Ухо / Очи',                 system_bg: 'Нервна' },
  { id: 'L-shoulder',    side: 'L',   mins: [38, 42], rings: [4,  9],  organ_bg: 'Рамо (ляво)',                system_bg: 'Опорно-двигателна' },
  { id: 'L-lung',        side: 'L',   mins: [20, 30], rings: [4,  7],  organ_bg: 'Бял дроб (ляв)',             system_bg: 'Дихателна' },
  { id: 'L-heart',       side: 'L',   mins: [23, 35], rings: [4,  8],  organ_bg: 'Сърце',                      system_bg: 'Сърдечно-съдова' },
  { id: 'L-spleen',      side: 'L',   mins: [28, 35], rings: [5,  9],  organ_bg: 'Далак',                      system_bg: 'Имунна' },
  { id: 'L-colon-desc',  side: 'L',   mins: [28, 40], rings: [4,  9],  organ_bg: 'Дебело черво (низходящо)',   system_bg: 'Храносмилателна' },
  { id: 'L-kidney',      side: 'L',   mins: [12, 22], rings: [4,  9],  organ_bg: 'Бъбрек (ляв)',               system_bg: 'Отделителна' },
  { id: 'L-adrenal',     side: 'L',   mins: [14, 20], rings: [5,  8],  organ_bg: 'Надбъбречна жлеза (лява)',  system_bg: 'Ендокринна' },
  { id: 'L-hip',         side: 'L',   mins: [10, 17], rings: [4,  9],  organ_bg: 'Тазобедрена става (лява)',  system_bg: 'Опорно-двигателна' },
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
3) Elimination channel priority: gut_ANW → kidney_6 → lymph → skin_scu.
4) Axes: stress (from nerve/brain zones), digestive (ANW + gut zones), immune (LYM + SCU zones).
5) Hypotheses: each must cite at least one fid from MAPPED.mappedFindings + its zoneId.

OUTPUT_JSON ONLY:
{
  "imgId": "${step3.imgId || ''}",
  "side": "${side}",
  "base": {
    "constitution": "LYM|HEM|BIL|unclear",
    "disposition": "SILK|LINEN|BURLAP|unclear",
    "diathesis": ["HAC", "LRS", "LIP", "DYS"],
    "ANW_status": "expanded|contracted|broken|normal|unclear"
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
                       pigment_spot, sodium_ring, scurf_rim, lymphatic_rosary
  location format: clock string using 5-min = 1-hour mapping:
    minute 0→"12:00", 5→"1:00", 10→"2:00", 15→"3:00", 20→"4:00", 25→"5:00",
    30→"6:00", 35→"7:00", 40→"8:00", 45→"9:00", 50→"10:00", 55→"11:00"
    Example: minuteRange [15,20] → "3:00-4:00"
  description: brief Bulgarian ≤ 60 chars, correlated with questionnaire if possible

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
