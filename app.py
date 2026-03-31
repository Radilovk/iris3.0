import os
import json
import cv2
import numpy as np
import base64
import hashlib
import requests
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ==========================================
# CONFIGURATION
# ==========================================

# Maximum file size: 10 MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

# Cloudflare Worker URL for AI analysis - can be overridden via env variable
WORKER_URL = os.environ.get('IRIS_WORKER_URL', '')
# Optional: Set timeout for worker requests (in seconds)
WORKER_TIMEOUT = int(os.environ.get('IRIS_WORKER_TIMEOUT', '120'))

# Default AI configuration (can be overridden per-request)
DEFAULT_AI_PROVIDER = os.environ.get('AI_PROVIDER', '')
DEFAULT_AI_MODEL = os.environ.get('AI_MODEL', '')

# ==========================================
# 1. PREPROCESSING
# ==========================================
def preprocess_image(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

# ==========================================
# 2. PUPIL DETECTION
# ==========================================
def find_pupil(gray_img):
    blur = cv2.medianBlur(gray_img, 13)
    _, thresh = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_p = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        (x, y), r = cv2.minEnclosingCircle(cnt)
        if area > max_area:
            max_area = area
            best_p = (int(x), int(y), int(r))
    return best_p

# ==========================================
# 3. IRIS DETECTION
# ==========================================
def find_iris_outer_boundary(img, px, py, pr):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Range: 2.2x to 7.5x pupil radius
    min_search_r = int(pr * 2.2)
    max_search_r = int(pr * 7.5)
    max_dist = min(px, py, w - px, h - py)
    if max_search_r > max_dist:
        max_search_r = max_dist

    if min_search_r >= max_search_r:
        return int(pr * 3.5)

    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    best_r = min_search_r
    max_grad = -1

    # Step 2 pixels
    for r in range(min_search_r, max_search_r, 2):
        score = 0
        samples = 0
        angles = [0, 180, 15, -15, 165, 195]

        for deg in angles:
            rad = np.deg2rad(deg)
            nx = np.cos(rad)
            ny = np.sin(rad)

            x_in = int(px + (r - 3) * nx)
            y_in = int(py + (r - 3) * ny)
            x_out = int(px + (r + 3) * nx)
            y_out = int(py + (r + 3) * ny)

            if 0 <= x_in < w and 0 <= y_in < h and 0 <= x_out < w and 0 <= y_out < h:
                val_in = int(gray_blur[y_in, x_in])
                val_out = int(gray_blur[y_out, x_out])
                score += (val_out - val_in)
                samples += 1

        if samples > 0:
            avg = score / samples
            if avg > max_grad:
                max_grad = avg
                best_r = r

    if max_grad < 5:
        return int(pr * 4.0)
    return best_r

# ==========================================
# 4. EYELID MASKING (ГРАДИЕНТ + RANSAC КРИВИ)
# ==========================================
def _ransac_polyfit(x, y, deg=2, iters=450, thr=4.0, seed=0, min_inliers=40):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    n = len(x)
    if n < deg + 1:
        return None

    best_coef = None
    best_cnt = -1
    idx = np.arange(n)
    best_inl = None

    for _ in range(iters):
        sample = rng.choice(idx, size=deg + 1, replace=False)
        coef = np.polyfit(x[sample], y[sample], deg)
        y_hat = np.polyval(coef, x)
        inl = np.abs(y - y_hat) < thr
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_coef = coef
            best_inl = inl

    if best_coef is None:
        return None

    if best_cnt < min_inliers or best_inl is None:
        return np.polyfit(x, y, deg)

    return np.polyfit(x[best_inl], y[best_inl], deg)


def _eyelid_points_from_circle(gray, cx, cy, R, band_frac=0.35):
    """
    Точки за горен/долен клепач чрез максимум на |dI/dy| в тесни пояси близо до ръба на ириса.
    """
    h, w = gray.shape[:2]

    g = cv2.GaussianBlur(gray, (5, 5), 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    gy = np.abs(gy)

    x0 = int(max(0, cx - R))
    x1 = int(min(w - 1, cx + R))
    xs = np.arange(x0, x1 + 1)

    up = []
    lo = []
    upv = []
    lov = []

    for xi in xs:
        dx = xi - cx
        inside = R * R - dx * dx
        if inside <= 0:
            continue

        y_top = int(max(0, np.floor(cy - np.sqrt(inside))))
        y_bot = int(min(h - 1, np.ceil(cy + np.sqrt(inside))))

        # пояси за търсене (към ръба, не към центъра)
        y_upper_end = int(cy - (1.0 - band_frac) * R)
        y_upper_end = max(y_top + 6, min(int(cy) - 6, y_upper_end))

        y_lower_start = int(cy + (1.0 - band_frac) * R)
        y_lower_start = min(y_bot - 6, max(int(cy) + 6, y_lower_start))

        # горен клепач
        if y_upper_end > y_top + 8:
            col = gy[y_top:y_upper_end, xi]
            k = int(np.argmax(col))
            yu = y_top + k
            up.append((xi, yu))
            upv.append(float(col[k]))

        # долен клепач
        if y_bot > y_lower_start + 8:
            col = gy[y_lower_start:y_bot, xi]
            k = int(np.argmax(col))
            yl = y_lower_start + k
            lo.append((xi, yl))
            lov.append(float(col[k]))

    return (
        np.array(up, np.int32),
        np.array(lo, np.int32),
        np.array(upv, np.float32),
        np.array(lov, np.float32),
    )


def segment_eyelids_ransac(img, cx, cy, ir,
                          band_frac=0.35,
                          base_grad_thr=25.0,
                          ransac_thr=4.0,
                          max_cut_frac=0.25,
                          seed=0):
    """
    Маска (uint8):
    255 = видим ирис (без клепачите)
    0   = клепачи / извън ириса
    """
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    up, lo, upv, lov = _eyelid_points_from_circle(gray, cx, cy, ir, band_frac=band_frac)

    # ако няма достатъчно точки → само кръг
    if len(up) < 40 or len(lo) < 40:
        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, (int(cx), int(cy)), int(ir), 255, -1)
        return mask

    # динамичен праг за точки (да не хваща текстурата на ириса)
    thr_u = max(base_grad_thr, float(np.percentile(upv, 60))) if len(upv) else base_grad_thr
    thr_l = max(base_grad_thr, float(np.percentile(lov, 60))) if len(lov) else base_grad_thr

    up_f = up[upv >= thr_u] if len(upv) else up
    lo_f = lo[lov >= thr_l] if len(lov) else lo

    if len(up_f) < 30 or len(lo_f) < 30:
        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, (int(cx), int(cy)), int(ir), 255, -1)
        return mask

    coef_u = _ransac_polyfit(up_f[:, 0], up_f[:, 1], deg=2, iters=450, thr=ransac_thr, seed=seed, min_inliers=40)
    coef_l = _ransac_polyfit(lo_f[:, 0], lo_f[:, 1], deg=2, iters=450, thr=ransac_thr, seed=seed + 1, min_inliers=40)

    if coef_u is None or coef_l is None:
        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, (int(cx), int(cy)), int(ir), 255, -1)
        return mask

    xs = np.arange(w, dtype=np.float64)
    dx = xs - float(cx)
    inside = float(ir) * float(ir) - dx * dx
    valid = inside > 0

    y_top = np.full(w, -1e9, dtype=np.float64)
    y_bot = np.full(w,  1e9, dtype=np.float64)

    rt = np.sqrt(np.maximum(inside, 0))
    y_top[valid] = float(cy) - rt[valid]
    y_bot[valid] = float(cy) + rt[valid]

    yu = np.polyval(coef_u, xs)
    yl = np.polyval(coef_l, xs)

    # предпазител: не режем повече от max_cut_frac от радиуса
    max_cut = float(ir) * float(max_cut_frac)
    yu = np.minimum(np.maximum(yu, y_top), y_top + max_cut)
    yl = np.maximum(np.minimum(yl, y_bot), y_bot - max_cut)

    # ако е практически по ръба → приемаме, че няма клепач там
    yu = np.where((yu - y_top) < 5.0, y_top, yu)
    yl = np.where((y_bot - yl) < 5.0, y_bot, yl)

    # гаранция yu < yl
    yl = np.maximum(yl, yu + 1.0)

    Y, X = np.ogrid[:h, :w]
    circle = (X - float(cx)) ** 2 + (Y - float(cy)) ** 2 <= float(ir) ** 2
    between = (Y >= yu[np.newaxis, :]) & (Y <= yl[np.newaxis, :])

    mask = np.zeros((h, w), np.uint8)
    mask[circle & between] = 255
    return mask

# ==========================================
# 5. UNWRAP (VECTORIZED)
# ==========================================
def unwrap_iris_fast(img, px, py, pr, ir):
    h_out, w_out = 300, 1200

    theta = np.linspace(-np.pi/2, 1.5*np.pi, w_out).astype(np.float32)
    r_vals = np.linspace(pr, ir, h_out).astype(np.float32)

    theta_grid, r_grid = np.meshgrid(theta, r_vals)

    map_x = (px + r_grid * np.cos(theta_grid)).astype(np.float32)
    map_y = (py + r_grid * np.sin(theta_grid)).astype(np.float32)

    unwrapped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return unwrapped

# ==========================================
# 6. DRAW MAP
# ==========================================
def draw_ai_grid_map_expanded(unwrapped, side="R"):
    if unwrapped is None:
        return np.zeros((300, 1200, 3), np.uint8)
    unwrapped = unwrapped.astype(np.uint8)

    img_h, img_w = unwrapped.shape[:2]
    pt, pl, pb, pr_pad = 50, 60, 40, 20

    cw = img_w + pl + pr_pad
    ch = img_h + pt + pb

    canvas = np.ones((ch, cw, 3), dtype=np.uint8) * 255
    canvas[pt:pt+img_h, pl:pl+img_w] = unwrapped

    c_grid = (200, 200, 200)
    c_txt = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # minutes
    for m in range(0, 61, 5):
        x = pl + int(m * (img_w / 60.0))
        if x >= pl + img_w:
            x = pl + img_w - 1
        cv2.line(canvas, (x, pt), (x, pt+img_h), c_grid, 1)
        cv2.line(canvas, (x, pt-5), (x, pt), c_txt, 1)
        txt = f"{m}"
        (tw, th), _ = cv2.getTextSize(txt, font, 0.4, 1)
        cv2.putText(canvas, txt, (x - tw//2, pt - 10), font, 0.4, c_txt, 1)

    # rings
    for r in range(12):
        y = pt + int(r * (img_h / 12.0))
        cv2.line(canvas, (pl, y), (pl+img_w, y), (240, 240, 240), 1)
        cv2.line(canvas, (pl-5, y), (pl, y), c_txt, 1)
        cv2.putText(canvas, f"R{r}", (5, y + 15), font, 0.5, c_txt, 1)

    lbl = "RIGHT EYE" if side == "R" else "LEFT EYE"
    cv2.putText(canvas, lbl, (pl, ch - 10), font, 0.8, c_txt, 2)

    if side == "R":
        nm, tm = 45, 15
    else:
        nm, tm = 15, 45

    nx = pl + int(nm * (img_w / 60.0))
    tx = pl + int(tm * (img_w / 60.0))
    cv2.putText(canvas, "^ NASAL", (nx - 30, ch - 10), font, 0.5, (0, 0, 255), 1)
    cv2.putText(canvas, "^ TEMPORAL", (tx - 40, ch - 10), font, 0.5, (100, 100, 100), 1)

    return canvas


def draw_overlay(img, px, py, pr, ir):
    out = img.copy()
    cv2.circle(out, (px, py), pr, (0, 255, 0), 2)
    cv2.circle(out, (px, py), ir, (0, 255, 255), 2)
    cv2.line(out, (px, py-ir), (px, py+ir), (255, 255, 0), 1)
    cv2.line(out, (px-ir, py), (px+ir, py), (255, 255, 0), 1)
    return out

# ==========================================
# 7. HELPER: Call Cloudflare Worker for AI analysis
# ==========================================
def call_worker_analysis(strip_b64, side, questionnaire=None, ai_provider=None, ai_model=None):
    """
    Call the Cloudflare Worker to analyze the unwrapped iris strip.
    Returns the AI analysis result or None if worker is not configured.
    
    Args:
        strip_b64: Base64-encoded JPEG of the unwrapped iris strip
        side: 'R' or 'L' for right or left eye
        questionnaire: Optional dict with patient data
        ai_provider: Optional AI provider override ('gemini', 'openai', 'openai-compatible')
        ai_model: Optional AI model name override
    """
    if not WORKER_URL:
        return None

    try:
        # Generate image hash for caching
        image_hash = hashlib.sha256(strip_b64.encode()).hexdigest()[:12]

        # Prepare form data
        data = {
            'strip_image': strip_b64,
            'side': side,
            'image_hash': image_hash,
        }
        if questionnaire:
            data['questionnaire'] = json.dumps(questionnaire)
        
        # Add AI model configuration if provided
        if ai_provider or DEFAULT_AI_PROVIDER:
            data['ai_provider'] = ai_provider or DEFAULT_AI_PROVIDER
        if ai_model or DEFAULT_AI_MODEL:
            data['ai_model'] = ai_model or DEFAULT_AI_MODEL

        # Call the worker
        response = requests.post(
            f"{WORKER_URL}/analyze",
            data=data,
            timeout=WORKER_TIMEOUT
        )

        if response.status_code == 200:
            return response.json()
        else:
            # Sanitize error message - don't expose raw response details
            return {'error': f'AI analysis service error (status {response.status_code})'}

    except requests.exceptions.Timeout:
        return {'error': 'AI analysis timed out. Please try again.'}
    except requests.exceptions.ConnectionError:
        return {'error': 'Cannot connect to AI analysis service.'}
    except Exception:
        return {'error': 'AI analysis encountered an unexpected error.'}


# ==========================================
# 8. FLASK APP
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    """
    Process uploaded iris images: detect pupil/iris, unwrap, and optionally
    send to Cloudflare Worker for AI analysis.
    
    Supports images up to 10 MB in size. Large images are automatically handled.
    AI model can be specified via ai_provider and ai_model parameters.
    """
    results = {}

    # Parse optional questionnaire from request
    questionnaire = None
    q_raw = request.form.get('questionnaire')
    if q_raw:
        try:
            questionnaire = json.loads(q_raw)
        except (json.JSONDecodeError, TypeError):
            pass

    # Check if AI analysis is requested
    run_ai = request.form.get('run_ai', 'false').lower() == 'true'
    
    # Get optional AI model configuration
    ai_provider = request.form.get('ai_provider')
    ai_model = request.form.get('ai_model')

    for sc, key in [('R', 'image_right'), ('L', 'image_left')]:
        f = request.files.get(key)
        if not f or f.filename == '':
            continue

        arr = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        
        # Handle large images by resizing if necessary (max 4000px on any side)
        max_dimension = 4000
        h, w = img.shape[:2]
        if h > max_dimension or w > max_dimension:
            scale = min(max_dimension / w, max_dimension / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        proc = preprocess_image(img)
        pupil = find_pupil(proc)

        if pupil:
            px, py, pr = pupil
            ir = find_iris_outer_boundary(img, px, py, pr)

            # Eyelid masking with RANSAC
            mask = segment_eyelids_ransac(img, px, py, ir, band_frac=0.35, max_cut_frac=0.25)

            ovl = draw_overlay(img, px, py, pr, ir)

            # Unwrap image (polar to rectangular)
            unw = unwrap_iris_fast(img, px, py, pr, ir)

            # Unwrap mask
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            unw_mask = unwrap_iris_fast(mask_bgr, px, py, pr, ir)

            # Whiteout: where mask is black (<100), paint white
            gray_m = cv2.cvtColor(unw_mask, cv2.COLOR_BGR2GRAY)
            unw[gray_m < 100] = (255, 255, 255)

            # Draw the grid map with coordinate labels
            mapped = draw_ai_grid_map_expanded(unw, side=sc)

            # Encode images to base64
            b_ovl = base64.b64encode(cv2.imencode('.jpg', ovl)[1]).decode()
            b_map = base64.b64encode(cv2.imencode('.jpg', mapped)[1]).decode()

            result = {
                'found': True,
                'overlay': b_ovl,
                'mapped': b_map,
            }

            # Run AI analysis if requested and worker is configured
            if run_ai and WORKER_URL:
                ai_result = call_worker_analysis(b_map, sc, questionnaire, ai_provider, ai_model)
                if ai_result:
                    result['ai_analysis'] = ai_result

            results[sc] = result
        else:
            results[sc] = {'found': False, 'error': 'Pupil not found'}

    # Include worker status in response
    results['worker_configured'] = bool(WORKER_URL)

    return jsonify(results)


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Standalone endpoint to trigger AI analysis on already-processed images.
    Expects: strip_image (base64), side (R/L), questionnaire (optional JSON)
    Optional: ai_provider, ai_model for model selection
    """
    if not WORKER_URL:
        return jsonify({'error': 'AI analysis worker not configured. Set IRIS_WORKER_URL environment variable.'}), 503

    strip_b64 = request.form.get('strip_image')
    side = request.form.get('side', 'R').upper()

    if not strip_b64:
        return jsonify({'error': 'strip_image is required'}), 400
    if side not in ('R', 'L'):
        return jsonify({'error': 'side must be R or L'}), 400

    questionnaire = None
    q_raw = request.form.get('questionnaire')
    if q_raw:
        try:
            questionnaire = json.loads(q_raw)
        except (json.JSONDecodeError, TypeError):
            pass
    
    # Get optional AI model configuration
    ai_provider = request.form.get('ai_provider')
    ai_model = request.form.get('ai_model')

    result = call_worker_analysis(strip_b64, side, questionnaire, ai_provider, ai_model)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=5000)