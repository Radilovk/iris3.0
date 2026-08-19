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
# SHARED GEOMETRY SPEC
# ==========================================
# This module is the Python twin of iris-geometry.js. Both read the same spec so
# the two paths cannot drift apart silently — which is exactly what had happened
# before: the browser path had lost eyelid masking entirely while every AI prompt
# still assumed lid regions arrive as white bands.
_SPEC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'geometry-spec.json')
with open(_SPEC_PATH, 'r', encoding='utf-8') as _f:
    SPEC = json.load(_f)

UNWRAP_W = SPEC['unwrap']['width']
UNWRAP_H = SPEC['unwrap']['height']
RING_GROUPS = SPEC['ringGroups']
DEST_CONTENT_H = sum(g['destHeight'] for g in RING_GROUPS)
N_SECTORS = SPEC['sectors']

# Sub-ring reference lines drawn inside the two-ring bands.
RING_SUBLINES = {'ANW': [2, 3], 'ORG_IN': [4, 5], 'ORG_MID': [6, 7], 'ORG_OUT': [8, 9]}

REJECT_MESSAGES = {
    'PUPIL_NOT_FOUND': 'Зеницата не беше открита. Снимайте отблизо, с окото в центъра на кадъра.',
    'PUPIL_NOT_ROUND': 'Зеницата не се вижда достатъчно ясно. Избягвайте отблясъци и гледайте право в камерата.',
    'IRIS_NOT_FOUND': 'Границата на ириса не се различава. Нужна е по-добра, равномерна светлина.',
    'IRIS_TOO_SMALL': 'Окото е твърде малко в кадъра. Приближете камерата.',
    'IRIS_OUT_OF_FRAME': 'Окото не се побира в кадъра. Центрирайте го и снимайте отново.',
    'RATIO_IMPLAUSIBLE': 'Зеницата и ирисът не бяха разделени коректно. Опитайте при по-различна светлина.',
    'TOO_OCCLUDED': 'Клепачите закриват твърде голяма част от ириса. Отворете окото по-широко.',
    'PUPIL_NOT_CLEAR': 'Зеницата не се вижда достатъчно ясно — клепачът или миглите я закриват. '
                       'Направете нова снимка с широко отворено око и зеница, видима изцяло.',
}


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
# 2. PUPIL DETECTION (adaptive)
# ==========================================
def _dark_threshold(gray_img, frac, lo, hi):
    """Intensity below which `frac` of pixels fall, clamped to [lo, hi]."""
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256]).flatten()
    target = gray_img.size * frac
    cum = 0.0
    t = lo
    for v in range(256):
        cum += hist[v]
        if cum >= target:
            t = v
            break
    return int(min(hi, max(lo, t)))


def _best_dark_blob(blur, t, cfg):
    """Largest roundness-weighted dark blob below intensity `t`."""
    _, thresh = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY_INV)

    # Corneal reflection punches bright holes in the pupil; close them so the
    # pupil stays one blob instead of a ring of fragments.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg['closeKernel'], cfg['closeKernel']))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < cfg['minArea']:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim <= 0:
            continue
        circularity = (4.0 * np.pi * area) / (perim * perim)
        (x, y), r = cv2.minEnclosingCircle(cnt)
        # Rank by area weighted by roundness, so a large ragged shadow cannot beat
        # a smaller genuinely circular pupil.
        score = area * max(circularity, 0.01)
        if score > best_score:
            best_score = score
            best = {'x': int(x), 'y': int(y), 'r': int(r), 'circularity': float(circularity)}
    return best


def find_pupil_candidates(gray_img):
    """
    Competing pupil hypotheses, darkest first.

    A single adaptive threshold is not enough: on a dark brown iris the iris disc
    itself falls below the cut and gets returned as one huge 'pupil', which is
    exactly how the old fixed threshold of 40 corrupted every downstream
    coordinate. Emitting several candidates lets the caller keep the one whose
    pupil+iris pair actually validates.
    """
    cfg = SPEC['pupil']
    blur = cv2.medianBlur(gray_img, cfg['medianBlur'])

    candidates = []
    seen_thresholds = set()
    for frac in cfg['darkFractions']:
        t = _dark_threshold(blur, frac, cfg['minThreshold'], cfg['maxThreshold'])
        if t in seen_thresholds:
            continue
        seen_thresholds.add(t)
        cand = _best_dark_blob(blur, t, cfg)
        if cand and not any(abs(cand['r'] - c['r']) <= 2 for c in candidates):
            candidates.append(cand)
    return candidates


def find_pupil(gray_img):
    """Best single pupil estimate (darkest plausible blob)."""
    candidates = find_pupil_candidates(gray_img)
    return candidates[0] if candidates else None


# ==========================================
# 3. IRIS DETECTION
# ==========================================
def find_iris_outer_boundary(gray, px, py, pr):
    """
    Iris outer boundary by radial gradient near the horizontal meridians, where
    the lids rarely reach. Returns the measured gradient too, so callers can tell
    a real edge from a guess — the old version silently returned pupil_r * 4.0.
    """
    cfg = SPEC['iris']
    h, w = gray.shape[:2]

    min_search_r = int(pr * cfg['minRadiusFactor'])
    max_search_r = int(pr * cfg['maxRadiusFactor'])
    max_dist = int(min(px, py, w - px, h - py))
    if max_search_r > max_dist:
        max_search_r = max_dist

    if min_search_r >= max_search_r:
        return {'r': int(pr * 3.5), 'gradient': 0.0, 'confidence': 0.0}

    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    best_r = min_search_r
    max_grad = -1.0

    angles = [0, 180, 15, -15, 165, 195, 30, -30, 150, 210]
    for r in range(min_search_r, max_search_r, cfg['step']):
        score = 0.0
        samples = 0
        for deg in angles:
            rad = np.deg2rad(deg)
            nx, ny = np.cos(rad), np.sin(rad)
            x_in, y_in = int(px + (r - 3) * nx), int(py + (r - 3) * ny)
            x_out, y_out = int(px + (r + 3) * nx), int(py + (r + 3) * ny)
            if 0 <= x_in < w and 0 <= y_in < h and 0 <= x_out < w and 0 <= y_out < h:
                score += int(gray_blur[y_out, x_out]) - int(gray_blur[y_in, x_in])
                samples += 1
        if samples > 0:
            avg = score / samples
            if avg > max_grad:
                max_grad = avg
                best_r = r

    confidence = max(0.0, min(1.0, max_grad / cfg['goodGradient']))
    return {'r': int(best_r), 'gradient': float(max_grad), 'confidence': float(confidence)}


# ==========================================
# 4. EYELID CURVES (GRADIENT + RANSAC)
# ==========================================
def _ransac_polyfit(x, y, deg=2, iters=None, thr=None, seed=0, min_inliers=None):
    cfg = SPEC['eyelid']
    iters = cfg['ransacIters'] if iters is None else iters
    thr = cfg['ransacThreshold'] if thr is None else thr
    min_inliers = cfg['minPoints'] if min_inliers is None else min_inliers

    rng = np.random.default_rng(seed)
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    n = len(x)
    if n < deg + 1:
        return None

    best_coef = None
    best_cnt = -1
    best_inl = None
    idx = np.arange(n)

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


def _eyelid_points_from_circle(gray, cx, cy, R, pupil_r):
    """
    Candidate lid points: strongest |dI/dy| within the iris annulus above/below
    the pupil. Searching the full annulus (rather than a thin band near the rim)
    is what lets a deeply intruding lid be found at all — with the old narrow band
    a half-closed eye yielded no fit, so it passed the gate as fully visible.
    The pupil disc itself is excluded so its own strong edge cannot be mistaken
    for a lid.
    """
    margin = SPEC['eyelid']['pupilMarginPx']
    h, w = gray.shape[:2]

    g = cv2.GaussianBlur(gray, (5, 5), 0)
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))

    x0 = int(max(0, cx - R))
    x1 = int(min(w - 1, cx + R))
    xs = np.arange(x0, x1 + 1)

    up, lo, upv, lov = [], [], [], []

    for xi in xs:
        dx = xi - cx
        inside = R * R - dx * dx
        if inside <= 0:
            continue

        y_top = int(max(0, np.floor(cy - np.sqrt(inside))))
        y_bot = int(min(h - 1, np.ceil(cy + np.sqrt(inside))))

        y_upper_end = int(cy - pupil_r - margin)
        y_upper_end = max(y_top + 6, min(int(cy) - 6, y_upper_end))

        y_lower_start = int(cy + pupil_r + margin)
        y_lower_start = min(y_bot - 6, max(int(cy) + 6, y_lower_start))

        if y_upper_end > y_top + 8:
            col = gy[y_top:y_upper_end, xi]
            k = int(np.argmax(col))
            up.append((xi, y_top + k))
            upv.append(float(col[k]))

        if y_bot > y_lower_start + 8:
            col = gy[y_lower_start:y_bot, xi]
            k = int(np.argmax(col))
            lo.append((xi, y_lower_start + k))
            lov.append(float(col[k]))

    return (np.array(up, np.int32), np.array(lo, np.int32),
            np.array(upv, np.float32), np.array(lov, np.float32))


def fit_eyelids(gray, cx, cy, ir, pupil_r, seed=0):
    """Fit upper/lower lid curves. Returns (coef_upper, coef_lower); either may be None."""
    cfg = SPEC['eyelid']
    up, lo, upv, lov = _eyelid_points_from_circle(gray, cx, cy, ir, pupil_r)

    if len(up) < cfg['minPoints'] or len(lo) < cfg['minPoints']:
        return None, None

    thr_u = max(cfg['baseGradientThreshold'], float(np.percentile(upv, 60))) if len(upv) else cfg['baseGradientThreshold']
    thr_l = max(cfg['baseGradientThreshold'], float(np.percentile(lov, 60))) if len(lov) else cfg['baseGradientThreshold']

    up_f = up[upv >= thr_u] if len(upv) else up
    lo_f = lo[lov >= thr_l] if len(lov) else lo

    if len(up_f) < cfg['minFilteredPoints'] or len(lo_f) < cfg['minFilteredPoints']:
        return None, None

    coef_u = _ransac_polyfit(up_f[:, 0], up_f[:, 1], deg=2, seed=seed)
    coef_l = _ransac_polyfit(lo_f[:, 0], lo_f[:, 1], deg=2, seed=seed + 1)
    return coef_u, coef_l


def pupil_clearance_occlusion(gray, cx, cy, pr, ir):
    """
    Fraction of the ring just outside the pupil that does not look like iris.

    This does not depend on the eyelid fit, which is exactly why it exists: when a
    lid closes far enough to reach the pupil rim, the lid search band collapses and
    no curve can be fitted at all, so the lid-based occlusion measure reports zero
    and a nearly shut eye would sail through the gate. Here the iris appearance is
    sampled near the horizontal meridians (which lids practically never reach) and
    used as the reference, then compared against samples straight above and below
    the pupil, where a lid must be if it is covering it.
    """
    if ir <= pr + 8:
        return 1.0

    h, w = gray.shape[:2]
    g = cv2.GaussianBlur(gray, (5, 5), 0)

    def sample(deg, frac):
        rad = np.deg2rad(deg)
        r = pr + frac * (ir - pr)
        x = int(round(cx + r * np.cos(rad)))
        y = int(round(cy + r * np.sin(rad)))
        if 0 <= x < w and 0 <= y < h:
            return float(g[y, x])
        return None

    # Reference: the iris either side of the pupil, horizontally.
    ref_vals = []
    for deg in list(range(-32, 33, 4)) + list(range(148, 213, 4)):
        for frac in (0.20, 0.35, 0.50):
            v = sample(deg, frac)
            if v is not None:
                ref_vals.append(v)
    if len(ref_vals) < 12:
        return 1.0

    ref = float(np.median(ref_vals))
    mad = float(np.median(np.abs(np.array(ref_vals) - ref)))
    # Generous tolerance: normal irises vary a lot sector to sector, and a false
    # rejection costs the user a retake for nothing.
    tol = max(45.0, 3.0 * 1.4826 * mad)

    occluded = 0
    total = 0
    for deg in list(range(62, 119, 4)) + list(range(242, 299, 4)):
        vals = [sample(deg, f) for f in (0.15, 0.30, 0.45)]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        total += 1
        if abs(float(np.median(vals)) - ref) > tol:
            occluded += 1

    if total == 0:
        return 1.0
    return occluded / float(total)


def estimate_roll(coef_u, coef_l, cx, ir):
    """
    Camera/head roll from the canthi (eye corners), taken as the intersections of
    the two lid curves. This is what makes minute 0 mean *anatomical* up rather
    than "up in the photo" — without it a routine phone tilt rotated the whole
    clock and shifted every organ sector.
    """
    none = {'roll_rad': 0.0, 'corners': None, 'source': 'none'}
    if coef_u is None or coef_l is None:
        return none

    a = float(coef_u[0] - coef_l[0])
    b = float(coef_u[1] - coef_l[1])
    c = float(coef_u[2] - coef_l[2])

    if abs(a) < 1e-9:
        return none  # parallel curves: no pair of corners

    disc = b * b - 4 * a * c
    if disc <= 0:
        return none

    sq = float(np.sqrt(disc))
    x1, x2 = sorted(((-b - sq) / (2 * a), (-b + sq) / (2 * a)))

    # The corners must straddle the iris and sit outside it, otherwise the curves
    # crossed somewhere meaningless and the estimate is not trustworthy.
    if not (x1 < cx < x2):
        return none
    if (x2 - x1) < SPEC['roll']['minCornerSpanFactor'] * ir:
        return none

    y1 = float(np.polyval(coef_u, x1))
    y2 = float(np.polyval(coef_u, x2))
    if not (np.isfinite(y1) and np.isfinite(y2)):
        return none

    roll = float(np.arctan2(y2 - y1, x2 - x1))
    if abs(np.rad2deg(roll)) > SPEC['roll']['maxRollDeg']:
        return {'roll_rad': 0.0, 'corners': None, 'source': 'out_of_range'}

    return {'roll_rad': roll, 'corners': [[x1, y1], [x2, y2]], 'source': 'canthi'}


def build_mask(h, w, cx, cy, ir, coef_u, coef_l):
    """
    Binary mask: 255 = usable iris, 0 = lid-covered or outside the iris.

    Returns (mask, true_occluded_fraction). The mask deliberately clamps how much
    it will cut (maxCutFraction) so one bad RANSAC fit cannot destroy an otherwise
    good capture — but that clamp also means a genuinely half-closed eye would be
    under-masked and analysed as if it were fine. So the occlusion actually implied
    by the fitted lids is measured separately, unclamped, and that honest number is
    what the validity gate judges.
    """
    max_cut = float(ir) * SPEC['eyelid']['maxCutFraction']

    xs = np.arange(w, dtype=np.float64)
    dx = xs - float(cx)
    inside = float(ir) * float(ir) - dx * dx
    valid = inside > 0

    y_top = np.full(w, -1e9, dtype=np.float64)
    y_bot = np.full(w, 1e9, dtype=np.float64)
    rt = np.sqrt(np.maximum(inside, 0))
    y_top[valid] = float(cy) - rt[valid]
    y_bot[valid] = float(cy) + rt[valid]

    true_occluded = 0.0
    if coef_u is not None and coef_l is not None:
        yu_raw = np.polyval(coef_u, xs)
        yl_raw = np.polyval(coef_l, xs)

        # Honest occlusion, measured before clamping.
        span = np.maximum(y_bot - y_top, 0.0)
        visible = np.clip(np.minimum(y_bot, yl_raw) - np.maximum(y_top, yu_raw), 0.0, None)
        total_area = float(span[valid].sum())
        if total_area > 0:
            true_occluded = 1.0 - float(visible[valid].sum()) / total_area

        yu = np.minimum(np.maximum(yu_raw, y_top), y_top + max_cut)
        yl = np.maximum(np.minimum(yl_raw, y_bot), y_bot - max_cut)
        # If effectively at the rim, treat as "no lid here".
        yu = np.where((yu - y_top) < 5.0, y_top, yu)
        yl = np.where((y_bot - yl) < 5.0, y_bot, yl)
        yl = np.maximum(yl, yu + 1.0)
    else:
        yu, yl = y_top, y_bot

    Y, X = np.ogrid[:h, :w]
    circle = (X - float(cx)) ** 2 + (Y - float(cy)) ** 2 <= float(ir) ** 2
    between = (Y >= yu[np.newaxis, :]) & (Y <= yl[np.newaxis, :])

    mask = np.zeros((h, w), np.uint8)
    mask[circle & between] = 255
    return mask, float(max(0.0, min(1.0, true_occluded)))


# ==========================================
# 5. UNWRAP (roll-corrected, mask-aware)
# ==========================================
def unwrap_iris_fast(img, mask, px, py, pr, ir, roll_rad=0.0, mirrored=True):
    """
    Polar -> rectangular, rotated by roll_rad so column 0 is anatomical 12 o'clock.
    Masked (lid) samples become white, which is exactly the signal every AI prompt
    expects for "ignore this region".

    `mirrored` says whether the photo is left-right mirrored, as a phone selfie
    camera produces. The zone map is written for that convention: for a right eye
    it places TEMPORAL at minute 15, which is the +x direction in the image, and
    that only holds for a mirrored frame. A photo taken by someone else with the
    rear camera is not mirrored, and analysing it under the wrong assumption
    silently flips every organ attribution left for right — so the sweep direction
    is reversed instead.
    """
    h_out, w_out = UNWRAP_H, UNWRAP_W
    theta0 = -np.pi / 2 + float(roll_rad)
    direction = 1.0 if mirrored else -1.0

    theta = np.linspace(theta0, theta0 + direction * 2 * np.pi, w_out, endpoint=False).astype(np.float32)
    r_vals = np.linspace(pr, ir, h_out).astype(np.float32)
    theta_grid, r_grid = np.meshgrid(theta, r_vals)

    map_x = (px + r_grid * np.cos(theta_grid)).astype(np.float32)
    map_y = (py + r_grid * np.sin(theta_grid)).astype(np.float32)

    unwrapped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    if mask is not None:
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        unw_mask = cv2.remap(mask_bgr, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        gray_m = cv2.cvtColor(unw_mask, cv2.COLOR_BGR2GRAY)
        occluded = gray_m < 100
        unwrapped[occluded] = (255, 255, 255)
        masked_fraction = float(occluded.mean())
    else:
        masked_fraction = 0.0

    return unwrapped, masked_fraction


# ==========================================
# 6. VALIDITY GATE
# ==========================================
def validate_geometry(pupil, iris, frame_w, frame_h, masked_fraction=None,
                      pupil_occlusion=None):
    """
    Refuse to produce a strip we cannot trust. A confidently mislabelled analysis
    is worse than asking for another photo, because every organ attribution
    downstream inherits the error invisibly.
    """
    v = SPEC['validity']
    reasons = []

    if not pupil:
        return {'ok': False, 'reasons': ['PUPIL_NOT_FOUND']}
    if pupil['circularity'] < SPEC['pupil']['minCircularity']:
        reasons.append('PUPIL_NOT_ROUND')
    if not iris or iris['gradient'] < SPEC['iris']['minGradient']:
        reasons.append('IRIS_NOT_FOUND')
    if iris and iris['r'] < v['minIrisRadiusPx']:
        reasons.append('IRIS_TOO_SMALL')

    if iris and iris['r'] > 0:
        ratio = pupil['r'] / float(iris['r'])
        if ratio < v['minPupilIrisRatio'] or ratio > v['maxPupilIrisRatio']:
            reasons.append('RATIO_IMPLAUSIBLE')
        margin = iris['r'] * v['edgeMarginFactor']
        if (pupil['x'] < margin or pupil['y'] < margin or
                pupil['x'] > frame_w - margin or pupil['y'] > frame_h - margin):
            reasons.append('IRIS_OUT_OF_FRAME')

    if masked_fraction is not None and masked_fraction > v['maxMaskedFraction']:
        reasons.append('TOO_OCCLUDED')
    if pupil_occlusion is not None and pupil_occlusion > v['maxPupilRingOcclusion']:
        reasons.append('PUPIL_NOT_CLEAR')

    return {'ok': len(reasons) == 0, 'reasons': reasons}


# ==========================================
# 7. DRAW MAP
# ==========================================
def redistribute_ring_bands(unwrapped):
    """Resample the uniform 12-ring strip into the non-linear layout in
    RING_GROUPS, giving the organ zone (ORG_IN/MID/OUT) much more height."""
    w = unwrapped.shape[1]
    out = np.ones((DEST_CONTENT_H, w, 3), np.uint8) * 255
    y_dst = 0
    for grp in RING_GROUPS:
        band = unwrapped[grp['srcStart']:grp['srcEnd'], :]
        out[y_dst:y_dst + grp['destHeight'], :] = cv2.resize(
            band, (w, grp['destHeight']), interpolation=cv2.INTER_LINEAR)
        y_dst += grp['destHeight']
    return out


def draw_ai_grid_map_expanded(unwrapped, side="R"):
    """
    Renders the labelled strip the AI actually reads. Two deliberate choices: the
    organ zone (ORG_IN/MID/OUT) is resampled to far more vertical pixels than the
    thin bands, and every sector is printed as S1..S12 with alternating tint — so
    locating a finding is reading a printed label, not estimating a pixel offset.
    """
    if unwrapped is None:
        return np.zeros((DEST_CONTENT_H, UNWRAP_W, 3), np.uint8)
    unwrapped = unwrapped.astype(np.uint8)

    content = redistribute_ring_bands(unwrapped)
    img_h, img_w = content.shape[:2]

    sector_w = img_w / N_SECTORS
    alpha = SPEC['grid']['sectorTintAlpha']
    for sec in range(1, N_SECTORS, 2):
        x0, x1 = int(sec * sector_w), int((sec + 1) * sector_w)
        tint = np.full_like(content[:, x0:x1], (150, 150, 150))
        content[:, x0:x1] = cv2.addWeighted(content[:, x0:x1], 1 - alpha, tint, alpha, 0)

    g = SPEC['grid']
    pt, pl, pb, pr_pad = g['paddingTop'], g['paddingLeft'], g['paddingBottom'], g['paddingRight']

    cw = img_w + pl + pr_pad
    ch = img_h + pt + pb

    canvas = np.ones((ch, cw, 3), dtype=np.uint8) * 255
    canvas[pt:pt + img_h, pl:pl + img_w] = content

    c_grid = (200, 200, 200)
    c_group = (140, 60, 0)   # BGR: bold navy for ring-group boundaries
    c_txt = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # minutes: tick lines/numbers + sector numbers (S1..S12) above them
    for m in range(0, 61, 5):
        x = pl + int(m * (img_w / 60.0))
        if x >= pl + img_w:
            x = pl + img_w - 1
        cv2.line(canvas, (x, pt), (x, pt + img_h), c_grid, 1)
        cv2.line(canvas, (x, pt - 5), (x, pt), c_txt, 1)
        txt = f"{m}"
        (tw, _), _ = cv2.getTextSize(txt, font, 0.4, 1)
        cv2.putText(canvas, txt, (x - tw // 2, pt - 10), font, 0.4, c_txt, 1)

    for sec in range(N_SECTORS):
        x_center = pl + int((sec + 0.5) * sector_w)
        stxt = f"S{sec + 1}"
        (tw, _), _ = cv2.getTextSize(stxt, font, 0.4, 1)
        cv2.putText(canvas, stxt, (x_center - tw // 2, pt - 30), font, 0.4, (150, 0, 0), 1)

    # ring groups: bold boundary line + name label; thin sub-line for 2-ring bands
    y = pt
    for grp in RING_GROUPS:
        name, dh = grp['name'], grp['destHeight']
        cv2.line(canvas, (pl, y), (pl + img_w, y), c_group, 2)
        cv2.putText(canvas, name, (5, y + dh // 2 + 5), font, 0.45, c_txt, 1)
        if name in RING_SUBLINES:
            r_first, r_second = RING_SUBLINES[name]
            yy = y + dh // 2
            cv2.line(canvas, (pl, yy), (pl + img_w, yy), c_grid, 1)
            cv2.putText(canvas, f"R{r_first}", (pl - 32, y + dh // 4 + 4), font, 0.32, (90, 90, 90), 1)
            cv2.putText(canvas, f"R{r_second}", (pl - 32, yy + dh // 4 + 4), font, 0.32, (90, 90, 90), 1)
        y += dh
    cv2.line(canvas, (pl, y), (pl + img_w, y), c_group, 2)

    lbl = "RIGHT EYE" if side == "R" else "LEFT EYE"
    cv2.putText(canvas, lbl, (pl, ch - 10), font, 0.8, c_txt, 2)

    nm, tm = (45, 15) if side == "R" else (15, 45)
    nx = pl + int(nm * (img_w / 60.0))
    tx = pl + int(tm * (img_w / 60.0))
    cv2.putText(canvas, "^ NASAL", (nx - 30, ch - 10), font, 0.5, (0, 0, 255), 1)
    cv2.putText(canvas, "^ TEMPORAL", (tx - 40, ch - 10), font, 0.5, (100, 100, 100), 1)

    return canvas


def draw_overlay(img, px, py, pr, ir, corners=None):
    out = img.copy()
    cv2.circle(out, (px, py), pr, (0, 255, 0), 2)
    cv2.circle(out, (px, py), ir, (0, 255, 255), 2)
    cv2.line(out, (px, py - ir), (px, py + ir), (255, 255, 0), 1)
    cv2.line(out, (px - ir, py), (px + ir, py), (255, 255, 0), 1)
    if corners:
        (x1, y1), (x2, y2) = corners
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
    return out


# ==========================================
# 8. FULL PIPELINE FOR ONE EYE
# ==========================================
def analyze_eye(img, side, mirrored=True):
    """
    Full geometry pipeline for one eye. Returns a dict with either ok=True plus the
    strip/overlay/quality, or ok=False plus an actionable Bulgarian message.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = preprocess_image(img)

    candidates = find_pupil_candidates(enhanced)
    if not candidates:
        return {'ok': False, 'code': 'PUPIL_NOT_FOUND', 'message': REJECT_MESSAGES['PUPIL_NOT_FOUND']}

    # Keep the first pupil hypothesis whose iris pairing actually validates,
    # rather than trusting a single threshold to have found the real pupil.
    pupil = iris = None
    first_failure = None
    for cand in candidates:
        cand_iris = find_iris_outer_boundary(gray, cand['x'], cand['y'], cand['r'])
        pre = validate_geometry(cand, cand_iris, w, h, None)
        if pre['ok']:
            pupil, iris = cand, cand_iris
            break
        if first_failure is None:
            first_failure = pre['reasons']

    if pupil is None:
        reasons = first_failure or ['PUPIL_NOT_FOUND']
        code = reasons[0]
        return {'ok': False, 'code': code,
                'message': REJECT_MESSAGES.get(code, 'Снимката не е подходяща за анализ.'),
                'reasons': reasons}

    pupil_occlusion = pupil_clearance_occlusion(gray, pupil['x'], pupil['y'], pupil['r'], iris['r'])

    coef_u, coef_l = fit_eyelids(enhanced, pupil['x'], pupil['y'], iris['r'], pupil['r'])
    roll = estimate_roll(coef_u, coef_l, pupil['x'], iris['r'])
    mask, true_occluded = build_mask(h, w, pupil['x'], pupil['y'], iris['r'], coef_u, coef_l)

    unw, masked_fraction = unwrap_iris_fast(
        img, mask, pupil['x'], pupil['y'], pupil['r'], iris['r'], roll['roll_rad'], mirrored)

    # Judge on the worse of what we masked and what the lids actually cover.
    occlusion = max(masked_fraction, true_occluded)
    post = validate_geometry(pupil, iris, w, h, occlusion, pupil_occlusion)
    if not post['ok']:
        code = post['reasons'][0]
        return {'ok': False, 'code': code,
                'message': REJECT_MESSAGES.get(code, 'Снимката не е подходяща за анализ.'),
                'reasons': post['reasons']}

    mapped = draw_ai_grid_map_expanded(unw, side=side)
    overlay = draw_overlay(img, pupil['x'], pupil['y'], pupil['r'], iris['r'], roll['corners'])

    # Honest composite: edge strength, pupil roundness, visible area, and whether
    # an anatomical rotation reference could be established at all.
    score = 100.0 * (
        0.35 * iris['confidence'] +
        0.20 * min(1.0, pupil['circularity'] / 0.85) +
        0.30 * max(0.0, 1 - occlusion / SPEC['validity']['maxMaskedFraction']) +
        0.15 * (1.0 if roll['source'] == 'canthi' else 0.0)
    )

    return {
        'ok': True,
        'overlay': overlay,
        'mapped': mapped,
        'geometry': {
            'pupil': pupil,
            'iris': iris,
            'rollDeg': float(np.rad2deg(roll['roll_rad'])),
            'rollSource': roll['source'],
            'lidsDetected': coef_u is not None and coef_l is not None,
            'pupilRingOcclusion': round(pupil_occlusion, 3),
            'mirrored': bool(mirrored),
        },
        'quality': {
            'irisConfidence': iris['confidence'],
            'pupilCircularity': pupil['circularity'],
            'visibleFraction': 1.0 - occlusion,
            'rollCorrected': roll['source'] == 'canthi',
            'score': int(round(score)),
        },
    }


# ==========================================
# 9. HELPER: Call Cloudflare Worker for AI analysis
# ==========================================
def call_worker_analysis(strips, questionnaire=None, ai_provider=None, ai_model=None):
    """
    Send the unwrapped iris strips to the Cloudflare Worker for one combined
    analysis. `strips` maps side ('R'/'L') to {'strip': base64, 'quality': dict}.

    Both eyes go in a single request because the report is written over both:
    the right eye reads the right side of the body and the left the left, and
    several zones exist in only one eye.
    """
    if not WORKER_URL or not strips:
        return None

    try:
        joined = ''.join(strips[s]['strip'] for s in sorted(strips))
        image_hash = hashlib.sha256(joined.encode()).hexdigest()[:12]

        data = {'image_hash': image_hash}
        for side, payload in strips.items():
            key = side.lower()
            data[f'strip_image_{key}'] = payload['strip']
            if payload.get('quality'):
                data[f'capture_quality_{key}'] = json.dumps(payload['quality'])

        if questionnaire:
            data['questionnaire'] = json.dumps(questionnaire)
        if ai_provider or DEFAULT_AI_PROVIDER:
            data['ai_provider'] = ai_provider or DEFAULT_AI_PROVIDER
        if ai_model or DEFAULT_AI_MODEL:
            data['ai_model'] = ai_model or DEFAULT_AI_MODEL

        response = requests.post(f"{WORKER_URL}/analyze", data=data, timeout=WORKER_TIMEOUT)

        if response.status_code == 200:
            return response.json()
        # Sanitize error message - don't expose raw response details
        return {'error': f'AI analysis service error (status {response.status_code})'}

    except requests.exceptions.Timeout:
        return {'error': 'AI analysis timed out. Please try again.'}
    except requests.exceptions.ConnectionError:
        return {'error': 'Cannot connect to AI analysis service.'}
    except Exception:
        return {'error': 'AI analysis encountered an unexpected error.'}


# ==========================================
# 10. FLASK APP
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    """
    Process uploaded iris images: detect pupil/iris, mask eyelids, correct for
    head/camera roll, unwrap, and optionally send to the Worker for AI analysis.
    """
    results = {}
    strips = {}

    questionnaire = None
    q_raw = request.form.get('questionnaire')
    if q_raw:
        try:
            questionnaire = json.loads(q_raw)
        except (json.JSONDecodeError, TypeError):
            pass

    run_ai = request.form.get('run_ai', 'false').lower() == 'true'
    # Selfie cameras mirror the frame; the zone map is written for that convention.
    mirrored = request.form.get('mirrored', 'true').lower() != 'false'
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
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        analysis = analyze_eye(img, sc, mirrored)

        if not analysis['ok']:
            results[sc] = {
                'found': False,
                'code': analysis['code'],
                'error': analysis['message'],
            }
            continue

        b_ovl = base64.b64encode(cv2.imencode('.jpg', analysis['overlay'])[1]).decode()
        b_map = base64.b64encode(cv2.imencode('.jpg', analysis['mapped'])[1]).decode()

        results[sc] = {
            'found': True,
            'overlay': b_ovl,
            'mapped': b_map,
            'quality': analysis['quality'],
            'geometry': analysis['geometry'],
        }
        strips[sc] = {'strip': b_map, 'quality': analysis['quality']}

    if run_ai and WORKER_URL and strips:
        ai_result = call_worker_analysis(strips, questionnaire, ai_provider, ai_model)
        if ai_result:
            results['ai_analysis'] = ai_result

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

    strips = {}
    for side in ('R', 'L'):
        strip = request.form.get(f'strip_image_{side.lower()}')
        if strip:
            strips[side] = {'strip': strip, 'quality': None}
    if not strips:
        strip_b64 = request.form.get('strip_image')
        side = request.form.get('side', 'R').upper()
        if strip_b64:
            if side not in ('R', 'L'):
                return jsonify({'error': 'side must be R or L'}), 400
            strips[side] = {'strip': strip_b64, 'quality': None}
    if not strips:
        return jsonify({'error': 'strip_image_r and/or strip_image_l are required'}), 400

    questionnaire = None
    q_raw = request.form.get('questionnaire')
    if q_raw:
        try:
            questionnaire = json.loads(q_raw)
        except (json.JSONDecodeError, TypeError):
            pass

    ai_provider = request.form.get('ai_provider')
    ai_model = request.form.get('ai_model')

    result = call_worker_analysis(strips, questionnaire, ai_provider, ai_model)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
