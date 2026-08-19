/*
 * iris-geometry.js — canonical client-side iris geometry pipeline.
 *
 * This file is the single implementation used by the browser path. app.py is
 * its Python twin and MUST stay behaviourally identical; config/geometry-spec.json
 * holds the constants both read, and tests/test_geometry.py asserts the two agree.
 * That contract exists because the two paths had already drifted badly: the browser
 * path had no eyelid masking at all, so eyelashes and lid skin were being analysed
 * as iris tissue while every AI prompt still said "white bands = eyelid, ignore".
 *
 * Pipeline: gray -> pupil -> iris boundary -> eyelid curves -> roll (canthi) ->
 *           validity gate -> unwrap (roll-corrected, mask-aware) -> labelled grid
 *
 * Requires OpenCV.js (global `cv`).
 */
(function (global) {
  'use strict';

  // Mirror of config/geometry-spec.json. Inlined rather than fetched so the app
  // still works when opened straight from disk (file:// blocks fetch).
  // tests/test_geometry.py fails if these drift from the JSON.
  const SPEC = {
    version: 12,
    unwrap: { width: 1200, height: 300 },
    sectors: 12,
    ringGroups: [
      { name: 'IPB',     srcStart: 0,   srcEnd: 25,  destHeight: 20,  rings: [0, 0] },
      { name: 'STOM',    srcStart: 25,  srcEnd: 50,  destHeight: 20,  rings: [1, 1] },
      { name: 'ANW',     srcStart: 50,  srcEnd: 100, destHeight: 60,  rings: [2, 3] },
      { name: 'ORG_IN',  srcStart: 100, srcEnd: 150, destHeight: 140, rings: [4, 5] },
      { name: 'ORG_MID', srcStart: 150, srcEnd: 200, destHeight: 140, rings: [6, 7] },
      { name: 'ORG_OUT', srcStart: 200, srcEnd: 250, destHeight: 140, rings: [8, 9] },
      { name: 'LYM',     srcStart: 250, srcEnd: 275, destHeight: 40,  rings: [10, 10] },
      { name: 'SCU',     srcStart: 275, srcEnd: 300, destHeight: 40,  rings: [11, 11] },
    ],
    grid: { paddingLeft: 100, paddingTop: 70, paddingBottom: 40, paddingRight: 20, sectorTintAlpha: 0.12 },
    pupil: { medianBlur: 7, darkFractions: [0.005, 0.01, 0.02, 0.04, 0.06], minThreshold: 15, maxThreshold: 90, minArea: 50, minCircularity: 0.55, closeKernel: 7 },
    iris: { minRadiusFactor: 2.2, maxRadiusFactor: 7.5, step: 2, minGradient: 5.0, goodGradient: 25.0 },
    eyelid: { pupilMarginPx: 5, baseGradientThreshold: 25.0, ransacThreshold: 4.0, ransacIters: 450, maxCutFraction: 0.25, minPoints: 40, minFilteredPoints: 30 },
    roll: { maxRollDeg: 25.0, minCornerSpanFactor: 1.5 },
    validity: { minPupilIrisRatio: 0.15, maxPupilIrisRatio: 0.75, maxMaskedFraction: 0.45, minIrisRadiusPx: 40, maxPupilRingOcclusion: 0.5, edgeMarginFactor: 0.9 },
  };

  const DEST_CONTENT_H = SPEC.ringGroups.reduce((s, g) => s + g.destHeight, 0);
  // Sub-ring reference lines drawn inside the two-ring bands.
  const RING_SUBLINES = { ANW: [2, 3], ORG_IN: [4, 5], ORG_MID: [6, 7], ORG_OUT: [8, 9] };

  // Reasons a capture is rejected, with the Bulgarian guidance shown to the user.
  const REJECT_MESSAGES = {
    PUPIL_NOT_FOUND: 'Зеницата не беше открита. Снимайте отблизо, с окото в центъра на кадъра.',
    PUPIL_NOT_ROUND: 'Зеницата не се вижда достатъчно ясно. Избягвайте отблясъци и гледайте право в камерата.',
    IRIS_NOT_FOUND: 'Границата на ириса не се различава. Нужна е по-добра, равномерна светлина.',
    IRIS_TOO_SMALL: 'Окото е твърде малко в кадъра. Приближете камерата.',
    IRIS_OUT_OF_FRAME: 'Окото не се побира в кадъра. Центрирайте го и снимайте отново.',
    RATIO_IMPLAUSIBLE: 'Зеницата и ирисът не бяха разделени коректно. Опитайте при по-различна светлина.',
    TOO_OCCLUDED: 'Клепачите закриват твърде голяма част от ириса. Отворете окото по-широко.',
    PUPIL_NOT_CLEAR: 'Зеницата не се вижда достатъчно ясно — клепачът или миглите я закриват. '
      + 'Направете нова снимка с широко отворено око и зеница, видима изцяло.',
  };

  // ---------------------------------------------------------------- utilities

  function toGray(srcMat) {
    const gray = new cv.Mat();
    if (srcMat.channels() === 1) {
      srcMat.copyTo(gray);
    } else if (srcMat.channels() === 4) {
      cv.cvtColor(srcMat, gray, cv.COLOR_RGBA2GRAY);
    } else {
      cv.cvtColor(srcMat, gray, cv.COLOR_RGB2GRAY);
    }
    return gray;
  }

  function enhance(grayMat) {
    const clahe = new cv.CLAHE(3.0, new cv.Size(8, 8));
    const out = new cv.Mat();
    clahe.apply(grayMat, out);
    clahe.delete();
    return out;
  }

  /** Intensity below which `frac` of pixels fall, clamped to [lo, hi]. */
  function darkThreshold(grayMat, frac, lo, hi) {
    const data = grayMat.data;
    const hist = new Uint32Array(256);
    for (let i = 0; i < data.length; i++) hist[data[i]]++;
    const target = data.length * frac;
    let cum = 0;
    let t = lo;
    for (let v = 0; v < 256; v++) {
      cum += hist[v];
      if (cum >= target) { t = v; break; }
    }
    return Math.min(hi, Math.max(lo, t));
  }

  function polyval2(c, x) { return (c[0] * x + c[1]) * x + c[2]; }

  /** Least-squares quadratic fit; returns [a, b, c] or null if degenerate. */
  function polyfit2(xs, ys) {
    const n = xs.length;
    if (n < 3) return null;
    let s0 = n, s1 = 0, s2 = 0, s3 = 0, s4 = 0, t0 = 0, t1 = 0, t2 = 0;
    for (let i = 0; i < n; i++) {
      const x = xs[i], y = ys[i];
      const x2 = x * x;
      s1 += x; s2 += x2; s3 += x2 * x; s4 += x2 * x2;
      t0 += y; t1 += x * y; t2 += x2 * y;
    }
    // Solve the 3x3 normal equations by Cramer's rule.
    const m = [[s4, s3, s2], [s3, s2, s1], [s2, s1, s0]];
    const rhs = [t2, t1, t0];
    const det3 = (a) =>
      a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
      a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
      a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    const D = det3(m);
    if (!isFinite(D) || Math.abs(D) < 1e-9) return null;
    const col = (a, i, v) => a.map((row, r) => row.map((val, c) => (c === i ? v[r] : val)));
    return [det3(col(m, 0, rhs)) / D, det3(col(m, 1, rhs)) / D, det3(col(m, 2, rhs)) / D];
  }

  /** RANSAC quadratic fit, robust to eyelash/texture outliers. */
  function ransacPolyfit2(xs, ys, seed) {
    const n = xs.length;
    const cfg = SPEC.eyelid;
    if (n < cfg.minFilteredPoints) return null;

    // Deterministic PRNG so the same photo always yields the same geometry.
    let state = (seed || 1) >>> 0;
    const rand = () => {
      state = (state * 1664525 + 1013904223) >>> 0;
      return state / 4294967296;
    };

    let bestCoef = null;
    let bestCount = -1;
    let bestInliers = null;

    for (let iter = 0; iter < cfg.ransacIters; iter++) {
      const i0 = (rand() * n) | 0, i1 = (rand() * n) | 0, i2 = (rand() * n) | 0;
      if (i0 === i1 || i1 === i2 || i0 === i2) continue;
      const coef = polyfit2([xs[i0], xs[i1], xs[i2]], [ys[i0], ys[i1], ys[i2]]);
      if (!coef) continue;
      let count = 0;
      const inliers = new Uint8Array(n);
      for (let i = 0; i < n; i++) {
        if (Math.abs(ys[i] - polyval2(coef, xs[i])) < cfg.ransacThreshold) { inliers[i] = 1; count++; }
      }
      if (count > bestCount) { bestCount = count; bestCoef = coef; bestInliers = inliers; }
    }

    if (!bestCoef) return polyfit2(xs, ys);
    if (bestCount < cfg.minPoints) return polyfit2(xs, ys);

    const ix = [], iy = [];
    for (let i = 0; i < n; i++) if (bestInliers[i]) { ix.push(xs[i]); iy.push(ys[i]); }
    return polyfit2(ix, iy) || bestCoef;
  }

  // ------------------------------------------------------------ pupil / iris

  /** Largest roundness-weighted dark blob below intensity `t`. */
  function bestDarkBlob(blur, t, cfg) {
    const thresh = new cv.Mat();
    cv.threshold(blur, thresh, t, 255, cv.THRESH_BINARY_INV);

    // Corneal reflection punches bright holes in the pupil; close them so the
    // pupil stays one blob instead of a ring of fragments.
    const k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(cfg.closeKernel, cfg.closeKernel));
    cv.morphologyEx(thresh, thresh, cv.MORPH_CLOSE, k);

    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.findContours(thresh, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    let best = null;
    let bestScore = 0;
    for (let i = 0; i < contours.size(); i++) {
      const cnt = contours.get(i);
      const area = cv.contourArea(cnt);
      if (area < cfg.minArea) { cnt.delete(); continue; }
      const perim = cv.arcLength(cnt, true);
      if (perim <= 0) { cnt.delete(); continue; }
      const circularity = (4 * Math.PI * area) / (perim * perim);
      const circle = cv.minEnclosingCircle(cnt);
      // Rank by area weighted by roundness, so a large ragged shadow cannot beat
      // a smaller genuinely circular pupil.
      const score = area * Math.max(circularity, 0.01);
      if (score > bestScore) {
        bestScore = score;
        best = {
          x: Math.round(circle.center.x),
          y: Math.round(circle.center.y),
          r: Math.round(circle.radius),
          circularity: circularity,
        };
      }
      cnt.delete();
    }

    thresh.delete(); k.delete(); contours.delete(); hierarchy.delete();
    return best;
  }

  /**
   * Competing pupil hypotheses, darkest first.
   *
   * A single adaptive threshold is not enough: on a dark brown iris the iris disc
   * itself falls below the cut and gets returned as one huge "pupil", which is
   * exactly how the old fixed threshold of 40 corrupted every downstream
   * coordinate. Emitting several candidates lets the caller keep the one whose
   * pupil+iris pair actually validates.
   */
  function findPupilCandidates(grayMat) {
    const cfg = SPEC.pupil;
    const blur = new cv.Mat();
    cv.medianBlur(grayMat, blur, cfg.medianBlur);

    const candidates = [];
    const seen = new Set();
    for (const frac of cfg.darkFractions) {
      const t = darkThreshold(blur, frac, cfg.minThreshold, cfg.maxThreshold);
      if (seen.has(t)) continue;
      seen.add(t);
      const cand = bestDarkBlob(blur, t, cfg);
      if (cand && !candidates.some((c) => Math.abs(c.r - cand.r) <= 2)) candidates.push(cand);
    }

    blur.delete();
    return candidates;
  }

  /** Best single pupil estimate (darkest plausible blob). */
  function findPupil(grayMat) {
    const candidates = findPupilCandidates(grayMat);
    return candidates.length ? candidates[0] : null;
  }

  /**
   * Iris outer boundary by radial gradient, sampled near the horizontal meridians
   * where the lids rarely reach. Returns the measured gradient so callers can tell
   * a real edge from a guess — the old code silently returned pupilR * 4.0.
   */
  function findIrisOuter(grayMat, px, py, pr) {
    const cfg = SPEC.iris;
    const w = grayMat.cols, h = grayMat.rows;

    let minR = Math.round(pr * cfg.minRadiusFactor);
    let maxR = Math.round(pr * cfg.maxRadiusFactor);
    const maxDist = Math.min(px, py, w - px, h - py);
    if (maxR > maxDist) maxR = Math.floor(maxDist);
    if (minR >= maxR) return { r: Math.round(pr * 3.5), gradient: 0, confidence: 0 };

    const blurred = new cv.Mat();
    cv.GaussianBlur(grayMat, blurred, new cv.Size(5, 5), 0);
    const data = blurred.data;
    const at = (x, y) => data[y * w + x];

    const angles = [0, 180, 15, -15, 165, 195, 30, -30, 150, 210].map((d) => (d * Math.PI) / 180);
    let bestR = minR, bestGrad = -1;

    for (let r = minR; r < maxR; r += cfg.step) {
      let sum = 0, samples = 0;
      for (const rad of angles) {
        const nx = Math.cos(rad), ny = Math.sin(rad);
        const xi = Math.round(px + (r - 3) * nx), yi = Math.round(py + (r - 3) * ny);
        const xo = Math.round(px + (r + 3) * nx), yo = Math.round(py + (r + 3) * ny);
        if (xi >= 0 && xi < w && yi >= 0 && yi < h && xo >= 0 && xo < w && yo >= 0 && yo < h) {
          sum += at(xo, yo) - at(xi, yi);
          samples++;
        }
      }
      if (samples > 0) {
        const avg = sum / samples;
        if (avg > bestGrad) { bestGrad = avg; bestR = r; }
      }
    }
    blurred.delete();

    const confidence = Math.max(0, Math.min(1, bestGrad / cfg.goodGradient));
    return { r: bestR, gradient: bestGrad, confidence: confidence };
  }

  // ----------------------------------------------------------------- eyelids

  /**
   * Candidate lid points: strongest |dI/dy| within the iris annulus above/below
   * the pupil. Searching the full annulus (rather than a thin band near the rim)
   * is what lets a deeply intruding lid be found at all — with the old narrow band
   * a half-closed eye yielded no fit, so it passed the gate as fully visible.
   * The pupil disc itself is excluded so its own strong edge cannot be mistaken
   * for a lid.
   */
  function eyelidPoints(grayMat, cx, cy, R, pupilR) {
    const cfg = SPEC.eyelid;
    const w = grayMat.cols, h = grayMat.rows;

    const blurred = new cv.Mat();
    cv.GaussianBlur(grayMat, blurred, new cv.Size(5, 5), 0);
    const gy = new cv.Mat();
    cv.Sobel(blurred, gy, cv.CV_32F, 0, 1, 3);
    const gyData = gy.data32F;
    const g = { get: (i) => Math.abs(gyData[i]) };

    const upX = [], upY = [], upV = [], loX = [], loY = [], loV = [];
    const x0 = Math.max(0, Math.floor(cx - R));
    const x1 = Math.min(w - 1, Math.ceil(cx + R));

    for (let xi = x0; xi <= x1; xi++) {
      const dx = xi - cx;
      const inside = R * R - dx * dx;
      if (inside <= 0) continue;
      const root = Math.sqrt(inside);
      const yTop = Math.max(0, Math.floor(cy - root));
      const yBot = Math.min(h - 1, Math.ceil(cy + root));

      let yUpperEnd = Math.round(cy - pupilR - cfg.pupilMarginPx);
      yUpperEnd = Math.max(yTop + 6, Math.min(Math.round(cy) - 6, yUpperEnd));
      let yLowerStart = Math.round(cy + pupilR + cfg.pupilMarginPx);
      yLowerStart = Math.min(yBot - 6, Math.max(Math.round(cy) + 6, yLowerStart));

      if (yUpperEnd > yTop + 8) {
        let bestY = yTop, bestV = -1;
        for (let y = yTop; y < yUpperEnd; y++) {
          const v = g.get(y * w + xi);
          if (v > bestV) { bestV = v; bestY = y; }
        }
        upX.push(xi); upY.push(bestY); upV.push(bestV);
      }
      if (yBot > yLowerStart + 8) {
        let bestY = yLowerStart, bestV = -1;
        for (let y = yLowerStart; y < yBot; y++) {
          const v = g.get(y * w + xi);
          if (v > bestV) { bestV = v; bestY = y; }
        }
        loX.push(xi); loY.push(bestY); loV.push(bestV);
      }
    }

    blurred.delete(); gy.delete();
    return { upX, upY, upV, loX, loY, loV };
  }

  function percentile(arr, p) {
    if (!arr.length) return 0;
    const sorted = Array.from(arr).sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))];
  }

  function filterByStrength(xs, ys, vs, base) {
    const thr = vs.length ? Math.max(base, percentile(vs, 0.6)) : base;
    const fx = [], fy = [];
    for (let i = 0; i < xs.length; i++) if (vs[i] >= thr) { fx.push(xs[i]); fy.push(ys[i]); }
    return { xs: fx, ys: fy };
  }

  /** Fit upper/lower lid curves. Returns {coefUpper, coefLower} or nulls. */
  function fitEyelids(grayMat, cx, cy, R, pupilR) {
    const cfg = SPEC.eyelid;
    const pts = eyelidPoints(grayMat, cx, cy, R, pupilR);
    if (pts.upX.length < cfg.minPoints || pts.loX.length < cfg.minPoints) {
      return { coefUpper: null, coefLower: null };
    }
    const up = filterByStrength(pts.upX, pts.upY, pts.upV, cfg.baseGradientThreshold);
    const lo = filterByStrength(pts.loX, pts.loY, pts.loV, cfg.baseGradientThreshold);
    if (up.xs.length < cfg.minFilteredPoints || lo.xs.length < cfg.minFilteredPoints) {
      return { coefUpper: null, coefLower: null };
    }
    return {
      coefUpper: ransacPolyfit2(up.xs, up.ys, 1),
      coefLower: ransacPolyfit2(lo.xs, lo.ys, 2),
    };
  }

  /**
   * Fraction of the ring just outside the pupil that does not look like iris.
   *
   * This does not depend on the eyelid fit, which is exactly why it exists: when a
   * lid closes far enough to reach the pupil rim, the lid search band collapses and
   * no curve can be fitted at all, so the lid-based occlusion measure reports zero
   * and a nearly shut eye would sail through the gate. Here the iris appearance is
   * sampled near the horizontal meridians (which lids practically never reach) and
   * used as the reference, then compared against samples straight above and below
   * the pupil, where a lid must be if it is covering it.
   */
  function pupilClearanceOcclusion(grayMat, cx, cy, pr, ir) {
    if (ir <= pr + 8) return 1;

    const blurred = new cv.Mat();
    cv.GaussianBlur(grayMat, blurred, new cv.Size(5, 5), 0);
    const data = blurred.data;
    const w = blurred.cols, h = blurred.rows;

    const sample = (deg, frac) => {
      const rad = (deg * Math.PI) / 180;
      const r = pr + frac * (ir - pr);
      const x = Math.round(cx + r * Math.cos(rad));
      const y = Math.round(cy + r * Math.sin(rad));
      if (x >= 0 && x < w && y >= 0 && y < h) return data[y * w + x];
      return null;
    };
    const median = (a) => {
      const s = Array.from(a).sort((p, q) => p - q);
      return s[Math.floor(s.length / 2)];
    };

    // Reference: the iris either side of the pupil, horizontally.
    const ref = [];
    for (const range of [[-32, 32], [148, 212]]) {
      for (let deg = range[0]; deg <= range[1]; deg += 4) {
        for (const frac of [0.20, 0.35, 0.50]) {
          const v = sample(deg, frac);
          if (v !== null) ref.push(v);
        }
      }
    }
    if (ref.length < 12) { blurred.delete(); return 1; }

    const refMed = median(ref);
    const mad = median(ref.map(v => Math.abs(v - refMed)));
    // Generous tolerance: normal irises vary a lot sector to sector, and a false
    // rejection costs the user a retake for nothing.
    const tol = Math.max(45, 3 * 1.4826 * mad);

    let occluded = 0, total = 0;
    for (const range of [[62, 118], [242, 298]]) {
      for (let deg = range[0]; deg <= range[1]; deg += 4) {
        const vals = [0.15, 0.30, 0.45].map(f => sample(deg, f)).filter(v => v !== null);
        if (!vals.length) continue;
        total++;
        if (Math.abs(median(vals) - refMed) > tol) occluded++;
      }
    }

    blurred.delete();
    return total ? occluded / total : 1;
  }

  /**
   * Camera/head roll from the canthi (eye corners), taken as the intersections of
   * the two lid curves. This is what makes minute 0 mean *anatomical* up rather
   * than "up in the photo" — without it a routine phone tilt rotated the whole
   * clock and shifted every organ sector.
   */
  function estimateRoll(coefUpper, coefLower, cx, irisR) {
    if (!coefUpper || !coefLower) return { rollRad: 0, corners: null, source: 'none' };

    const a = coefUpper[0] - coefLower[0];
    const b = coefUpper[1] - coefLower[1];
    const c = coefUpper[2] - coefLower[2];

    let x1, x2;
    if (Math.abs(a) < 1e-9) {
      if (Math.abs(b) < 1e-9) return { rollRad: 0, corners: null, source: 'none' };
      return { rollRad: 0, corners: null, source: 'none' }; // single crossing: not a pair of corners
    }
    const disc = b * b - 4 * a * c;
    if (disc <= 0) return { rollRad: 0, corners: null, source: 'none' };
    const sq = Math.sqrt(disc);
    x1 = (-b - sq) / (2 * a);
    x2 = (-b + sq) / (2 * a);
    if (x1 > x2) { const t = x1; x1 = x2; x2 = t; }

    // The corners must straddle the iris and sit outside it, otherwise the two
    // curves crossed somewhere meaningless and the estimate is not trustworthy.
    if (!(x1 < cx && cx < x2)) return { rollRad: 0, corners: null, source: 'none' };
    if ((x2 - x1) < SPEC.roll.minCornerSpanFactor * irisR) return { rollRad: 0, corners: null, source: 'none' };

    const y1 = polyval2(coefUpper, x1);
    const y2 = polyval2(coefUpper, x2);
    if (!isFinite(y1) || !isFinite(y2)) return { rollRad: 0, corners: null, source: 'none' };

    const rollRad = Math.atan2(y2 - y1, x2 - x1);
    const maxRad = (SPEC.roll.maxRollDeg * Math.PI) / 180;
    if (Math.abs(rollRad) > maxRad) return { rollRad: 0, corners: null, source: 'out_of_range' };

    return {
      rollRad: rollRad,
      corners: [[x1, y1], [x2, y2]],
      source: 'canthi',
    };
  }

  /**
   * Binary mask: 255 = usable iris, 0 = lid-covered or outside the iris.
   *
   * Returns { mask, trueOccluded }. The mask deliberately clamps how much it will
   * cut (maxCutFraction) so one bad RANSAC fit cannot destroy an otherwise good
   * capture — but that clamp also means a genuinely half-closed eye would be
   * under-masked and analysed as if it were fine. So the occlusion actually implied
   * by the fitted lids is measured separately, unclamped, and that honest number is
   * what the validity gate judges.
   */
  function buildMask(w, h, cx, cy, R, coefUpper, coefLower) {
    const mask = new Uint8Array(w * h);
    const maxCut = R * SPEC.eyelid.maxCutFraction;
    const haveLids = !!(coefUpper && coefLower);

    let spanTotal = 0, visibleTotal = 0;

    for (let x = 0; x < w; x++) {
      const dx = x - cx;
      const inside = R * R - dx * dx;
      if (inside <= 0) continue;
      const root = Math.sqrt(inside);
      const yTop = cy - root;
      const yBot = cy + root;
      spanTotal += yBot - yTop;

      let yu = yTop, yl = yBot;
      if (haveLids) {
        const yuRaw = polyval2(coefUpper, x);
        const ylRaw = polyval2(coefLower, x);
        visibleTotal += Math.max(0, Math.min(yBot, ylRaw) - Math.max(yTop, yuRaw));

        yu = Math.min(Math.max(yuRaw, yTop), yTop + maxCut);
        yl = Math.max(Math.min(ylRaw, yBot), yBot - maxCut);
        if (yu - yTop < 5) yu = yTop;      // effectively at the rim: no lid here
        if (yBot - yl < 5) yl = yBot;
        if (yl < yu + 1) yl = yu + 1;
      } else {
        visibleTotal += yBot - yTop;
      }

      const y0 = Math.max(0, Math.ceil(Math.max(yTop, yu)));
      const y1 = Math.min(h - 1, Math.floor(Math.min(yBot, yl)));
      for (let y = y0; y <= y1; y++) mask[y * w + x] = 255;
    }

    const trueOccluded = spanTotal > 0
      ? Math.max(0, Math.min(1, 1 - visibleTotal / spanTotal))
      : 0;
    return { mask: mask, trueOccluded: trueOccluded };
  }

  // ------------------------------------------------------------------ unwrap

  /**
   * Polar -> rectangular, rotated by `rollRad` so column 0 is anatomical 12 o'clock.
   * Masked (lid) samples are written as white, which is exactly the signal every
   * AI prompt expects for "ignore this region".
   */
  function unwrap(srcMat, mask, px, py, pr, ir, rollRad) {
    const W = SPEC.unwrap.width, H = SPEC.unwrap.height;
    const out = new cv.Mat(H, W, cv.CV_8UC4);
    const src = srcMat.data;
    const sw = srcMat.cols, sh = srcMat.rows, sc = srcMat.channels();
    const dst = out.data;
    const theta0 = -Math.PI / 2 + (rollRad || 0);

    let masked = 0;
    for (let y = 0; y < H; y++) {
      const radius = pr + (y / H) * (ir - pr);
      for (let x = 0; x < W; x++) {
        const angle = theta0 + (x / W) * 2 * Math.PI;
        const sx = Math.round(px + radius * Math.cos(angle));
        const sy = Math.round(py + radius * Math.sin(angle));
        const di = (y * W + x) * 4;

        let r = 255, g = 255, b = 255;
        if (sx >= 0 && sx < sw && sy >= 0 && sy < sh) {
          if (mask && mask[sy * sw + sx] === 0) {
            masked++;                       // lid-covered: leave it white
          } else {
            const si = (sy * sw + sx) * sc;
            r = src[si]; g = src[si + 1]; b = src[si + 2];
          }
        } else {
          masked++;
        }
        dst[di] = r; dst[di + 1] = g; dst[di + 2] = b; dst[di + 3] = 255;
      }
    }
    return { mat: out, maskedFraction: masked / (W * H) };
  }

  // ------------------------------------------------------------- grid render

  /**
   * Renders the labelled strip the AI actually reads. Two deliberate choices:
   * the organ zone (ORG_IN/MID/OUT) is resampled to far more vertical pixels
   * than the thin bands, and every sector is printed as S1..S12 with alternating
   * tint — so locating a finding is reading a printed label, not estimating a
   * pixel offset.
   */
  function renderGrid(unwrappedMat, side, canvas) {
    const imgWidth = SPEC.unwrap.width;
    const imgHeight = DEST_CONTENT_H;
    const { paddingLeft, paddingTop, paddingBottom, paddingRight, sectorTintAlpha } = SPEC.grid;
    const canvasWidth = imgWidth + paddingLeft + paddingRight;
    const canvasHeight = imgHeight + paddingTop + paddingBottom;

    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    const tmp = document.createElement('canvas');
    tmp.width = unwrappedMat.cols;
    tmp.height = unwrappedMat.rows;
    cv.imshow(tmp, unwrappedMat);

    let yDst = paddingTop;
    for (const grp of SPEC.ringGroups) {
      ctx.drawImage(tmp, 0, grp.srcStart, unwrappedMat.cols, grp.srcEnd - grp.srcStart,
                    paddingLeft, yDst, imgWidth, grp.destHeight);
      yDst += grp.destHeight;
    }

    const nSectors = SPEC.sectors;
    const sectorW = imgWidth / nSectors;
    ctx.fillStyle = `rgba(150, 150, 150, ${sectorTintAlpha})`;
    for (let s = 1; s < nSectors; s += 2) {
      ctx.fillRect(paddingLeft + s * sectorW, paddingTop, sectorW, imgHeight);
    }

    ctx.lineWidth = 1;
    for (let m = 0; m <= 60; m += 5) {
      const x = paddingLeft + (m / 60) * imgWidth;
      ctx.strokeStyle = 'rgb(200,200,200)';
      ctx.beginPath(); ctx.moveTo(x, paddingTop); ctx.lineTo(x, paddingTop + imgHeight); ctx.stroke();
      ctx.strokeStyle = 'rgb(0,0,0)';
      ctx.beginPath(); ctx.moveTo(x, paddingTop - 5); ctx.lineTo(x, paddingTop); ctx.stroke();
    }

    let y = paddingTop;
    for (const grp of SPEC.ringGroups) {
      ctx.strokeStyle = 'rgb(0,60,140)'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(paddingLeft, y); ctx.lineTo(paddingLeft + imgWidth, y); ctx.stroke();
      if (RING_SUBLINES[grp.name]) {
        const yy = y + grp.destHeight / 2;
        ctx.strokeStyle = 'rgb(200,200,200)'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(paddingLeft, yy); ctx.lineTo(paddingLeft + imgWidth, yy); ctx.stroke();
      }
      y += grp.destHeight;
    }
    ctx.strokeStyle = 'rgb(0,60,140)'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(paddingLeft, y); ctx.lineTo(paddingLeft + imgWidth, y); ctx.stroke();

    ctx.fillStyle = '#000000';
    ctx.font = '10px sans-serif';
    for (let m = 0; m <= 60; m += 5) {
      const x = paddingLeft + (m / 60) * imgWidth;
      const text = String(m);
      ctx.fillText(text, x - ctx.measureText(text).width / 2, paddingTop - 10);
    }

    ctx.fillStyle = 'rgb(150,0,0)';
    for (let s = 0; s < nSectors; s++) {
      const xc = paddingLeft + (s + 0.5) * sectorW;
      const text = 'S' + (s + 1);
      ctx.fillText(text, xc - ctx.measureText(text).width / 2, paddingTop - 30);
    }

    y = paddingTop;
    for (const grp of SPEC.ringGroups) {
      ctx.fillStyle = '#000000'; ctx.font = '11px sans-serif';
      ctx.fillText(grp.name, 5, y + grp.destHeight / 2 + 4);
      if (RING_SUBLINES[grp.name]) {
        const [rA, rB] = RING_SUBLINES[grp.name];
        ctx.font = '9px sans-serif'; ctx.fillStyle = 'rgb(90,90,90)';
        ctx.fillText('R' + rA, paddingLeft - 32, y + grp.destHeight / 4 + 4);
        ctx.fillText('R' + rB, paddingLeft - 32, y + (3 * grp.destHeight) / 4 + 4);
      }
      y += grp.destHeight;
    }

    ctx.font = 'bold 16px sans-serif';
    ctx.fillStyle = '#000000';
    ctx.fillText(side === 'R' ? 'RIGHT EYE' : 'LEFT EYE', paddingLeft, canvasHeight - 10);

    const nm = side === 'R' ? 45 : 15;
    const tm = side === 'R' ? 15 : 45;
    ctx.font = '12px sans-serif';
    ctx.fillStyle = 'rgb(0,0,255)';
    ctx.fillText('^ NASAL', paddingLeft + (nm / 60) * imgWidth - 30, canvasHeight - 10);
    ctx.fillStyle = 'rgb(100,100,100)';
    ctx.fillText('^ TEMPORAL', paddingLeft + (tm / 60) * imgWidth - 40, canvasHeight - 10);

    return canvas;
  }

  // ----------------------------------------------------------------- gate

  /**
   * Refuse to produce a strip we cannot trust. A confidently mislabelled analysis
   * is worse than asking for another photo, because every organ attribution
   * downstream inherits the error invisibly.
   */
  function validate(pupil, iris, frameW, frameH, maskedFraction, pupilOcclusion) {
    const v = SPEC.validity;
    const reasons = [];

    if (!pupil) return { ok: false, reasons: ['PUPIL_NOT_FOUND'] };
    if (pupil.circularity < SPEC.pupil.minCircularity) reasons.push('PUPIL_NOT_ROUND');
    if (!iris || iris.gradient < SPEC.iris.minGradient) reasons.push('IRIS_NOT_FOUND');
    if (iris && iris.r < v.minIrisRadiusPx) reasons.push('IRIS_TOO_SMALL');

    if (iris && iris.r > 0) {
      const ratio = pupil.r / iris.r;
      if (ratio < v.minPupilIrisRatio || ratio > v.maxPupilIrisRatio) reasons.push('RATIO_IMPLAUSIBLE');
      const margin = iris.r * v.edgeMarginFactor;
      if (pupil.x < margin || pupil.y < margin ||
          pupil.x > frameW - margin || pupil.y > frameH - margin) {
        reasons.push('IRIS_OUT_OF_FRAME');
      }
    }
    if (maskedFraction !== undefined && maskedFraction > v.maxMaskedFraction) reasons.push('TOO_OCCLUDED');
    if (pupilOcclusion !== undefined && pupilOcclusion > v.maxPupilRingOcclusion) reasons.push('PUPIL_NOT_CLEAR');

    return { ok: reasons.length === 0, reasons: reasons };
  }

  // ------------------------------------------------------------ orchestrator

  /**
   * Full pipeline for one eye.
   * Returns { ok, code, message, strip (canvas), quality, geometry } — and on
   * failure, actionable Bulgarian guidance instead of a silently bad strip.
   */
  function analyzeEye(srcMat, side, stripCanvas) {
    const gray = toGray(srcMat);
    const enhanced = enhance(gray);

    const candidates = findPupilCandidates(enhanced);
    if (!candidates.length) {
      gray.delete(); enhanced.delete();
      return { ok: false, code: 'PUPIL_NOT_FOUND', message: REJECT_MESSAGES.PUPIL_NOT_FOUND };
    }

    // Keep the first pupil hypothesis whose iris pairing actually validates,
    // rather than trusting a single threshold to have found the real pupil.
    let pupil = null, iris = null, firstFailure = null;
    for (const cand of candidates) {
      const candIris = findIrisOuter(gray, cand.x, cand.y, cand.r);
      const pre = validate(cand, candIris, srcMat.cols, srcMat.rows, undefined);
      if (pre.ok) { pupil = cand; iris = candIris; break; }
      if (!firstFailure) firstFailure = pre.reasons;
    }

    if (!pupil) {
      gray.delete(); enhanced.delete();
      const reasons = firstFailure || ['PUPIL_NOT_FOUND'];
      const code = reasons[0];
      return { ok: false, code: code, message: REJECT_MESSAGES[code] || 'Снимката не е подходяща за анализ.', reasons: reasons };
    }

    const pupilOcclusion = pupilClearanceOcclusion(gray, pupil.x, pupil.y, pupil.r, iris.r);

    const lids = fitEyelids(enhanced, pupil.x, pupil.y, iris.r, pupil.r);
    const roll = estimateRoll(lids.coefUpper, lids.coefLower, pupil.x, iris.r);
    const masked = buildMask(srcMat.cols, srcMat.rows, pupil.x, pupil.y, iris.r, lids.coefUpper, lids.coefLower);

    const un = unwrap(srcMat, masked.mask, pupil.x, pupil.y, pupil.r, iris.r, roll.rollRad);

    // Judge on the worse of what we masked and what the lids actually cover.
    const occlusion = Math.max(un.maskedFraction, masked.trueOccluded);
    const post = validate(pupil, iris, srcMat.cols, srcMat.rows, occlusion, pupilOcclusion);
    if (!post.ok) {
      gray.delete(); enhanced.delete(); un.mat.delete();
      const code = post.reasons[0];
      return { ok: false, code: code, message: REJECT_MESSAGES[code] || 'Снимката не е подходяща за анализ.', reasons: post.reasons };
    }

    renderGrid(un.mat, side, stripCanvas);

    const rollDeg = (roll.rollRad * 180) / Math.PI;
    const result = {
      ok: true,
      strip: stripCanvas,
      geometry: {
        pupil: pupil,
        iris: iris,
        rollDeg: rollDeg,
        rollSource: roll.source,
        corners: roll.corners,
        lidsDetected: !!(lids.coefUpper && lids.coefLower),
        pupilRingOcclusion: Math.round(pupilOcclusion * 1000) / 1000,
      },
      quality: {
        // Honest composite: edge strength, pupil roundness, visible area, and
        // whether we could establish an anatomical rotation reference at all.
        irisConfidence: iris.confidence,
        pupilCircularity: pupil.circularity,
        visibleFraction: 1 - occlusion,
        rollCorrected: roll.source === 'canthi',
        score: Math.round(100 * (
          0.35 * iris.confidence +
          0.20 * Math.min(1, pupil.circularity / 0.85) +
          0.30 * Math.max(0, 1 - occlusion / SPEC.validity.maxMaskedFraction) +
          0.15 * (roll.source === 'canthi' ? 1 : 0)
        )),
      },
    };

    gray.delete(); enhanced.delete(); un.mat.delete();
    return result;
  }

  global.IrisGeometry = {
    SPEC: SPEC,
    DEST_CONTENT_H: DEST_CONTENT_H,
    REJECT_MESSAGES: REJECT_MESSAGES,
    toGray: toGray,
    findPupil: findPupil,
    findIrisOuter: findIrisOuter,
    fitEyelids: fitEyelids,
    estimateRoll: estimateRoll,
    pupilClearanceOcclusion: pupilClearanceOcclusion,
    buildMask: buildMask,
    unwrap: unwrap,
    renderGrid: renderGrid,
    validate: validate,
    analyzeEye: analyzeEye,
    polyfit2: polyfit2,
    polyval2: polyval2,
  };
})(typeof window !== 'undefined' ? window : globalThis);
