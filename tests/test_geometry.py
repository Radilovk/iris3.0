"""
Ground-truth tests for the iris geometry pipeline.

The point of this file is that the reliability claims are falsifiable. Before it
existed, changes to the strip layout and prompts were shipped with no way to tell
whether localization had improved, stayed the same, or silently regressed.

Run: python3 tests/test_geometry.py
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app  # noqa: E402
from synth_eye import synth_eye  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAILURES = []


def check(name, condition, detail=''):
    status = 'PASS' if condition else 'FAIL'
    print(f'  [{status}] {name}' + (f' — {detail}' if detail else ''))
    if not condition:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
def test_spec_conformance():
    """iris-geometry.js inlines the spec; it must match config/geometry-spec.json."""
    print('\nSPEC CONFORMANCE (JS twin vs JSON source of truth)')
    with open(os.path.join(ROOT, 'config', 'geometry-spec.json'), encoding='utf-8') as f:
        spec = json.load(f)
    js = open(os.path.join(ROOT, 'iris-geometry.js'), encoding='utf-8').read()

    # Ring group layout is the part that silently breaks coordinates if it drifts.
    for grp in spec['ringGroups']:
        pattern = (rf"name:\s*'{grp['name']}',\s*srcStart:\s*{grp['srcStart']},\s*"
                   rf"srcEnd:\s*{grp['srcEnd']},\s*destHeight:\s*{grp['destHeight']}")
        check(f"JS ring group {grp['name']} matches spec", re.search(pattern, js) is not None)

    check('JS unwrap width matches spec', f"width: {spec['unwrap']['width']}" in js)
    check('JS unwrap height matches spec', f"height: {spec['unwrap']['height']}" in js)
    check('JS sector count matches spec', f"sectors: {spec['sectors']}" in js)
    for key in ('minPupilIrisRatio', 'maxPupilIrisRatio', 'maxMaskedFraction'):
        check(f'JS validity.{key} matches spec', f"{key}: {spec['validity'][key]}" in js)

    fracs = ', '.join(str(x) for x in spec['pupil']['darkFractions'])
    check('JS pupil.darkFractions matches spec', f'darkFractions: [{fracs}]' in js)
    check('JS eyelid.pupilMarginPx matches spec',
          f"pupilMarginPx: {spec['eyelid']['pupilMarginPx']}" in js)


# ---------------------------------------------------------------------------
def test_roll_estimation():
    """Roll must be recovered from the canthi within a couple of degrees."""
    print('\nROLL ESTIMATION (camera/head tilt recovered from eye corners)')
    for true_roll in (-15, -8, 0, 8, 15):
        img = synth_eye(roll_deg=true_roll)
        res = app.analyze_eye(img, 'R')
        if not res['ok']:
            check(f'roll {true_roll:+d}° accepted', False, res['code'])
            continue
        est = res['geometry']['rollDeg']
        err = abs(est - true_roll)
        check(f'roll {true_roll:+d}° recovered within 2°', err <= 2.0,
              f'estimated {est:+.1f}° (err {err:.1f}°)')


# ---------------------------------------------------------------------------
def _mark_sector(img, side, expected_sector, roll_deg):
    """Unwrap and report which sector the darkest column falls in."""
    h, w = img.shape[:2]
    import cv2
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = app.preprocess_image(img)
    pupil = app.find_pupil(enhanced)
    iris = app.find_iris_outer_boundary(gray, pupil['x'], pupil['y'], pupil['r'])
    coef_u, coef_l = app.fit_eyelids(enhanced, pupil['x'], pupil['y'], iris['r'], pupil['r'])
    roll = app.estimate_roll(coef_u, coef_l, pupil['x'], iris['r'])
    mask, _ = app.build_mask(h, w, pupil['x'], pupil['y'], iris['r'], coef_u, coef_l)
    unw, _ = app.unwrap_iris_fast(img, mask, pupil['x'], pupil['y'], pupil['r'],
                                  iris['r'], roll['roll_rad'])

    g = cv2.cvtColor(unw, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Ignore white (masked) columns; look for the darkest real content.
    g[g > 200] = np.nan
    with np.errstate(all='ignore'):
        col = np.nanmean(g, axis=0)
    col = np.nan_to_num(col, nan=255.0)
    x = int(np.argmin(col))
    sector = int(x / (app.UNWRAP_W / app.N_SECTORS)) + 1
    return sector


def test_sector_roundtrip():
    """
    A finding placed at a known anatomical angle must come back in the matching
    sector — including when the photo is tilted. This is the check that actually
    measures 'correct localization'.
    """
    print('\nSECTOR ROUND-TRIP (known finding -> expected sector)')
    # (anatomical angle, expected sector). Sector N spans 30*(N-1) .. 30*N degrees.
    cases = [(15, 1), (105, 4), (195, 7), (285, 10)]

    for roll in (0, 12):
        for angle, expected in cases:
            img = synth_eye(roll_deg=roll, marks=[(angle, 0.55, 14, 30)])
            got = _mark_sector(img, 'R', expected, roll)
            check(f'roll {roll:+d}°, mark at {angle}° -> S{expected}', got == expected,
                  f'got S{got}')


def test_roll_matters():
    """
    Guard against the correction being a no-op: with roll correction disabled a
    tilted capture should land in the wrong sector, proving the fix does work.
    """
    print('\nCONTROL: tilt without correction must mis-localize')
    import cv2
    angle, expected = (15, 1)
    img = synth_eye(roll_deg=20, marks=[(angle, 0.55, 14, 30)])
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = app.preprocess_image(img)
    pupil = app.find_pupil(enhanced)
    iris = app.find_iris_outer_boundary(gray, pupil['x'], pupil['y'], pupil['r'])
    mask, _ = app.build_mask(h, w, pupil['x'], pupil['y'], iris['r'], None, None)
    unw, _ = app.unwrap_iris_fast(img, mask, pupil['x'], pupil['y'], pupil['r'], iris['r'], 0.0)
    g = cv2.cvtColor(unw, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g[g > 200] = np.nan
    with np.errstate(all='ignore'):
        col = np.nan_to_num(np.nanmean(g, axis=0), nan=255.0)
    sector_uncorrected = int(int(np.argmin(col)) / (app.UNWRAP_W / app.N_SECTORS)) + 1

    corrected = _mark_sector(img, 'R', expected, 20)
    check('uncorrected 20° tilt lands off-target', sector_uncorrected != expected,
          f'uncorrected S{sector_uncorrected}')
    check('corrected 20° tilt lands on target', corrected == expected,
          f'corrected S{corrected}')


# ---------------------------------------------------------------------------
def test_eyelid_masking():
    """Lid regions must arrive as white, since every AI prompt keys off that."""
    print('\nEYELID MASKING')
    img = synth_eye(lid_cut=0.7)
    res = app.analyze_eye(img, 'R')
    check('capture accepted', res['ok'], res.get('code', ''))
    if res['ok']:
        vis = res['quality']['visibleFraction']
        check('some iris is masked by lids', vis < 0.98, f'visible {vis:.2f}')
        check('most of the iris survives', vis > 0.55, f'visible {vis:.2f}')

    # A half-closed eye must be refused rather than analysed as if fully visible.
    # Known limit: below roughly lid_cut 0.32 the lid edge reaches the pupil rim,
    # the lid search band collapses, and no fit is produced — such a capture keeps
    # lidsDetected=false and therefore scores low, but is not hard-rejected.
    for lid_cut in (0.5, 0.35):
        res2 = app.analyze_eye(synth_eye(lid_cut=lid_cut), 'R')
        check(f'half-closed eye (lid_cut={lid_cut}) rejected', not res2['ok'],
              res2.get('code', 'accepted'))


# ---------------------------------------------------------------------------
def test_validity_gate():
    """Bad captures must be refused rather than silently mislabelled."""
    print('\nVALIDITY GATE')
    tiny = synth_eye(w=600, h=400, pupil_r=8, iris_r=25)
    res = app.analyze_eye(tiny, 'R')
    check('too-small eye rejected', not res['ok'], res.get('code', 'accepted'))

    blank = np.full((400, 600, 3), 200, np.uint8)
    res2 = app.analyze_eye(blank, 'R')
    check('blank frame rejected', not res2['ok'], res2.get('code', 'accepted'))

    # A dark iris used to defeat the old fixed threshold of 40 entirely: the iris
    # disc itself fell below the cut and was returned as one huge "pupil".
    dark = synth_eye(iris_gray=38, pupil_gray=10)
    res3 = app.analyze_eye(dark, 'R')
    check('dark iris still analysable', res3['ok'], res3.get('code', ''))
    if res3['ok']:
        ratio = res3['geometry']['pupil']['r'] / res3['geometry']['iris']['r']
        check('dark iris: pupil not merged with iris', 0.15 < ratio < 0.75,
              f'pupil/iris ratio {ratio:.2f}')
        check('dark iris: pupil radius close to truth', abs(res3['geometry']['pupil']['r'] - 40) <= 8,
              f"got r={res3['geometry']['pupil']['r']} (truth 40)")


# ---------------------------------------------------------------------------
def test_determinism():
    """Same input must give the same geometry — otherwise results are not trustworthy."""
    print('\nDETERMINISM')
    img = synth_eye(roll_deg=7)
    a = app.analyze_eye(img, 'R')
    b = app.analyze_eye(img, 'R')
    check('two runs agree on roll', abs(a['geometry']['rollDeg'] - b['geometry']['rollDeg']) < 1e-6)
    check('two runs agree on iris radius', a['geometry']['iris']['r'] == b['geometry']['iris']['r'])
    check('two runs produce identical strips', np.array_equal(a['mapped'], b['mapped']))


if __name__ == '__main__':
    test_spec_conformance()
    test_roll_estimation()
    test_sector_roundtrip()
    test_roll_matters()
    test_eyelid_masking()
    test_validity_gate()
    test_determinism()

    print('\n' + '=' * 60)
    if FAILURES:
        print(f'{len(FAILURES)} FAILED: ' + ', '.join(FAILURES))
        sys.exit(1)
    print('All geometry tests passed.')
