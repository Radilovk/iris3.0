"""
Synthetic eye generator used by the geometry tests.

It renders an eye with known ground truth (pupil/iris radii, camera roll, lid
position) so the pipeline can be checked against numbers we actually know,
rather than against whatever it happened to output last time.
"""
import numpy as np


def synth_eye(w=600, h=400, cx=300, cy=200, pupil_r=40, iris_r=140,
              roll_deg=0.0, lid_cut=0.8, corner_span_factor=1.3,
              iris_gray=110, pupil_gray=18, sclera_gray=235, lid_gray=55,
              seed=0, marks=None):
    """
    Render a synthetic eye.

    lid_cut: how far the lids reach toward the centre, as a fraction of iris_r
             (0.8 => lids bite into the outer 20% of the iris).
    corner_span_factor: canthi sit at +/- this * iris_r from the centre, which
             sets the lid curvature.
    marks: optional list of (angle_deg, radius_frac, size, gray) blobs painted
             into the iris — used as ground-truth "findings".
    """
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), sclera_gray, np.uint8)

    Y, X = np.ogrid[:h, :w]
    d = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    # Iris: radial fibre-like texture so gradients behave plausibly.
    tex = (rng.random((h, w)) * 35 + iris_gray).astype(np.uint8)
    iris_area = (d <= iris_r) & (d > pupil_r)
    for c in range(3):
        ch = img[:, :, c]
        ch[iris_area] = tex[iris_area]

    img[d <= pupil_r] = (pupil_gray, pupil_gray, pupil_gray)

    # Ground-truth marks (dark blobs at known polar positions).
    if marks:
        for (angle_deg, r_frac, size, gray) in marks:
            # angle 0 = 12 o'clock, clockwise, in ANATOMICAL frame
            a = np.deg2rad(angle_deg - 90.0 + roll_deg)
            rr = pupil_r + r_frac * (iris_r - pupil_r)
            mx = int(cx + rr * np.cos(a))
            my = int(cy + rr * np.sin(a))
            blob = (X - mx) ** 2 + (Y - my) ** 2 <= size ** 2
            img[blob] = (gray, gray, gray)

    # Eyelids: parabolas whose intersections (the canthi) define the roll axis.
    a_off = lid_cut * iris_r
    k = a_off / ((corner_span_factor * iris_r) ** 2)
    t = np.tan(np.deg2rad(roll_deg))

    xs = np.arange(w)
    xx = xs - cx
    yu = cy - a_off + k * xx ** 2 + t * xx
    yl = cy + a_off - k * xx ** 2 + t * xx

    for x in xs:
        top = int(np.clip(yu[x], 0, h))
        bot = int(np.clip(yl[x], 0, h))
        if top > 0:
            img[:top, x] = (lid_gray, lid_gray, lid_gray)
        if bot < h:
            img[bot:, x] = (lid_gray, lid_gray, lid_gray)

    return img
