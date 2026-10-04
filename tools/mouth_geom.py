"""Mouth contour measurement: outer lip line, inner (opening) edges and corners of one pass.

Every pass is described in its own mouth coordinate u in [-1, 1] (left corner .. right corner), sampled at
N points, so passes with different widths can be compared and blended point for point."""
import numpy as np, cv2
from layers_lib import largest_components, fill_holes

N = 41
U = np.linspace(-1, 1, N)
ZONE = (1215, 1440, 1570, 1910)          # y0, y1, x0, x1 search window (model px)
LIP_A = 20.0                             # LAB a* level separating lip from skin


def _lab(img):
    lab = cv2.cvtColor((img * 255).astype(np.uint8) if img.dtype != np.uint8 else img, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab[..., 0], cv2.GaussianBlur(lab[..., 1] - 128, (0, 0), 1.0)


def _smooth(v, s=1.6):
    v = np.asarray(v, np.float32)
    pad = int(s * 3) + 1
    vp = np.pad(v, pad, mode='edge')
    return cv2.GaussianBlur(vp.reshape(1, -1), (0, 0), sigmaX=s).ravel()[pad:-pad]


def _crossing(prof, lvl, i0, step):
    """Subpixel position where prof first exceeds lvl walking from i0 by step."""
    i = i0
    while 0 <= i < len(prof) and prof[i] < lvl: i += step
    if not (0 <= i < len(prof)): return None
    j = i - step
    if not (0 <= j < len(prof)): return float(i)
    t = (lvl - prof[j]) / max(prof[i] - prof[j], 1e-6)
    return j + t * step


def measure(img, closed, dark=125):
    """Returns corners + T (lip top), Su (upper lip inner edge), Sl (lower lip inner edge), B (lip bottom) at U."""
    y0, y1, x0, x1 = ZONE
    L, a = _lab(img[y0:y1, x0:x1])
    hull = fill_holes(largest_components((a > LIP_A).astype(np.float32), 1))
    hull = cv2.morphologyEx(hull, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    cols = np.where(hull.any(0))[0]
    cl, cr = cols.min(), cols.max()
    def cy(c):
        rows = np.where(hull[:, c] > 0)[0]; return rows.mean()
    corner_l = (x0 + cl, y0 + np.mean([cy(c) for c in range(cl, cl + 3)]))
    corner_r = (x0 + cr, y0 + np.mean([cy(c) for c in range(cr - 2, cr + 1)]))
    # opening: dark cavity or bright teeth, inside the lips
    def open_mask(lvl):
        m = (((L < lvl) | ((a < 20) & (L > 140))) & (hull > 0)).astype(np.float32)
        return largest_components(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), 1)
    # the upper edge is crisp; the lower lip's inner shadow is dark too, so its edge needs a stricter level
    opening = open_mask(125); opening_lo = open_mask(dark)
    xs = corner_l[0] + (U + 1) / 2 * (corner_r[0] - corner_l[0])
    T, B, Su, Sl, has = [], [], [], [], []
    for x in xs:
        c = int(round(x - x0)); c = min(max(c, cl), cr)
        rows = np.where(hull[:, c] > 0)[0]
        prof = a[:, c]
        t = _crossing(prof, LIP_A, rows.min() - 4, 1); b = _crossing(prof, LIP_A, rows.max() + 4, -1)
        T.append(y0 + (t if t is not None else rows.min())); B.append(y0 + (b if b is not None else rows.max()))
        orow = np.where(opening[:, c] > 0)[0]; lrow = np.where(opening_lo[:, c] > 0)[0]
        if not closed and len(orow) >= 2:
            lo = lrow.max() if len(lrow) else orow.max()
            Su.append(y0 + orow.min() - 0.5); Sl.append(y0 + max(lo, orow.min() + 1) + 0.5); has.append(1)
        else:
            # closed: the seam is the darkest row between the outer contours (parabolic subpixel)
            r0, r1 = rows.min() + 2, rows.max() - 2
            if r1 <= r0: s = rows.mean()
            else:
                col = cv2.GaussianBlur(L[:, max(c - 1, 0):c + 2], (0, 0), 0.8).mean(1)
                k = r0 + int(np.argmin(col[r0:r1]))
                d = col[k - 1] - 2 * col[k] + col[k + 1]
                s = k + (0.5 * (col[k - 1] - col[k + 1]) / d if d > 1e-6 else 0)
            Su.append(y0 + s); Sl.append(y0 + s); has.append(0)
    T, B, Su, Sl = [np.array(v, np.float32) for v in (T, B, Su, Sl)]
    has = np.array(has, np.float32)
    # corners where the opening pinches off: inner edges meet on the line between the outer contours
    if not closed:
        mid = (Su + Sl) / 2
        for i in range(N):
            if has[i] == 0: Su[i] = Sl[i] = mid[i] if np.isfinite(mid[i]) else (T[i] + B[i]) / 2
    # pin the ends onto the corner points and smooth along u
    for v in (T, B, Su, Sl):
        v[0] = corner_l[1]; v[-1] = corner_r[1]
    T, B = _smooth(T), _smooth(B); Su, Sl = _smooth(Su, 2.6), _smooth(Sl, 2.6)   # inner edges are noisier
    for v in (T, B, Su, Sl):
        v[0] = corner_l[1]; v[-1] = corner_r[1]
    Su = np.minimum(Su, Sl); T = np.minimum(T, Su); B = np.maximum(B, Sl)
    return dict(xl=float(corner_l[0]), yl=float(corner_l[1]), xr=float(corner_r[0]), yr=float(corner_r[1]),
                T=T, Su=Su, Sl=Sl, B=B, hull=hull, opening=opening, off=(x0, y0))


def chin_y(img, xs=(1700, 1750, 1800)):
    g = cv2.GaussianBlur(img.mean(2), (5, 5), 0)
    return float(np.mean([1380 + np.argmin(np.diff(g[1380:1520, x])) for x in xs]))
