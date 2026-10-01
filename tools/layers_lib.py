"""Shared helpers for building rig layers."""
import cv2, numpy as np, os, json

GEN = os.path.join(os.path.dirname(__file__), '..', 'gen')

def imread(p):
    im = cv2.imread(p if os.path.isabs(p) else os.path.join(GEN, p), cv2.IMREAD_UNCHANGED)
    if im is None: raise FileNotFoundError(p)
    return im

def f32(im): return im.astype(np.float32) / 255.0

def load_rgb(p): return f32(imread(p)[..., :3])

def load_mask(p):
    m = imread(p)
    if m.ndim == 3: m = m[..., 0]
    return f32(m)

def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t)

def blur(m, s):
    if s <= 0: return m
    k = int(s * 3) * 2 + 1
    return cv2.GaussianBlur(m, (k, k), s)

def fast_blur(m, s, down=4):
    """Large-sigma blur done on a downscaled copy."""
    h, w = m.shape[:2]
    sm = cv2.resize(m, (w // down, h // down), interpolation=cv2.INTER_AREA)
    sm = blur(sm, s / down)
    return cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)

def norm_blur(img, w, s, down=2):
    """Normalized convolution: spread colors of weighted pixels outward."""
    w3 = w[..., None] if img.ndim == 3 else w
    num = fast_blur(img * w3, s, down); den = fast_blur(w, s, down)
    den3 = den[..., None] if img.ndim == 3 else den
    return num / np.maximum(den3, 1e-5), den

def extend_colors(img, alpha, passes=(2, 6, 18, 50)):
    """Fill RGB of low-alpha pixels with nearby opaque colors (avoids dark/grey fringes)."""
    out = img.copy(); w = (alpha > 0.9).astype(np.float32)
    filled = w.copy()
    for s in passes:
        c, d = norm_blur(img, w, s, down=1 if s < 4 else 2)
        need = (filled < 0.5) & (d > 1e-3)
        out[need] = c[need]; filled[need] = 1
    keep = alpha > 0.008
    out[keep] = img[keep]
    return out

def unmix(I, B, a, eps=0.03):
    """Given I = a*F + (1-a)*B, solve F."""
    a3 = np.maximum(a, eps)[..., None]
    return np.clip((I - (1 - a[..., None]) * B) / a3, 0, 1)

def known_bg_alpha(I, B, F):
    d = F - B
    den = (d * d).sum(2)
    a = ((I - B) * d).sum(2) / np.maximum(den, 1e-6)
    return np.clip(a, 0, 1), den

def poly_mask(shape, pts, blur_px=0):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.array(pts, np.int32)], 255)
    m = f32(m)
    return blur(m, blur_px) if blur_px else m

def color_mask(seg, col, tol=60):
    return (np.abs(seg.astype(np.int32) - np.array(col[::-1])).max(2) < tol).astype(np.float32)

def dilate(m, r):
    if r <= 0: return m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m, k)

def erode(m, r):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.erode(m, k)

def fill_holes(m):
    m8 = (m > 0.5).astype(np.uint8) * 255
    h, w = m8.shape; ff = m8.copy(); msk = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, msk, (0, 0), 255)
    return ((m8 | cv2.bitwise_not(ff)) > 0).astype(np.float32)

def largest_components(m, n=1, min_area=0):
    m8 = (m > 0.5).astype(np.uint8)
    k, lab, st, _ = cv2.connectedComponentsWithStats(m8, 8)
    order = sorted(range(1, k), key=lambda i: -st[i, cv2.CC_STAT_AREA])
    out = np.zeros_like(m, np.float32)
    for i in order[:n]:
        if st[i, cv2.CC_STAT_AREA] >= min_area: out[lab == i] = 1
    return out

class PartWriter:
    def __init__(self, outdir, scale=1.0):
        self.outdir = outdir; os.makedirs(outdir, exist_ok=True); self.parts = {}; self.scale = scale

    def save(self, name, rgb, alpha, pad=8, thresh=3 / 255, scale=None, **meta):
        """Crop to alpha bbox, write PNG (straight alpha), record bbox in model px."""
        sc = self.scale if scale is None else scale
        # drop isolated specks (tiny components) before computing the bbox
        m8 = (alpha > thresh).astype(np.uint8)
        k, lab, st, _ = cv2.connectedComponentsWithStats(m8, 8)
        small = np.where(st[:, cv2.CC_STAT_AREA] < 40)[0]
        if len(small) > 0:
            kill = np.isin(lab, small[small > 0]); alpha = alpha.copy(); alpha[kill] = 0
        ys, xs = np.where(alpha > thresh)
        if len(xs) == 0: print('EMPTY part', name); return
        H, W = alpha.shape
        x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
        y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
        rgb_e = extend_colors(rgb[y0:y1, x0:x1], alpha[y0:y1, x0:x1])
        a = alpha[y0:y1, x0:x1]
        out = np.dstack([np.clip(rgb_e * 255 + 0.5, 0, 255).astype(np.uint8), np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8)])
        if sc != 1.0:
            out = cv2.resize(out, (max(1, round((x1 - x0) * sc)), max(1, round((y1 - y0) * sc))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(self.outdir, name + '.png'), out, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        self.parts[name] = dict(file=name + '.png', x=int(x0), y=int(y0), w=int(x1 - x0), h=int(y1 - y0), **meta)
        print(f'part {name:14s} bbox=({x0},{y0})-({x1},{y1}) {x1-x0}x{y1-y0}')

    def dump(self, path, extra):
        json.dump(dict(parts=self.parts, **extra), open(path, 'w'), indent=1)

def clamp_chroma(F, ref, tol=0.04):
    """Keep F's luminance, but keep its chroma within tol of the reference colour's chroma."""
    lf = F.mean(2, keepdims=True); lr = ref.mean(2, keepdims=True)
    cf = F - lf; cr = ref - lr
    return np.clip(lf + np.clip(cf, cr - tol, cr + tol), 0, 1)
