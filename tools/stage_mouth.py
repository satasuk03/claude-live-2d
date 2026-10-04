"""Stage 3: the rigged mouth. Runs after stage_parts.py.

Splits the master's closed mouth into upper / lower lip layers over a lip-free face, builds the mouth interior
(from the "A" pass) and the upper teeth (from the "E" pass), and measures every mouth pass as a keyform: the lip
contours, corners, teeth reveal and chin drop the rig blends between. At rest face + lips recombine to the
master exactly (up to 8-bit rounding)."""
import numpy as np, cv2, os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from layers_lib import *
from mouth_geom import measure, chin_y, N, U

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUTDIR = os.path.join(ROOT, 'web', 'assets', 'parts')
PJ = os.path.join(ROOT, 'web', 'assets', 'parts.json')
PV = os.environ.get('PREVIEW_DIR')
pw = PartWriter(OUTDIR)

M = load_rgb('master.jpg'); H, W = M.shape[:2]
PASSES = {  # name: (image, closed?, darkness level for the lower inner edge)
    'A': ('aligned/mouthA.png', False, 125), 'O': ('aligned/mouthO.png', False, 95), 'E': ('aligned/mouthE.png', False, 95),
    'U': ('aligned/mouthU.png', True, 125), 'smile': ('aligned/smile.png', False, 95)}
IMG = {k: load_rgb(f) for k, (f, _, _) in PASSES.items()}
Y, X = np.mgrid[0:H, 0:W].astype(np.float32)


def full(mask, m):
    out = np.zeros((H, W), np.float32); x0, y0 = m['off']; h, w = mask.shape
    out[y0:y0 + h, x0:x0 + w] = mask; return out


def lab(img):
    l = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    return l[..., 0], l[..., 1] - 128


def curve_at(m, x, name):
    xs = m['xl'] + (U + 1) / 2 * (m['xr'] - m['xl'])
    return np.interp(x, xs, m[name])


# ---------------------------------------------------------------- measure every pass
rest = measure(M, True)
keys = {k: measure(IMG[k], c, d) for k, (_, c, d) in PASSES.items()}
# the face is frontal, so the open shapes' inner edges are mirror-averaged about the corner line: shadow in one
# corner otherwise reads as a lopsided opening (the smile pass measured flat on the left, dipped on the right)
for k, m in keys.items():
    if PASSES[k][1]: continue
    line = m['yl'] + (U + 1) / 2 * (m['yr'] - m['yl'])
    for name in ('Su', 'Sl'):
        r = m[name] - line; m[name] = (line + (r + r[::-1]) / 2).astype(np.float32)
    m['Su'] = np.minimum(m['Su'], m['Sl']); m['T'] = np.minimum(m['T'], m['Su']); m['B'] = np.maximum(m['B'], m['Sl'])
chin0 = chin_y(M)

# ---------------------------------------------------------------- skin under the lips (membrane fill of the master)
lips = full(rest['hull'], rest)
R = dilate(lips, 9)
for cx_, cy_ in ((rest['xl'], rest['yl']), (rest['xr'], rest['yr'])):   # corner creases ride with the corners
    cv2.circle(R, (int(round(cx_)), int(round(cy_))), 16, 1, -1)
ys, xs = np.where(R > 0); by0, by1, bx0, bx1 = ys.min() - 4, ys.max() + 5, xs.min() - 4, xs.max() + 5
sl = (slice(by0, by1), slice(bx0, bx1))
unk = R[sl] > 0
skin = M[sl].copy()
skin[unk] = norm_blur(M[sl], (~unk).astype(np.float32), 12, down=1)[0][unk]
for s in (8, 4, 2, 1):                       # coarse-to-fine Jacobi relaxation of the Laplace equation
    k = 2 * s + 1
    for _ in range(400 // s):
        sm = cv2.blur(skin, (k, k))
        skin[unk] = sm[unk]
B = M.copy(); B[sl] = skin

# ---------------------------------------------------------------- lip alpha / colour: exact unmix over the skin
core = erode(lips, 1)
lipF = norm_blur(M, core, 6, down=1)[0]
a_est, _ = known_bg_alpha(M, B, lipF)
a_est = np.maximum(blur(a_est, 0.8), core)
# smallest alpha that keeps the unmixed colour inside [0, 1] for every channel
d = M - B
a_min = np.where(d < 0, -d / np.maximum(B, 1e-4), d / np.maximum(1 - B, 1e-4)).max(2)
a = np.clip(np.maximum(a_est, a_min), 0, 1) * R
a = np.round(a * 255) / 255
F = unmix(M, B, a, eps=1 / 255)

# split along the seam. The upper lip keeps the dark seam line (it reads as the shadow under the upper lip once
# the mouth opens) and ends in a 1 px anti-aliased edge; the lower lip starts under it, its own soft edge hidden
# beneath the upper lip's opaque part, so at rest the pair recombines exactly
seam = curve_at(rest, X[0], 'Su')
x_l, x_r = rest['xl'], rest['xr']
seam = np.where(X[0] < x_l, rest['yl'], np.where(X[0] > x_r, rest['yr'], seam))[None, :]
# along the seam the lips are opaque right out through the corner creases: a translucent unmixed fringe there
# would land on the other lip once the corners move and show up as a bright sliver
band = (np.abs(Y - seam) <= 3) * (X >= x_l - 8) * (X <= x_r + 8) * R
a = np.maximum(a, band); F = np.where(band[..., None] > 0, M, F)
upper = np.clip(seam + 2.0 - Y, 0, 1)
lower = np.clip(Y - (seam - 0.5), 0, 1)
# where the lips themselves are translucent (corner tips) layering two partial copies would double them up:
# there each pixel belongs to exactly one lip
hard = a < 1
upper = np.where(hard, (Y <= seam).astype(np.float32), upper)
lower = np.where(hard, 1 - upper, lower)
pw.save('lip_upper', F, a * upper, pad=4)
pw.save('lip_lower', F, a * lower, pad=4)

# face: master with the lips replaced by skin
fp = json.load(open(PJ))
fi = fp['parts']['face']
face = cv2.imread(os.path.join(OUTDIR, fi['file']), cv2.IMREAD_UNCHANGED)
fy, fx = fi['y'], fi['x']
crop = (slice(by0 - fy, by1 - fy), slice(bx0 - fx, bx1 - fx))
face[crop][..., :3] = np.where(unk[..., None], np.clip(skin * 255 + 0.5, 0, 255).astype(np.uint8), face[crop][..., :3])
cv2.imwrite(os.path.join(OUTDIR, fi['file']), face, [cv2.IMWRITE_PNG_COMPRESSION, 6])

# ---------------------------------------------------------------- mouth interior (from the A pass, teeth painted out)
kA = keys['A']; IA = IMG['A']
openA = full(kA['opening'], kA)
LA, aA = lab(IA)
teethA = dilate(((aA < 20) & (LA > 140)).astype(np.float32) * openA, 2)
cav = erode(openA, 3) * (1 - teethA)                    # away from the lips' inner rims, which are lip coloured
inner = IA.copy()
fill = norm_blur(IA, cav, 5, down=1)[0] * 0.82          # behind the teeth: the darker roof of the mouth
inner = np.where(teethA[..., None] > 0, fill, inner)
inner = extend_colors(inner, cav, passes=(2, 5, 12, 30))
# the far corners of the mouth are in deep shadow
uA = np.clip((X - kA['xl']) / (kA['xr'] - kA['xl']) * 2 - 1, -1.5, 1.5)
inner = inner * (1 - 0.55 * smoothstep(0.55, 1.0, np.abs(uA)))[..., None]
pw.save('mouth_inner', inner, dilate(openA, 16), pad=2)

# ---------------------------------------------------------------- upper teeth (from the E pass)
kE = keys['E']; IE = IMG['E']
openE = full(kE['opening'], kE)
LE, aE = lab(IE)
core_t = ((LE > 100) & (aE < 17)).astype(np.float32) * erode(openE, 1)
t = cv2.morphologyEx(core_t, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
t = largest_components(fill_holes(t), 1)
# soft matte: teeth are bright against the dark cavity, so luminance gives the edge coverage (and lets the
# dark gaps between teeth and the shaded side teeth fade into the cavity behind them)
soft = smoothstep(55, 130, LE) * (aE < 26) * dilate(t, 2)
roots = np.zeros_like(t)
for c in np.where(t.any(0))[0]:                         # roots continue up under the upper lip
    r0 = np.where(t[:, c] > 0)[0].min()
    roots[max(r0 - 16, 0):r0 + 2, c] = 1
teeth_a = np.maximum(soft, blur(roots, 1.0) * dilate(t, 2).any(0)[None, :])
teeth_rgb = extend_colors(IE, ((LE > 95) & (aE < 20)).astype(np.float32) * t, passes=(2, 5, 12))
pw.save('teeth', teeth_rgb, teeth_a, pad=2)


def teeth_bottom(m, img):
    """Lowest teeth pixel near the middle of the mouth (None if no teeth show)."""
    L_, a_ = lab(img)
    o = full(m['opening'], m)
    cx = (m['xl'] + m['xr']) / 2
    tm = ((a_ < 22) & (L_ > 140)).astype(np.float32) * o * (np.abs(X - cx) < 18)
    ys_ = np.where(tm.any(1))[0]
    return float(ys_.max() + 0.5) if tm.sum() > 40 else None


def pack(m):
    return {k: (np.round(m[k], 2).tolist() if isinstance(m[k], np.ndarray) else round(m[k], 2)) for k in ('xl', 'yl', 'xr', 'yr', 'T', 'Su', 'Sl', 'B')}


def cavity_rgb(m, img):
    L_, a_ = lab(img)
    c = erode(full(m['opening'], m), 3) * (1 - dilate(((a_ < 22) & (L_ > 120)).astype(np.float32), 3))
    return (img * c[..., None]).sum((0, 1)) / max(c.sum(), 1)


cavA = cavity_rgb(kA, IA)
tbE = teeth_bottom(kE, IE)
meta = {'N': N, 'rest': pack(rest), 'keys': {}, 'teethRef': {'cx': round((kE['xl'] + kE['xr']) / 2, 2), 'yb': tbE},
        'innerRef': pack(kA)}
for k, m in keys.items():
    tb = teeth_bottom(m, IMG[k])
    cxk = curve_at(m, (m['xl'] + m['xr']) / 2, 'Su')
    meta['keys'][k] = {**pack(m), 'chin': round(chin_y(IMG[k]) - chin0, 2),
                       'reveal': round(tb - float(cxk), 2) if tb is not None else -4.0,
                       'innerMul': np.round(np.clip(cavity_rgb(m, IMG[k]) / cavA, 0.3, 1.5)[::-1].astype(np.float64), 3).tolist() if not PASSES[k][1] else [1, 1, 1]}
    print(f"key {k:6s} corners ({m['xl']:.0f},{m['yl']:.0f})-({m['xr']:.0f},{m['yr']:.0f}) chin {meta['keys'][k]['chin']:+.1f} reveal {meta['keys'][k]['reveal']:+.1f}")

# ---------------------------------------------------------------- parts.json: swap the old mouth patches for the rig layers
for k in [k for k in fp['parts'] if k.startswith('mouth_') and k != 'mouth_inner']:
    del fp['parts'][k]
    p = os.path.join(OUTDIR, k + '.png')
    if os.path.exists(p): os.remove(p)
fp['parts'].update(pw.parts)
fp['meta']['mouth'] = meta
import re
txt = json.dumps(fp, indent=1)
txt = re.sub(r'\[\s*(-?[\d.]+(?:,\s*-?[\d.]+)*)\s*\]', lambda m: '[' + ', '.join(m.group(1).split()).replace(',,', ',') + ']', txt)
open(PJ, 'w').write(txt)

# ---------------------------------------------------------------- rest check: face + lower + upper over the master
def load_part(n):
    p = fp['parts'][n]; im = f32(cv2.imread(os.path.join(OUTDIR, p['file']), cv2.IMREAD_UNCHANGED))
    return im, (slice(p['y'], p['y'] + p['h']), slice(p['x'], p['x'] + p['w']))
fim, fsl = load_part('face')
canvas = M.copy(); canvas[fsl] = fim[..., :3] * fim[..., 3:] + canvas[fsl] * (1 - fim[..., 3:])
for n in ('lip_lower', 'lip_upper'):
    im, s_ = load_part(n); canvas[s_] = im[..., :3] * im[..., 3:] + canvas[s_] * (1 - im[..., 3:])
err = np.abs(canvas - M)[by0:by1, bx0:bx1] * 255
print(f'rest error in the mouth box: max {err.max():.2f}/255, mean {err.mean():.3f}/255')
if PV:
    z = (slice(1230, 1420), slice(1590, 1890))
    cv2.imwrite(os.path.join(PV, 'mouth_rest.png'), cv2.resize(np.hstack([canvas[z], M[z], B[z]]) * 255, None, fx=2, fy=2))
