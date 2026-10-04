"""Stage 2: build every rig part from the master + aligned edit passes."""
import numpy as np, cv2, os, sys, json
from scipy.ndimage import median_filter
sys.path.insert(0, os.path.dirname(__file__))
from layers_lib import *
from eyes_geom import EYE_L, EYE_R

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUTDIR = os.path.join(ROOT, 'web', 'assets', 'parts')
ST = os.path.join(GEN, 'stage')
PV = os.environ.get('PREVIEW_DIR', ST) + '/'
pw = PartWriter(OUTDIR)

M = load_rgb('master.jpg'); H, W = M.shape[:2]
NA = load_rgb('aligned/noarms.png'); CL = load_rgb('aligned/closed2_c.png')
SME = load_rgb('aligned/smile2_c.png'); EC = load_rgb('aligned/eyeclean_c.png')
NB = load_rgb('aligned/nobrow.png'); NE = load_rgb('aligned/noear.png')
BDc = load_rgb('stage/bald_clean.png'); BD = load_rgb('aligned/bald.png')
BDc_nb = load_rgb('stage/bald_nobrow.png')
SEG = imread('aligned/seg.png')
FIG = load_mask('mattes/fig.png'); FIG_NA = load_mask('mattes/fig_noarms.png'); FIG_BD = load_mask('mattes/fig_bald.png')
HA = np.load(os.path.join(ST, 'hair_a.npy')).astype(np.float32)
TA = np.load(os.path.join(ST, 'tiara_a.npy')).astype(np.float32)
HF = load_rgb('stage/hair_F.png'); HSRC = load_rgb('stage/hair_src.png')
EYE_EX = np.load(os.path.join(ST, 'eye_ex.npy')).astype(np.float32)
EARZ = np.load(os.path.join(ST, 'earz.npy')).astype(np.float32)
Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
meta = {}

def sharpen_matte(a, lo=0.04, hi=0.96): return smoothstep(lo, hi, a)

# ---------------------------------------------------------------- face / head masks
face_seg = color_mask(SEG, (255, 0, 0), 90)
face_seg = largest_components(face_seg, 1)
face_seg = (blur(fill_holes(face_seg), 3) > 0.5).astype(np.float32)
# jaw line per column = lowest face pixel
cols = np.where(face_seg.any(0))[0]
jaw = np.full(W, -1, np.float32)
for x in cols: jaw[x] = np.where(face_seg[:, x] > 0)[0].max()
fx0, fx1 = cols.min(), cols.max()
jaw_s = jaw.copy(); valid = jaw_s > 0
jaw_s[valid] = cv2.GaussianBlur(jaw_s[valid].reshape(1, -1), (1, 31), 0, sigmaY=6).ravel() if valid.sum() > 40 else jaw_s[valid]
EAR_BOTTOM = 1195
below_jaw = np.zeros((H, W), np.float32)
for x in range(W):
    if fx0 + 6 <= x <= fx1 - 6 and jaw_s[x] > EAR_BOTTOM: below_jaw[int(jaw_s[x]):, x] = 1
    else: below_jaw[EAR_BOTTOM:, x] = 1
head_mask = FIG_BD * (1 - below_jaw)
head_mask[:, :1250] = 0; head_mask[:, 2250:] = 0
head_mask = np.maximum(head_mask, face_seg * (1 - below_jaw))
head_mask = blur(head_mask, 1.2)
meta['jaw'] = {'x0': int(fx0), 'x1': int(fx1), 'chinY': float(jaw_s[valid].max())}
print('face cols', fx0, fx1, 'chin', jaw_s[valid].max())

# ---------------------------------------------------------------- eye geometry
eyes = {}
for side, poly, (icx, icy, ir) in [('L', EYE_L, (1590.5, 1043.5, 36.0)), ('R', EYE_R, (1878.5, 1039.5, 37.5))]:
    P = np.array(poly, np.float32)
    opening = poly_mask(M.shape, P, 0)
    op_soft = blur(opening, 1.0)
    x0, y0 = P.min(0); x1, y1 = P.max(0)
    eyes[side] = dict(opening=opening, op_soft=op_soft, iris=(icx, icy, ir), box=(x0, y0, x1, y1), poly=P)

# eye "patch" region: where open vs closed differ, around each eye
eye_patch = {}
dcl = np.abs(EC - CL).max(2)
for side, e in eyes.items():
    x0, y0, x1, y1 = e['box']
    zone = np.zeros((H, W), np.float32)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    cv2.ellipse(zone, (int(cx), int(cy - 6)), (int((x1 - x0) / 2 + 40), int((y1 - y0) / 2 + 42)), 0, 0, 360, 1, -1)
    d = smoothstep(0.05, 0.14, dcl) * zone
    d = fill_holes(dilate((d > 0.3).astype(np.float32), 9))
    d = np.maximum(d, dilate(e['opening'], 14))
    patch = blur(d * zone, 7)
    core = np.load(os.path.join(ST, 'eye_core.npy')).astype(np.float32)
    eye_patch[side] = np.minimum(patch, core)

# ---------------------------------------------------------------- brows
brow_seg = color_mask(SEG, (128, 64, 0), 70)
brows = {}
dnb = np.abs(M - NB).max(2)
for side, xr in [('L', (1440, 1750)), ('R', (1750, 2060))]:
    z = dilate(brow_seg, 26); z[:, :xr[0]] = 0; z[:, xr[1]:] = 0; z[:900] = 0; z[1010:] = 0
    z = blur(z, 6)
    a = smoothstep(0.035, 0.16, dnb) * z
    a = blur(a, 0.8)
    brows[side] = a

# ---------------------------------------------------------------- earrings
ear_boxes = {'L': (1405, 1130, 1505, 1310), 'R': (1970, 1125, 2075, 1305)}
earrings = {}
dne = np.abs(M - NE).max(2)
for side, (x0, y0, x1, y1) in ear_boxes.items():
    z = np.zeros((H, W), np.float32); z[y0:y1, x0:x1] = 1; z = blur(z, 3)
    seg_e = color_mask(SEG, (192, 192, 192), 45) * (z > 0.5)
    sz = blur(dilate(seg_e, 7), 2)
    a = smoothstep(0.05, 0.16, dne) * sz
    a = np.maximum(a, erode(seg_e, 4) * smoothstep(0.03, 0.08, dne))
    a = blur(np.clip(a, 0, 1), 0.6)
    earrings[side] = a

# ---------------------------------------------------------------- hair partition
hair_a = HA.copy()
key_core = erode((HA > 0.6).astype(np.float32), 22)
not_face = 1 - dilate(face_seg, 8)
hair_a = np.maximum(hair_a, blur(key_core, 8) * not_face)  # opaque crown interior
hair_a = np.clip(hair_a, 0, 1)
tiara_a = TA

def curve_mask(pts, above=True):
    """Region above (or below) a polyline through pts (x ascending)."""
    xs = np.array([p[0] for p in pts], np.float32); ys = np.array([p[1] for p in pts], np.float32)
    yc = np.interp(np.arange(W), xs, ys)
    return (Y < yc[None, :]).astype(np.float32) if above else (Y >= yc[None, :]).astype(np.float32)

pony_r = curve_mask([(1200, 700), (1350, 640), (1430, 560), (1510, 485), (1620, 448), (1750, 438), (1880, 448), (1990, 485), (2070, 560), (2150, 640), (2300, 700)])
bangs_r = poly_mask(M.shape, [(1425, 1130), (1418, 860), (1455, 760), (1530, 680), (1640, 630), (1750, 612), (1860, 630), (1970, 680), (2045, 760), (2082, 860), (2075, 1130)])
bangs_r = bangs_r * (1 - pony_r)
side_r = (Y > 940).astype(np.float32) * (1 - bangs_r) * (1 - pony_r)
crown_r = (1 - pony_r) * (1 - bangs_r) * (1 - side_r)
sideL_r = side_r * (X < 1750); sideR_r = side_r * (X >= 1750)

# hair colour source: unmixed foreground; under the tiara inpaint ponytail hair
hairF = HF.copy()
t8 = (dilate((tiara_a > 0.1).astype(np.float32), 4) * 255).astype(np.uint8)
hairF_inp = f32(cv2.inpaint((hairF * 255).astype(np.uint8), t8, 7, cv2.INPAINT_TELEA))
hairF = hairF_inp
hair_under_tiara = np.maximum(hair_a, dilate((tiara_a > 0.1).astype(np.float32), 3) * 0.999)
hair_a_full = hair_under_tiara
# the hair matte picked up bits of the open eyes' lashes; they would float over closed / smiling eyes. Drop matte
# blobs lying wholly around an eye (real strands crossing an eye corner run on into the bangs, so they stay)
eye_zone = dilate(np.maximum(eyes['L']['opening'], eyes['R']['opening']), 45) > 0.5
nlab, lab = cv2.connectedComponents((hair_a_full > 0.05).astype(np.uint8), connectivity=8)
outside = np.bincount(lab[~eye_zone], minlength=nlab)
stray = (outside == 0) & (np.bincount(lab.ravel(), minlength=nlab) > 0); stray[0] = False
stray_m = stray[lab].astype(np.float32)
print('stray lash blobs removed from the hair:', int(stray.sum()), 'px', int(stray_m.sum()))
hair_a_full = hair_a_full * (1 - blur(dilate(stray_m, 2), 1.0))
# lash bits still joined to the bangs: around the eyes, trust the hair-only pass (green there means no hair)
HP = imread('aligned/hair.png')[..., :3].astype(np.float32)
green = smoothstep(40, 110, HP[..., 1] - np.maximum(HP[..., 0], HP[..., 2]))
lash_band = blur(dilate(np.maximum(eyes['L']['opening'], eyes['R']['opening']), 24), 5)
hair_a_full = hair_a_full * (1 - blur(green, 1.5) * lash_band)

# underlap: parts drawn beneath a neighbour also carry a strip of that neighbour's hair,
# so no gap opens when they move apart
def shift_region(r, dy):
    out = np.zeros_like(r)
    if dy > 0: out[dy:] = r[:-dy]
    else: out[:dy] = r[-dy:]
    return out
pony_ext = np.clip(pony_r + (1 - pony_r) * np.maximum.reduce([shift_region(pony_r, d) for d in (30, 60, 90)]) * (1 - bangs_r), 0, 1)
side_up = np.maximum.reduce([shift_region(side_r, -d) for d in (30, 60)]) * crown_r
crown_ext = np.clip(crown_r + np.maximum.reduce([shift_region(crown_r, d) for d in (15, 30, 45)]) * bangs_r * (Y < 820), 0, 1)
sideL_ext = np.clip(sideL_r + side_up * (X < 1750), 0, 1); sideR_ext = np.clip(sideR_r + side_up * (X >= 1750), 0, 1)
for name, region, meta_extra in [
        ('ponytail', pony_ext, {}), ('hair_crown', crown_ext, {}), ('bangs', bangs_r, {})]:
    a = hair_a_full * region
    pw.save(name, hairF, a)

# side locks hang behind the neck and shoulders in the master, so their inner and bottom edges are occlusion cuts.
# The back copy (drawn behind the body) is the whole lock, continued behind the neck and down behind the shoulders;
# the front copy keeps what lies over the face (ending at the jaw), plus any strand that crosses the skin.
# the master's own skin outline around the hair (the bald pass's neck and shoulders are a little wider)
skin_sil = ((FIG_NA > 0.5) & (hair_a_full < 0.5) & (Y > 1150)).astype(np.float32)
skin_sil = cv2.morphologyEx(skin_sil, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
skin_sil = fill_holes(largest_components(skin_sil, 1))
occluder = skin_sil * below_jaw                                # neck + shoulders, below the head
occ_in = erode((occluder > 0.5).astype(np.float32), 4)
head_bottom = np.where((np.arange(W) >= fx0 + 6) & (np.arange(W) <= fx1 - 6) & (jaw_s > EAR_BOTTOM), jaw_s, EAR_BOTTOM)
keep_front = smoothstep(head_bottom[None, :] + 20, head_bottom[None, :] + 2, Y)   # below the jaw the lock hangs behind the neck
keep_front = keep_front * (1 - blur(dilate((occluder > 0.5).astype(np.float32), 2), 1.5))
# below the ears the locks hang behind the jaw: the matte's rim around the face outline is that occlusion edge, and
# drawn in front it would float off the cheek as a dark stroke when the head turns
face_hard = (head_mask > 0.5).astype(np.float32)
keep_front = keep_front * (1 - blur(dilate(face_hard, 22) - erode(face_hard, 10), 1.5) * smoothstep(EAR_BOTTOM - 60, EAR_BOTTOM - 20, Y))
# strands lying across the skin (real ones only: the matte's faint fringe along the neck would smear over it when the head turns)
keep_front = np.maximum(keep_front, occ_in * smoothstep(0.3, 0.6, hair_a_full))

def cut_line(a, occ, band, min_run=20):
    """Per row: the last opaque hair pixel with the body just past it (+x), smoothed along the edge; -1 where none.
    The hair must be at least `min_run` px thick there (a loose strand lying along the body is not a cut lock)."""
    near = np.zeros_like(occ)
    for k in range(1, 5): near[:, :-k] |= occ[:, k:]            # body within 4 px to the right
    solid = cv2.erode((a > 0.5).astype(np.uint8), np.ones((1, min_run), np.uint8), anchor=(min_run - 1, 0)) > 0
    xc = np.full(a.shape[0], -1)
    for r in range(a.shape[0]):
        cut = np.nonzero(solid[r] & near[r])[0]
        if len(cut): xc[r] = cut.max()
    idx = np.nonzero(xc >= 0)[0]
    if len(idx): xc[idx] = median_filter(xc[idx], size=9, mode='nearest')   # single rows snag on loose strands
    return xc

def wrap_on(rgb, a, occ, reach, band, dark, xfade=12):
    """Grow hair rightwards (+x) past its cut edges: each row restarts `band` px inside its cut (so the shaded cut edge
    goes too) and repeats its own last stretch of lock, as wide as the lock is there. Translated, so slanted strands
    keep their slant; after the first few px (blended in from a mirror image) the row joins up with itself seamlessly."""
    w = a.shape[1]; xc = cut_line(a, occ, band)
    rows = np.nonzero(xc >= 0)[0]
    if not len(rows): return
    # lock width per row (opaque run ending at the cut), smoothed down the edge
    wid = np.zeros(len(rows))
    for n, r in enumerate(rows):
        run = a[r, :xc[r] + 1][::-1] > 0.5
        wid[n] = np.argmin(run) if not run.all() else len(run)
    per = np.clip(median_filter(wid, size=15, mode='nearest') - band - 6, 24, 120).astype(int)
    for r, P in zip(rows, per):
        c = xc[r] - band
        if c < P: continue
        src_rgb, src_a = rgb[r].copy(), a[r].copy(); out_rgb, out_a = src_rgb.copy(), src_a.copy()
        for d0 in range(1, reach + band + 1, P):                # chunks: each reads what the previous one wrote
            d = np.arange(d0, min(d0 + P, reach + band + 1)); t = c + d; d, t = d[t < w], t[t < w]
            if not len(d): break
            wgt = smoothstep(0, xfade, d); m = np.maximum(c - d, 0)
            out_rgb[t] = src_rgb[m] * (1 - wgt)[:, None] + out_rgb[t - P] * wgt[:, None]
            out_a[t] = src_a[m] * (1 - wgt) + out_a[t - P] * wgt
        d = np.arange(1, reach + band + 1); t = c + d; t = t[t < w]
        past = np.maximum(t - xc[r], 0)
        ok = (past == 0) | occ[r, t]                           # the cut band, then only where the body covers it
        t, past = t[ok], past[ok]
        rgb[r, t] = out_rgb[t] * (1 - dark * past / reach)[:, None]
        a[r, t] = out_a[t] * smoothstep(reach, reach * 0.45, past)

def flow_down(rgb, a, occ, reach, band, dark):
    """Grow hair downwards past its cut edges by carrying each strand on in its own direction: every pixel below the
    cut takes the colour where its strand crossed the row `band` px above the cut. Strands stay straight lines."""
    h, w = a.shape
    yc = cut_line(a.T, occ.T, band)                             # per column
    cols = np.nonzero(yc >= 0)[0]
    if not len(cols): return
    # strand direction (dx per dy) from the structure tensor of the hair just above the cut
    L = (rgb @ np.float32([0.3, 0.59, 0.11])).astype(np.float32)
    gx, gy = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    wa = (a > 0.6).astype(np.float32)
    J = [cv2.blur(v * wa, (31, 61)) for v in (gx * gx, gx * gy, gy * gy)]
    ang = 0.5 * np.arctan2(2 * J[1], J[0] - J[2])               # dominant gradient angle; strands run across it
    slope = np.zeros(w, np.float32)
    for x in cols:
        y = max(0, yc[x] - band - 40); g = ang[y, x]
        slope[x] = np.clip(-np.sin(g) / (np.cos(g) + 1e-6) if abs(np.cos(g)) > 0.2 else 0, -0.8, 0.8)
    has = np.zeros(w, np.float32); has[cols] = 1
    num, den = cv2.GaussianBlur(slope[None] * has[None], (0, 0), 12), cv2.GaussianBlur(has[None], (0, 0), 12)
    slope = (num / np.maximum(den, 1e-6))[0]
    c = np.where(yc >= 0, yc - band, -1).astype(np.float32)
    cf = np.interp(np.arange(w), cols, c[cols]).astype(np.float32)   # source row for columns without their own cut
    Yg, Xg = np.mgrid[0:h, 0:w].astype(np.float32)
    d = Yg - cf[None, :]
    mx = (Xg - slope[None, :] * np.maximum(d, 0)).astype(np.float32)
    my = np.interp(mx, np.arange(w), cf).astype(np.float32)
    # sample a copy averaged over the 24 rows above (premultiplied): keeps each strand's colour but closes the little
    # gaps between the wisps at the cut, which would otherwise run on as empty stripes
    k = np.zeros((47, 1), np.float32); k[:24] = 1 / 24           # centre row 23 -> rows y-23..y
    sm = lambda v: cv2.filter2D(v, -1, k, borderType=cv2.BORDER_REPLICATE)
    pa = sm(a); prgb = sm(rgb * a[..., None]) / np.maximum(pa, 1e-4)[..., None]
    prgb = np.where(pa[..., None] > 1e-3, prgb, rgb).astype(np.float32)
    src_rgb = cv2.remap(prgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    src_a = cv2.remap(pa.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    past = np.maximum(Yg - np.where(yc >= 0, yc, 1e9)[None, :], 0)
    tgt = (d > 0) & (yc >= 0)[None, :] & (d <= reach + band) & ((Yg <= yc[None, :]) | occ)
    rgb[tgt] = (src_rgb * (1 - dark * np.minimum(past, reach) / reach)[..., None])[tgt]
    # the lock's sides taper off below the cut instead of ending in a straight wall (seen when the head tilts)
    side = cv2.GaussianBlur(has[None], (0, 0), 10)[0]
    taper = 1 - smoothstep(0, 40, past) * (1 - smoothstep(0.15, 0.85, side)[None, :])
    a[tgt] = (src_a * smoothstep(reach, reach * 0.45, past) * taper)[tgt]

def extend_lock(rgb, a, inward):
    """Continue a lock behind the body: inward behind the neck (repeating the lock), then down behind the shoulders (along the
    strands). Only fills where the body covers it at rest, plus the few px of shaded cut edge."""
    ys, xs = np.where(a > 0.6)
    y0, y1, x0, x1 = ys.min(), min(H, ys.max() + 340), max(0, xs.min() - 120), min(W, xs.max() + 120)
    sl = (slice(y0, y1), slice(x0, x1))
    rgb_c, a_c, occ = rgb[sl].copy(), a[sl].copy(), occluder[sl] > 0.5
    flip = (lambda v: v[:, ::-1]) if inward < 0 else (lambda v: v)
    R_, A_ = flip(rgb_c).copy(), flip(a_c).copy()
    wrap_on(R_, A_, flip(occ).copy(), 96, 6, 0.10)              # behind the neck
    rgb_c, a_c = flip(R_).copy(), flip(A_).copy()
    flow_down(rgb_c, a_c, occ, 320, 6, 0.18)                    # down behind the shoulders
    rgb = rgb.copy(); a = a.copy(); rgb[sl] = rgb_c; a[sl] = a_c
    return rgb, a

for s, region, inward in [('L', sideL_ext, 1), ('R', sideR_ext, -1)]:
    a = hair_a_full * region
    af = a * keep_front
    pw.save('side_' + s, hairF, af)
    # back alpha so that front over back recombines to the lock's own alpha at rest
    ab = np.clip(a * (1 - keep_front) / np.maximum(1 - af, 1e-3), 0, 1)
    ab = np.where(af > 0.995, a, ab)
    rgb_b, a_ext = extend_lock(hairF, a, inward)
    grown = (a_ext != a) | (rgb_b != hairF).any(2)
    pw.save('sideback_' + s, rgb_b, np.where(grown, a_ext, ab))

# tiara
pw.save('tiara', M, tiara_a)

# ---------------------------------------------------------------- hair back (synthesized behind-head mass)
crown_shape = ((hair_a_full * (1 - side_r)) > 0.5).astype(np.float32)
bald_head = FIG_BD * (Y < 1150) * (X > 1250) * (X < 2250)
back_shape = fill_holes(crown_shape)
back_shape = back_shape * (Y < 1150) * (Y > 300)
back_shape = largest_components(back_shape, 1)
back_shape = blur(erode(back_shape, 14), 4)
# colour: darkened, blurred crown hair stretched
base = norm_blur(hairF, (hair_a_full > 0.8).astype(np.float32) * (1 - side_r), 25)[0]
tex = hairF * (hair_a_full[..., None] > 0.8) + base * (hair_a_full[..., None] <= 0.8)
back_rgb = np.clip(tex * 0.8, 0, 1)
pw.save('hair_back', back_rgb, back_shape)

# ---------------------------------------------------------------- face layer (head skin)
face_rgb = M.copy()
def mix(dst, src, w): return dst * (1 - w[..., None]) + src * w[..., None]
brow_zone = blur(dilate(brow_seg, 30) * (Y > 880) * (Y < 1020), 8)
face_rgb = mix(face_rgb, BDc_nb, brow_zone)                  # no brows in base
face_rgb = mix(face_rgb, NE, EARZ)                          # no earrings in base
hair_cover = blur(dilate((hair_a_full > 0.02).astype(np.float32), 4), 3)
eye_keep = np.maximum(eye_patch['L'], eye_patch['R']) * np.load(os.path.join(ST, 'eye_core.npy')).astype(np.float32)
# forehead under the bangs: clean bald skin everywhere (the bangs' shadow becomes its own layer)
forehead = bangs_r * face_seg_soft if False else None
fore = blur(poly_mask(M.shape, [(1440, 700), (2060, 700), (2060, 1010), (1440, 1010)]), 6) * blur(dilate(face_seg, 6), 4)
under = np.maximum(hair_cover, fore)
EYE_CORE0 = np.load(os.path.join(ST, 'eye_core.npy')).astype(np.float32)
face_rgb = mix(face_rgb, BDc_nb, under * (1 - EYE_CORE0))   # bald skin under hair (no brows)
EYE_CORE = np.load(os.path.join(ST, 'eye_core.npy')).astype(np.float32)
for s_ in 'LR': face_rgb = mix(face_rgb, CL, blur(dilate((eye_patch[s_] > 0.02).astype(np.float32), 5), 3) * EYE_CORE)  # closed eyes in base
# cheeks: under the hair the skin is the bald pass, which draws its own cheek outline a few px inside its silhouette.
# Turning the head slides the hair off it (a dark line with a pale strip outside), so the face ends at that line.
Lf = face_rgb @ np.float32([0.3, 0.59, 0.11])
chin_y = int(meta['jaw']['chinY'])
face_a = head_mask.copy()
for side, xs_ in (('L', np.arange(1250, 1750)), ('R', np.arange(2249, 1750, -1))):
    rows, cuts = [], []
    for y in range(EAR_BOTTOM - 20, chin_y - 50):
        inside = xs_[head_mask[y, xs_] > 0.5]
        if len(inside) < 60: continue
        xb = inside[0]; step = 1 if side == 'L' else -1
        seg = Lf[y, xb:xb + 30 * step:step]
        ref = np.median(Lf[y, xb + 32 * step:xb + 48 * step:step])
        k = int(np.argmin(seg))
        if ref - seg[k] > 0.05: rows.append(y); cuts.append(k)
    if len(rows) < 10: continue
    rows = np.array(rows); cuts = median_filter(np.array(cuts, np.float32), size=15, mode='nearest')
    allr = np.arange(rows.min(), rows.max() + 1); cut = np.interp(allr, rows, cuts)
    print('cheek line', side, 'rows', rows.min(), rows.max(), 'depth px', cut.min(), cut.max())
    for y, k in zip(allr, cut):
        inside = xs_[head_mask[y, xs_] > 0.5]
        if not len(inside): continue
        dpx = np.abs(xs_ - inside[0])                          # px in from the outer edge (xs_ runs outer -> inner)
        fade = smoothstep(rows.min() - 1, rows.min() + 25, y) * smoothstep(rows.max() + 1, rows.max() - 25, y)
        keep = smoothstep(k - 1.0, k + 1.0, dpx)                # cut at the line's middle: half of it stays as a soft rim
        face_a[y, xs_] *= 1 - fade * (1 - keep)
pw.save('face', face_rgb, face_a)

# bangs shadow (multiply): whatever darkening the master shows on the forehead beyond the hair itself
ha = hair_a_full * (bangs_r + crown_r)
num = M - ha[..., None] * hairF
den = (1 - ha[..., None]) * face_rgb
mul = np.clip(num / np.maximum(den, 1e-3), 0.35, 1.0)
valid = (ha < 0.92) * fore * (1 - eye_keep) * head_mask
mul = mul * valid[..., None] + (1 - valid[..., None])
mul = blur(mul, 1.2)
sh_a = np.clip(blur(dilate((valid > 0.02).astype(np.float32), 4), 2), 0, 1)
pass  # bangs shadow is baked into the forehead base now

# ---------------------------------------------------------------- body layer
def spread(img, known, region, passes=(3, 8, 20, 45)):
    """Smoothly fill `region` with colours spread from `known` pixels (multi-scale normalized convolution)."""
    out = img.copy(); filled = known > 0.5; todo = (region > 0.5) & ~filled
    for s_ in passes:
        c, d = norm_blur(img, known.astype(np.float32), s_, down=1 if s_ < 6 else 4)
        need = todo & (d > 1e-3)
        out[need] = c[need]; todo &= ~need
    return out

neck_hair = blur(dilate((hair_a_full > 0.02).astype(np.float32), 10), 6) * (Y < 1900)
body_a = FIG_NA * (1 - neck_hair) + blur(skin_sil, 1.0) * neck_hair
dna0 = np.abs(M - NA).max(2)
arm_zone = blur(dilate(fill_holes(cv2.morphologyEx((smoothstep(0.07, 0.16, dna0) * (Y > 1980) > 0.5).astype(np.float32), cv2.MORPH_CLOSE,
                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))), 22), 10)
body_rgb = mix(M, NA, arm_zone)
# skin under / beside the hair: the master's own skin spread smoothly over what the hair covered (the side locks hang
# behind the neck, so this is only ever seen through stray strands or when the hair swings away)
hair_near = blur(dilate((hair_a_full > 0.03).astype(np.float32), 4), 2)
skin_known = (hair_near < 0.01) & (erode(skin_sil, 4) > 0.5) & (Y < 2000) & (Y > 1150)
under = spread(body_rgb, skin_known, neck_hair * (Y < 2000))
body_rgb = mix(body_rgb, under, neck_hair * hair_near)

# neck behind the chin: the master's jaw line sits in this layer just below the face's cut, so the band under the jaw
# and the hidden neck above it are refilled smoothly from the skin further down the neck
head_hard = (head_mask > 0.5).astype(np.float32)
jy = np.where(jaw_s > 0, jaw_s, 1440)[None, :]
NECK_Y = int(meta['jaw']['chinY']) + 40                         # first row fully below the chin
row = (skin_sil[NECK_Y] > 0.5).nonzero()[0]; row = row[(row > 1450) & (row < 2050)]
nl, nr = row.min(), row.max()
print('neck at', NECK_Y, nl, nr)
neck_cols = smoothstep(nl, nl + 5, X) * smoothstep(nr, nr - 5, X)   # crisp like the neck's own outline (seen when the head turns)
# the neck is close to a vertical cylinder: carry the skin profile from just below the chin straight up
prof = body_rgb[NECK_Y:NECK_Y + 24].mean(0, keepdims=True)
prof = cv2.GaussianBlur(prof, (0, 0), sigmaX=5, sigmaY=0.1)
neck_fill = neck_cols * (Y > 1180) * smoothstep(jy + 56, jy + 16, Y)
body_rgb = mix(body_rgb, np.broadcast_to(prof, body_rgb.shape), neck_fill)
ext = neck_cols * (Y > 1180)
head_cut = blur(dilate(head_hard, 5), 1.5)                     # a little wider, so no sliver of the jaw outline stays behind
body_a = np.maximum(body_a * (1 - head_cut), ext)              # ext unscaled: the neck stays opaque across the cut's soft edge
body_a = np.clip(body_a, 0, 1)
body_a[:1150] = 0
body_a = body_a * smoothstep(H - 4, H - 420, Y)               # legs fade out softly at the bottom of the canvas
# silhouette rim: the matte's edge pixels carry the master's light background; take their colour from just inside
inner = erode((body_a > 0.98).astype(np.float32), 3)
body_rgb = mix(body_rgb, spread(body_rgb, inner, (body_a > 0.003) & (inner < 0.5), passes=(2, 5, 12)), 1 - blur(inner, 1.0))
pw.save('body', body_rgb, body_a)

# ---------------------------------------------------------------- arms
dna = np.abs(M - NA).max(2)
arm_m = smoothstep(0.07, 0.16, dna) * (Y > 1980)
arm_m = cv2.morphologyEx(arm_m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
arm_m = fill_holes(arm_m)
arms = {}
for side, cond in [('L', X < 1750), ('R', X >= 1750)]:
    m = largest_components(arm_m * cond, 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    m = blur(m, 2.0)
    a = FIG * m
    arms[side] = a
    pw.save('arm_' + side, M, a)

# ---------------------------------------------------------------- eyes: sclera / iris / lashes
for side, e in eyes.items():
    op = e['opening']; ops = e['op_soft']
    x0, y0, x1, y1 = [int(v) for v in e['box']]
    icx, icy, ir = e['iris']
    iris_disc = ((X - icx) ** 2 + (Y - icy) ** 2 <= (ir + 1.5) ** 2).astype(np.float32)
    # sclera: inpaint iris out of the opening
    m8 = (dilate(iris_disc, 3) * 255).astype(np.uint8)
    roi = (slice(y0 - 30, y1 + 30), slice(x0 - 30, x1 + 30))
    scl = EC.copy()
    sub = (EC[roi] * 255).astype(np.uint8)
    inp = cv2.inpaint(sub, m8[roi], 12, cv2.INPAINT_TELEA)
    inp = f32(inp)
    # smooth the inpainted area & add a soft top shadow from the lid
    sm = blur(inp, 6)
    w = blur(dilate(iris_disc, 3), 4)[roi][..., None]
    scl[roi] = inp * (1 - w) + sm * w
    ytop = y0; hgt = (y1 - y0)
    shade = 1 - 0.18 * smoothstep(ytop + hgt * 0.55, ytop, Y)
    scl = scl * shade[..., None] * (dilate(iris_disc, 3)[..., None]) + scl * (1 - dilate(iris_disc, 3)[..., None])
    scl_a = blur(dilate(op, 3), 1.0)
    pw.save('eyewhite_' + side, scl, scl_a)
    # iris: visible pixels + symmetric completion of the hidden top
    vis = op * iris_disc
    vis_e = erode(vis, 2)
    # iris colour: transfer the clean pass's iris to the master's iris statistics (LAB mean/std)
    good = (vis_e > 0.5) & (np.abs(M - EC).max(2) < 0.08)
    lab_m = cv2.cvtColor(M.astype(np.float32), cv2.COLOR_BGR2LAB); lab_e = cv2.cvtColor(EC.astype(np.float32), cv2.COLOR_BGR2LAB)
    ve = vis_e > 0.5
    mu_m, sd_m = lab_m[good].mean(0), lab_m[good].std(0) + 1e-3
    mu_e, sd_e = lab_e[ve].mean(0), lab_e[ve].std(0) + 1e-3
    lab_t = (lab_e - mu_e) / sd_e * sd_m + mu_m
    ECi = np.clip(cv2.cvtColor(lab_t.astype(np.float32), cv2.COLOR_LAB2BGR), 0, 1)
    ir_rgb = ECi.copy()
    yy, xx = np.mgrid[int(icy - ir - 4):int(icy + ir + 5), int(icx - ir - 4):int(icx + ir + 5)]
    for (py, px) in zip(yy.ravel(), xx.ravel()):
        if (px - icx) ** 2 + (py - icy) ** 2 > (ir + 2.5) ** 2 or vis_e[py, px] > 0.5: continue
        # mirror across the horizontal axis through the centre, then rotate inward until visible
        found = False
        for ang_scale in (1.0, 0.85, 0.7, 0.55, 0.4):
            dx, dy = px - icx, py - icy
            r = np.hypot(dx, dy); th = np.arctan2(dy, dx)
            th2 = -th * ang_scale if dy < 0 else th
            qx, qy = icx + r * np.cos(-th2 if dy < 0 else th2), icy + r * np.sin(-th2 if dy < 0 else th2)
            qx, qy = int(round(qx)), int(round(qy))
            if vis_e[qy, qx] > 0.5:
                c = ECi[qy, qx].copy()
                c *= 0.8 if dy < 0 else 1.0              # top of iris is in lid shadow
                ir_rgb[py, px] = c; found = True; break
        if not found: ir_rgb[py, px] = ECi[int(icy), int(icx)]
    iris_a = blur(((X - icx) ** 2 + (Y - icy) ** 2 <= ir ** 2).astype(np.float32), 0.9)
    pw.save('iris_' + side, ir_rgb, iris_a)
    # lashes / lid: everything in the patch outside the opening
    lash_a = eye_patch[side] * (1 - blur(erode(op, 1), 0.8))
    pw.save('lash_' + side, EC, lash_a)
    # smile-eye patch
    pw.save('eyesmile_' + side, SME, blur(dilate((np.maximum(eye_patch[side], smoothstep(0.05, 0.14, np.abs(SME - CL).max(2)) * (eye_patch[side] > 0.02)) > 0.3).astype(np.float32), 4), 5) * (dilate(eye_patch[side], 12) > 0))
    eyes[side]['geom'] = dict(cx=float((x0 + x1) / 2), cy=float((y0 + y1) / 2), x0=float(e['box'][0]), y0=float(e['box'][1]),
                              x1=float(e['box'][2]), y1=float(e['box'][3]), irisX=icx, irisY=icy, irisR=ir,
                              poly=e['poly'].tolist())

meta['eyes'] = {s: e['geom'] for s, e in eyes.items()}

# brows / earrings
BROW_A = np.load(os.path.join(ST, 'brow_a.npy')).astype(np.float32); BROW_F = load_rgb('stage/brow_F.png')
for s in 'LR':
    pw.save('brow_' + s, BROW_F, BROW_A * ((X < 1750) if s == 'L' else (X >= 1750)))
    eF = unmix(M, NE, earrings[s])
    pw.save('earring_' + s, clamp_chroma(eF, np.ones_like(eF) * 0.6, 0.05), earrings[s])

# mouth: built and rigged by stage_mouth.py

# blush (procedural soft discs from the master's cheek colour)
pw.dump(os.path.join(ROOT, 'web', 'assets', 'parts.json'), dict(canvas=dict(w=W, h=H), meta=meta))

# ---------------------------------------------------------------- reconstruction preview
order = ['hair_back', 'ponytail', 'sideback_L', 'sideback_R', 'body', 'arm_L', 'arm_R', 'face', 'eyewhite_L', 'iris_L', 'lash_L', 'eyewhite_R', 'iris_R', 'lash_R',
         'brow_L', 'brow_R', 'side_L', 'side_R', 'earring_L', 'earring_R', 'hair_crown', 'bangs', 'tiara']
canvas = np.ones((H, W, 3), np.float32) * np.array([0.85, 0.85, 0.85])
for n in order:
    p = pw.parts[n]; im = f32(cv2.imread(os.path.join(OUTDIR, p['file']), cv2.IMREAD_UNCHANGED))
    a = im[..., 3:4]; sl = (slice(p['y'], p['y'] + p['h']), slice(p['x'], p['x'] + p['w']))
    if n == 'bangs_shadow':
        canvas[sl] = canvas[sl] * (1 - a + a * im[..., :3]); continue
    if n.startswith('iris'):
        s = n[-1]; clipm = blur(dilate(eyes[s]['opening'], 3), 1.0)[sl][..., None]; a = a * clipm
    canvas[sl] = im[..., :3] * a + canvas[sl] * (1 - a)
cv2.imwrite(os.path.join(ST, 'recon.png'), (canvas * 255).astype(np.uint8))
err = np.abs(canvas - M).max(2)
cv2.imwrite(PV + 'recon_err.jpg', cv2.resize(np.clip(err * 4, 0, 1) * 255, None, fx=0.25, fy=0.25))
cv2.imwrite(PV + 'recon_head.jpg', np.hstack([canvas[100:1750, 1150:2350], M[100:1750, 1150:2350]]) * 255)
print('mean err', err.mean(), 'p99', np.percentile(err, 99))

# ---------------------------------------------------------------- dress sheen masks (where the satin can catch light)
dress = (color_mask(SEG, (0, 255, 0), 90) + color_mask(SEG, (0, 128, 0), 70) > 0).astype(np.float32)
dress = blur(erode(dress, 6), 6)
Lm = M.mean(2)
hp = Lm - blur(Lm, 25)                                   # local highlights of the fabric
spec = np.clip(smoothstep(0.45, 0.95, Lm) * 0.6 + smoothstep(-0.02, 0.12, hp) * 0.7, 0, 1) * dress
for n in ['body', 'arm_L', 'arm_R']:
    p = pw.parts[n]
    sl = (slice(p['y'], p['y'] + p['h']), slice(p['x'], p['x'] + p['w']))
    m = (spec[sl] * 255).astype(np.uint8)
    m = cv2.resize(m, (max(1, p['w'] // 2), max(1, p['h'] // 2)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(OUTDIR, f'sheen_{n}.png'), m)
    pw.parts[n]['sheen'] = f'sheen_{n}.png'
pw.dump(os.path.join(ROOT, 'web', 'assets', 'parts.json'), dict(canvas=dict(w=W, h=H), meta=meta))
print('sheen ok')
