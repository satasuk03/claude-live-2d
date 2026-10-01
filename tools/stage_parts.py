"""Stage 2: build every rig part from the master + aligned edit passes."""
import numpy as np, cv2, os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from layers_lib import *
from eyes_geom import EYE_L, EYE_R

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUTDIR = os.path.join(ROOT, 'web', 'assets', 'parts')
ST = os.path.join(GEN, 'stage')
PV = '/private/tmp/claude-501/-Users-satasuk-Desktop-dev-live2d/1569c189-670b-4d29-93dc-ba7e90b2bb2a/scratchpad/'
pw = PartWriter(OUTDIR)

M = load_rgb('master.jpg'); H, W = M.shape[:2]
NA = load_rgb('aligned/noarms.png'); CL = load_rgb('aligned/closed2_c.png'); SMI = load_rgb('aligned/smile.png')
SME = load_rgb('aligned/smile2_c.png'); EC = load_rgb('aligned/eyeclean_c.png')
NB = load_rgb('aligned/nobrow.png'); NE = load_rgb('aligned/noear.png')
BDc = load_rgb('stage/bald_clean.png'); BD = load_rgb('aligned/bald.png')
BDc_nb = load_rgb('stage/bald_nobrow.png')
MOUTH = {k: load_rgb(f'aligned/{f}.png') for k, f in [('A', 'mouthA'), ('O', 'mouthO'), ('E', 'mouthE'), ('U', 'mouthU')]}
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
        ('ponytail', pony_ext, {}), ('hair_crown', crown_ext, {}), ('bangs', bangs_r, {}),
        ('side_L', sideL_ext, {}), ('side_R', sideR_ext, {})]:
    a = hair_a_full * region
    pw.save(name, hairF, a)

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
pw.save('face', face_rgb, head_mask)

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
neck_hair = blur(dilate((hair_a_full > 0.02).astype(np.float32), 10), 6) * (Y < 1900)
body_a = FIG_NA * (1 - neck_hair) + FIG_BD * neck_hair
dna0 = np.abs(M - NA).max(2)
arm_zone = blur(dilate(fill_holes(cv2.morphologyEx((smoothstep(0.07, 0.16, dna0) * (Y > 1980) > 0.5).astype(np.float32), cv2.MORPH_CLOSE,
                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))), 22), 10)
body_rgb = mix(M, NA, arm_zone)
skin_free = ((color_mask(SEG, (255, 128, 192), 90) > 0) & (hair_a_full < 0.01) & (Y < 2000)).astype(np.float32)
skin_free = erode(skin_free, 3)
nr_num, nr_d = norm_blur(M, skin_free, 40, down=4); nr_den, _ = norm_blur(BD, skin_free, 40, down=4)
nratio = np.clip(nr_num / np.maximum(nr_den, 1e-3), 0.8, 1.25); nratio[nr_d < 1e-4] = 1
BDn = np.clip(BD * nratio, 0, 1)
body_rgb = mix(body_rgb, BDn, neck_hair)
# cut out the head (above jaw) but extend the neck up behind the chin
head_hard = (head_mask > 0.5).astype(np.float32)
neck_cols = (X > 1585) & (X < 1915)
ext = np.zeros((H, W), np.float32)
ext_rgb = body_rgb.copy()
for x in range(1585, 1915):
    jy = int(jaw_s[x]) if jaw_s[x] > 0 else 1440
    src_y = min(jy + 14, H - 1)
    top = max(1180, jy - 210)
    ext[top:jy + 14, x] = 1
    col = body_rgb[src_y:src_y + 6, x].mean(0)
    ramp = np.linspace(0.86, 1.0, jy + 14 - top)[:, None]   # slightly darker (shadow under chin)
    ext_rgb[top:jy + 14, x] = col[None, :] * ramp
ext = blur(ext * (X > 1600) * (X < 1900), 6) * neck_cols
body_rgb = mix(body_rgb, ext_rgb, ext * head_hard)
body_a = body_a * (1 - head_hard) + np.maximum(body_a * (1 - head_hard), ext) * head_hard
body_a = np.clip(body_a, 0, 1)
body_a[:1150] = 0
body_a = body_a * smoothstep(H - 4, H - 420, Y)               # legs fade out softly at the bottom of the canvas
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

# ---------------------------------------------------------------- mouth patches
mz = np.zeros((H, W), np.float32)
cv2.ellipse(mz, (1752, 1305), (150, 95), 0, 0, 360, 1, -1)
mz = blur(mz, 10)
for k, img in list(MOUTH.items()) + [('smile', SMI)]:
    d = smoothstep(0.05, 0.14, np.abs(img - M).max(2)) * mz
    d = fill_holes(dilate((d > 0.25).astype(np.float32), 10))
    a = blur(d, 8) * mz
    pw.save('mouth_' + k, img, a)

# blush (procedural soft discs from the master's cheek colour)
pw.dump(os.path.join(ROOT, 'web', 'assets', 'parts.json'), dict(canvas=dict(w=W, h=H), meta=meta))

# ---------------------------------------------------------------- reconstruction preview
order = ['hair_back', 'ponytail', 'body', 'arm_L', 'arm_R', 'face', 'eyewhite_L', 'iris_L', 'lash_L', 'eyewhite_R', 'iris_R', 'lash_R',
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
