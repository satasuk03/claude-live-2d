"""Stage 1: hair alpha matte using known-background matting (bald edit = background)."""
import numpy as np, cv2, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from layers_lib import *

OUT = os.path.join(GEN, 'stage'); os.makedirs(OUT, exist_ok=True)
M = load_rgb('master.jpg')
NB = load_rgb('aligned/nobrow.png')
NE = load_rgb('aligned/noear.png')
BD = load_rgb('aligned/facebase.png')   # bald, no brows, eyes closed, no earrings
HK = load_rgb('aligned/hair.png')
SEG = imread('aligned/seg.png')
H, W = M.shape[:2]

# --- clean bald background: remove its faint eyebrows by inpainting
brow_seg = color_mask(SEG, (128, 64, 0), 70)
brow_zone = dilate(brow_seg, 28) * poly_mask(M.shape, [(1450, 880), (2050, 880), (2050, 1010), (1450, 1010)])
bd8 = (BD * 255).astype(np.uint8)
BDc = f32(cv2.inpaint(bd8, (brow_zone > 0.5).astype(np.uint8) * 255, 9, cv2.INPAINT_TELEA))
BDc = BD * (1 - blur(brow_zone, 6)[..., None]) + BDc * blur(brow_zone, 6)[..., None]

# --- the "with hair" image: nobrow version (so brows don't count as hair), with no-earring patches
ear_boxes = [poly_mask(M.shape, [(1405, 1130), (1505, 1130), (1505, 1310), (1405, 1310)]),
             poly_mask(M.shape, [(1970, 1125), (2075, 1125), (2075, 1305), (1970, 1305)])]
ear_sil = color_mask(SEG, (192, 192, 192), 45) * np.maximum(*ear_boxes)
dne_ = np.abs(M - NE).max(2) * np.maximum(*ear_boxes)
ear_sil = np.maximum(ear_sil, (dne_ > 0.12).astype(np.float32) * dilate(ear_sil, 30))
earz = blur(dilate(ear_sil, 14), 4)
# colour-match the regenerated passes to the master (they drift slightly warm)
def gain_match(src, ref, m):
    g = ref[m].mean(0) / np.maximum(src[m].mean(0), 1e-4); return np.clip(src * g, 0, 1)
hair_guess = (HK[..., 1] - np.maximum(HK[..., 0], HK[..., 2]) < 0.2) & (M.mean(2) < 0.35)
hair_guess[1750:] = False
NBg = gain_match(NB, M, hair_guess); NEg = gain_match(NE, M, hair_guess)
I = M * (1 - earz[..., None]) + NEg * earz[..., None]

# --- hair zone from the green-screen pass (computed early for skin colour matching)
g = HK
greenness = g[..., 1] - np.maximum(g[..., 0], g[..., 2])
key = 1 - smoothstep(0.25, 0.6, greenness)       # 1 = hair (non-green)
key[1750:] = 0                                    # green pass only valid on head
seg_hair = color_mask(SEG, (0, 0, 255), 90) + color_mask(SEG, (255, 255, 0), 80)
zone = np.minimum(np.maximum(dilate(key, 24), dilate(seg_hair, 30)), dilate(seg_hair, 50))
head_box = poly_mask(M.shape, [(1380, 560), (2120, 560), (2120, 1150), (1380, 1150)])
zone = np.maximum(zone, dilate(key, 12) * head_box)
zone[1720:] = 0
zone = blur(zone, 10)
# exclude eyes (lashes differ between bald/master) – strands over the eyes stay baked in eye layers
eye_ex = np.zeros_like(zone)
for (cx, cy) in [(1585, 1045), (1910, 1045)]:
    cv2.ellipse(eye_ex, (cx, cy), (128, 62), 0, 0, 360, 1, -1)
eye_ex = blur(eye_ex, 6)

# --- colour-correct the bald pass to the master's skin, using only skin that a first-pass matte says is hair-free
def matte(Bk):
    diff = np.abs(I - Bk).max(2)
    core = (diff > 0.22) & (key > 0.9) & (I.mean(2) < 0.5)
    F, _ = norm_blur(I, core.astype(np.float32), 10)
    a, den = known_bg_alpha(I, Bk, F)
    return a * smoothstep(0.004, 0.02, den) * zone
a1 = matte(BDc)
face_like = color_mask(SEG, (255, 0, 0), 90) + color_mask(SEG, (255, 128, 0), 90) + color_mask(SEG, (255, 128, 192), 90)
skin_vis = ((dilate(a1, 3) < 0.06) & (face_like > 0) & (M.mean(2) > 0.45)).astype(np.float32)
skin_vis = erode(skin_vis, 2); skin_vis[:450] = 0; skin_vis[1750:] = 0
skin_vis *= (1 - eye_ex)
skin_vis[:1010] = 0                                   # forehead skin carries the bangs' shadow: leave that to the shadow layer
gm = np.median((M / np.maximum(BDc, 1e-3))[skin_vis > 0.5], axis=0)
num, d1 = norm_blur(M, skin_vis, 60, down=8); den_, _ = norm_blur(BDc, skin_vis, 60, down=8)
loc = np.clip(num / np.maximum(den_, 1e-3), 0.7, 1.35)
wl = smoothstep(0.0, 0.05, d1)[..., None]
ratio = loc * wl + gm[None, None, :] * (1 - wl)
print('global skin ratio', gm)
BDc = np.clip(BDc * ratio, 0, 1)
# forehead: bake the soft shadow the bangs cast, from hair-free forehead skin (second pass)
a2 = matte(BDc)
fh = poly_mask(M.shape, [(1430, 640), (2070, 640), (2070, 1030), (1430, 1030)])
sv2 = ((dilate(a2, 3) < 0.04) & (face_like > 0) & (fh > 0) & (M.mean(2) > 0.4)).astype(np.float32)
sv2 = erode(sv2, 2)
n2, d2 = norm_blur(M, sv2, 45, down=4); b2, _ = norm_blur(BDc, sv2, 45, down=4)
rf = np.clip(n2 / np.maximum(b2, 1e-3), 0.7, 1.15)
wf = blur(smoothstep(0.0, 0.12, d2) * blur(fh, 25), 20)[..., None]
BDc = np.clip(BDc * (rf * wf + (1 - wf)), 0, 1)
cv2.imwrite(os.path.join(OUT, 'bald_nobrow.png'), (BDc * 255).astype(np.uint8))

# --- around the eyes the true background is the clean-eye pass (same face, strands removed)
EC = load_rgb('aligned/eyeclean_c.png')
CL2 = load_rgb('aligned/closed2_c.png')
Yg = np.mgrid[0:M.shape[0], 0:M.shape[1]][0].astype(np.float32)
from eyes_geom import EYE_L, EYE_R
eye_core = np.zeros(M.shape[:2], np.float32)
for P_ in (EYE_L, EYE_R):
    op_ = poly_mask(M.shape, P_)
    up = np.zeros_like(op_); up[:-22] = op_[22:]               # opening shifted up: covers the upper lash band
    eye_core = np.maximum(eye_core, dilate(np.maximum(op_, up), 14))
excore = blur(dilate(eye_core, 3), 2)
np.save(os.path.join(OUT, 'eye_core.npy'), excore.astype(np.float16))
I = I * (1 - excore[..., None]) + M * excore[..., None]
BDc = BDc * (1 - excore[..., None]) + EC * excore[..., None]
darker = (EC.mean(2) - M.mean(2)) > 0.05
zoneE = blur(dilate(darker.astype(np.float32), 2), 1.5) * (1 - blur(erode(eye_core, 4), 3))
zone = np.maximum(zone * (1 - excore), zoneE * excore)
zone = np.maximum(zone, blur(dilate(key, 6), 3) * eye_ex * (1 - excore))   # hair around the eyes vs the clean base
ex2 = excore

# --- brows: separate layer (master vs no-brow pass); the matting background gets the brows composited in
Yb = np.mgrid[0:M.shape[0], 0:M.shape[1]][0].astype(np.float32)
Xb = np.mgrid[0:M.shape[0], 0:M.shape[1]][1].astype(np.float32)
bzone = dilate(brow_seg, 12) * smoothstep(890, 905, Yb) * smoothstep(1000, 985, Yb)
bzone = blur(bzone * (1 - dilate(eye_core, 10)), 4)
Lnb = NBg.mean(2); Lm_ = M.mean(2)
dark = np.clip(Lnb - Lm_, 0, 1)                                   # brows only ever darken the skin/hair under them
core_b = (dark > 0.12) & (bzone > 0.5)
bcol = np.median(M[core_b], axis=0) if core_b.sum() > 50 else np.array([0.32, 0.36, 0.45])
denom = np.maximum(Lnb - bcol.mean(), 0.08)
brow_a = blur(np.clip(dark / denom, 0, 1) * bzone, 0.7)
brow_a = np.where(brow_a > 0.06, brow_a, 0).astype(np.float32)
brow_F = np.ones_like(M) * bcol
# keep a little of the original texture (hair-like brow strokes) in the colour
brow_F = np.clip(brow_F + np.minimum(M - blur(M, 3), 0) * 0.8, 0, 1)   # keep only darker strokes
np.save(os.path.join(OUT, 'brow_a.npy'), brow_a.astype(np.float16))
cv2.imwrite(os.path.join(OUT, 'brow_F.png'), (brow_F * 255).astype(np.uint8))
# hair over the brows: matte it from the no-brow pass so strands stay in the hair layer
bz_soft = blur(dilate((brow_a > 0.02).astype(np.float32), 10), 6)
I = I * (1 - bz_soft[..., None]) + NBg * bz_soft[..., None]

# --- known-background alpha
diff = np.abs(I - BDc).max(2)
core = ((diff > 0.22) & (key > 0.9) & (I.mean(2) < 0.5)) | ((diff > 0.15) & (ex2 > 0.5) & (I.mean(2) < 0.45))
F, dens = norm_blur(I, core.astype(np.float32), 10)
a, den = known_bg_alpha(I, BDc, F)
reliable = smoothstep(0.004, 0.02, den)
a = a * reliable
a = np.clip((a - 0.035) / (0.93 - 0.035), 0, 1) * zone
Lm = I.mean(2)
FIGm = load_mask('mattes/fig.png')
bgw = (FIGm < 0.03).astype(np.float32)
bg_est, _ = norm_blur(M, bgw, 30, down=4)
bg_like = smoothstep(0.07, 0.035, np.abs(M - bg_est).max(2))
a = a * (1 - bg_like * smoothstep(0.6, 0.1, FIGm))
# tiara: opaque where it differs from bald
tiara_zone = dilate(color_mask(SEG, (255, 255, 0), 80), 18) * poly_mask(M.shape, [(1560, 240), (1930, 240), (1930, 430), (1560, 430)])
tiara_box = poly_mask(M.shape, [(1592, 262), (1900, 262), (1905, 415), (1820, 418), (1750, 428), (1680, 420), (1590, 418)], 2)
tiara_a = smoothstep(0.30, 0.50, M.mean(2)) * tiara_box
np.save(os.path.join(OUT, 'hair_a.npy'), a.astype(np.float16))
np.save(os.path.join(OUT, 'eye_ex.npy'), eye_ex.astype(np.float16))
np.save(os.path.join(OUT, 'earz.npy'), earz.astype(np.float16))
np.save(os.path.join(OUT, 'tiara_a.npy'), tiara_a.astype(np.float16))
cv2.imwrite(os.path.join(OUT, 'bald_clean.png'), (BDc * 255).astype(np.uint8))
cv2.imwrite(os.path.join(OUT, 'hair_src.png'), (I * 255).astype(np.uint8))
Fh = unmix(I, BDc, a)
# stabilise colours of faint pixels: fall back to the local opaque hair colour
Floc, _ = norm_blur(I, (a > 0.85).astype(np.float32), 12)
wF = smoothstep(0.015, 0.12, a)[..., None]
Fh = Fh * wF + Floc * (1 - wF)
Fh = clamp_chroma(Fh, Floc, 0.035)
# unmixing against a dark background (brows) can invent bright hair: cap luminance near the local hair colour
Lf = Fh.mean(2, keepdims=True); Lcap = np.maximum(Floc.mean(2, keepdims=True) + 0.28, M.mean(2, keepdims=True))
Fh = np.clip(Fh * np.minimum(1, Lcap / np.maximum(Lf, 1e-3)), 0, 1)
cv2.imwrite(os.path.join(OUT, 'hair_F.png'), (Fh * 255).astype(np.uint8))

# previews
crop = (slice(100, 1750), slice(1150, 2350))
pv = np.dstack([a] * 3)[crop]
checker = ((np.indices(pv.shape[:2]).sum(0) // 40) % 2)[..., None] * 0.25 + 0.6
comp = Fh[crop] * a[crop][..., None] + checker * (1 - a[crop][..., None])
comp2 = Fh[crop] * a[crop][..., None] + BDc[crop] * (1 - a[crop][..., None])
cv2.imwrite('/private/tmp/claude-501/-Users-satasuk-Desktop-dev-live2d/1569c189-670b-4d29-93dc-ba7e90b2bb2a/scratchpad/hair_prev.jpg',
            cv2.resize(np.hstack([pv, comp, comp2]) * 255, None, fx=0.5, fy=0.5))
print('ok')
