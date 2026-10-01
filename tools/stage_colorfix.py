"""Colour-match the clean-eye family of passes back to the master."""
import numpy as np, cv2, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from layers_lib import *
M = load_rgb('master.jpg'); EC = load_rgb('aligned/eyeclean.png')
SEG = imread('aligned/seg.png'); H, W = M.shape[:2]
skin = (color_mask(SEG, (255, 0, 0), 90) + color_mask(SEG, (255, 128, 192), 90) > 0).astype(np.float32)
hairish = color_mask(SEG, (0, 0, 255), 90)
eyes = np.zeros((H, W), np.float32)
for (cx, cy) in [(1590, 1045), (1882, 1045)]: cv2.ellipse(eyes, (cx, cy), (150, 80), 0, 0, 360, 1, -1)
w = erode(skin, 4) * (1 - dilate(eyes, 10)) * (1 - dilate(hairish, 10))
w *= (np.abs(M - EC).max(2) < 0.12)
# hair too: helps the strands near the eyes
wh = erode(hairish, 6) * (1 - dilate(eyes, 10))
wt = np.clip(w + wh, 0, 1)
num, d = norm_blur(M, wt, 45, down=4); den, _ = norm_blur(EC, wt, 45, down=4)
ratio = np.clip(num / np.maximum(den, 1e-3), 0.7, 1.4)
for n in ['eyeclean', 'closed2', 'smile2']:
    X = load_rgb(f'aligned/{n}.png')
    cv2.imwrite(os.path.join(GEN, f'aligned/{n}_c.png'), (np.clip(X * ratio, 0, 1) * 255 + 0.5).astype(np.uint8))
E2 = np.clip(EC * ratio, 0, 1)
cv2.imwrite('/private/tmp/claude-501/-Users-satasuk-Desktop-dev-live2d/1569c189-670b-4d29-93dc-ba7e90b2bb2a/scratchpad/ecc.jpg',
            (np.hstack([M[930:1130, 1430:2060], E2[930:1130, 1430:2060]]) * 255).astype(np.uint8))
print('ok', np.abs(M - E2)[w > 0].mean(), np.abs(M - EC)[w > 0].mean())
