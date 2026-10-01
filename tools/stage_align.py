"""Stage 0: register every edit pass to the master (ECC affine) and compute figure mattes.
Recreates gen/aligned/ and gen/mattes/ from gen/master.jpg + gen/edits/*.jpg."""
import cv2, numpy as np, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from align import ecc
G = os.path.join(os.path.dirname(__file__), '..', 'gen')
os.makedirs(os.path.join(G, 'aligned'), exist_ok=True); os.makedirs(os.path.join(G, 'mattes'), exist_ok=True)

def small(im): return cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None, fx=.25, fy=.25).astype(np.float32) / 255

def register(src, ref, out):
    E = cv2.imread(os.path.join(G, src)); R = cv2.imread(os.path.join(G, ref))
    cc, w = ecc(small(R), small(E)); w[:, 2] *= 4
    A = cv2.warpAffine(E, w, (E.shape[1], E.shape[0]), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(os.path.join(G, 'aligned', out), A); print(f'{out:16s} cc={cc:.4f}')

for n in ['noarms', 'closed', 'mouthA', 'smile', 'hair', 'noear', 'nobrow', 'mouthO', 'mouthE', 'mouthU', 'eyeclean', 'closed2', 'smile2']:
    register(f'edits/{n}.jpg', 'master.jpg', f'{n}.png')
register('edits/bald_noear.jpg', 'master.jpg', 'bald.png')            # bald + no earrings is the "bald" pass
register('edits/facebase.jpg', 'aligned/bald.png', 'facebase.png')    # bald, no brows, eyes closed
cv2.imwrite(os.path.join(G, 'aligned', 'seg.png'), cv2.imread(os.path.join(G, 'edits/seg.jpg')))  # flat colours: no ECC

from rembg import new_session, remove
from PIL import Image
s = new_session('birefnet-general', providers=['CPUExecutionProvider'])
for name, src in [('fig', 'master.jpg'), ('fig_noarms', 'aligned/noarms.png'), ('fig_bald', 'aligned/bald.png')]:
    im = Image.open(os.path.join(G, src)).convert('RGB')
    m = remove(im.resize((im.width // 2, im.height // 2)), session=s, only_mask=True)
    m.resize(im.size, Image.BICUBIC).save(os.path.join(G, 'mattes', name + '.png')); print('matte', name)
