import cv2, numpy as np, sys
def load(p, s=0.25):
    im = cv2.imread(p); return im, cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None, fx=s, fy=s).astype(np.float32)/255
def ecc(ref, mov, mask=None, mode=cv2.MOTION_AFFINE):
    w = np.eye(2,3,dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 300, 1e-6)
    cc, w = cv2.findTransformECC(ref, mov, w, mode, crit, mask, 5)
    return cc, w
if __name__ == '__main__':
    M, m = load('master.jpg')
    for name in sys.argv[1:]:
        E, e = load(f'edits/{name}.jpg')
        cc, w = ecc(m, e)
        print(name, 'cc=%.4f'%cc, 'scale~%.4f %.4f'%(w[0,0], w[1,1]), 'shift(px fullres)=%.1f %.1f'%(w[0,2]*4, w[1,2]*4), 'shear %.4f %.4f'%(w[0,1],w[1,0]))
