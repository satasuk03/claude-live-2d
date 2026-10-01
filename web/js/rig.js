// The character rig: parts, meshes, parameters and all deformers.
import { makePhysics } from './physics.js';

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const lerp = (a, b, t) => a + (b - a) * t;
const D2R = Math.PI / 180;

// ------------------------------------------------------------------ parameters
export const PARAMS = [
  // id, min, max, default, group
  ['AngleX', -30, 30, 0, 'Head'], ['AngleY', -30, 30, 0, 'Head'], ['AngleZ', -30, 30, 0, 'Head'],
  ['EyeLOpen', 0, 1.25, 1, 'Eyes'], ['EyeROpen', 0, 1.25, 1, 'Eyes'],
  ['EyeLSmile', 0, 1, 0, 'Eyes'], ['EyeRSmile', 0, 1, 0, 'Eyes'],
  ['EyeBallX', -1, 1, 0, 'Eyes'], ['EyeBallY', -1, 1, 0, 'Eyes'], ['EyeBallScale', 0.8, 1.15, 1, 'Eyes'],
  ['BrowLY', -1, 1, 0, 'Brows'], ['BrowRY', -1, 1, 0, 'Brows'],
  ['BrowLAngle', -1, 1, 0, 'Brows'], ['BrowRAngle', -1, 1, 0, 'Brows'],
  ['BrowAngry', 0, 1, 0, 'Brows'],
  ['MouthOpenY', 0, 1, 0, 'Mouth'], ['MouthForm', -1, 1, 0, 'Mouth'],
  ['MouthA', 0, 1, 1, 'Mouth'], ['MouthI', 0, 1, 0, 'Mouth'], ['MouthU', 0, 1, 0, 'Mouth'], ['MouthO', 0, 1, 0, 'Mouth'],
  ['MouthGrin', 0, 1, 0, 'Mouth'], ['MouthPout', 0, 1, 0, 'Mouth'],
  ['Cheek', 0, 1, 0, 'Face'],
  ['BodyAngleX', -10, 10, 0, 'Body'], ['BodyAngleY', -10, 10, 0, 'Body'], ['BodyAngleZ', -10, 10, 0, 'Body'],
  ['Breath', 0, 1, 0, 'Body'],
  ['ArmL', -1, 1, 0, 'Arms'], ['ArmR', -1, 1, 0, 'Arms'], ['ElbowL', -1, 1, 0, 'Arms'], ['ElbowR', -1, 1, 0, 'Arms'],
  ['HairFront', -1, 1, 0, 'Physics'], ['HairSideL', -1, 1, 0, 'Physics'], ['HairSideR', -1, 1, 0, 'Physics'],
  ['HairPony', -1, 1, 0, 'Physics'], ['EarringL', -1, 1, 0, 'Physics'], ['EarringR', -1, 1, 0, 'Physics'],
];

// ------------------------------------------------------------------ geometry constants (model px)
const HEAD = { cx: 1750, cy: 1010, R: 340, pivotX: 1750, pivotY: 1460 };
const BODY = { cx: 1750, cy: 2650, R: 720, hipX: 1750, hipY: 3950, neckY: 1600 };
const ARMS = {
  L: { sx: 1040, sy: 2110, ex: 840, ey: 3000, wx: 470, wy: 3960, dir: -1 },
  R: { sx: 2460, sy: 2110, ex: 2660, ey: 3000, wx: 3030, wy: 3960, dir: 1 },
};
const EAR_PIV = { L: [1453, 1158], R: [2022, 1154] };
const MOUTH = { cx: 1737, upperY: 1287, cornerL: [1662, 1300], cornerR: [1842, 1300] };
const BROW_C = { L: [1615, 965], R: [1885, 958] };

// per-part rig description; order = draw order
const PART_DEFS = [
  { n: 'hair_back', head: true, z: -0.55, grid: [18, 18] },
  { n: 'ponytail', head: true, z: -0.35, grid: [22, 16], strand: { key: 'pony', root: 450, len: -300, pow: 1.4, amp: 0.22 } },
  { n: 'body', body: true, grid: [26, 60], neck: true },
  { n: 'arm_L', arm: 'L', grid: [16, 44] },
  { n: 'arm_R', arm: 'R', grid: [16, 44] },
  { n: 'face', head: true, z: 0, grid: [44, 54], mouthWarp: true, faceWarp: true },
  { n: 'blush', head: true, z: 0.02, proc: 'blush', mouthWarp: false, faceWarp: true },
  { n: 'mouth_U', head: true, z: 0.03, grid: [20, 16], mouth: 'U', mouthWarp: true },
  { n: 'mouth_A', head: true, z: 0.03, grid: [20, 16], mouth: 'A', mouthWarp: true },
  { n: 'mouth_E', head: true, z: 0.03, grid: [20, 16], mouth: 'E', mouthWarp: true },
  { n: 'mouth_O', head: true, z: 0.03, grid: [20, 16], mouth: 'O', mouthWarp: true },
  { n: 'mouth_smile', head: true, z: 0.03, grid: [20, 16], mouth: 'smile', mouthWarp: true },
  { n: 'eyewhite_L', head: true, z: 0.02, grid: [16, 10], eye: 'L', role: 'white', mask: true },
  { n: 'iris_L', head: true, z: 0.03, grid: [8, 8], eye: 'L', role: 'iris', clip: 'eyewhite_L' },
  { n: 'lash_L', head: true, z: 0.03, grid: [24, 18], eye: 'L', role: 'lash' },
  { n: 'eyesmile_L', head: true, z: 0.03, grid: [12, 10], eye: 'L', role: 'smile' },
  { n: 'eyewhite_R', head: true, z: 0.02, grid: [16, 10], eye: 'R', role: 'white', mask: true },
  { n: 'iris_R', head: true, z: 0.03, grid: [8, 8], eye: 'R', role: 'iris', clip: 'eyewhite_R' },
  { n: 'lash_R', head: true, z: 0.03, grid: [24, 18], eye: 'R', role: 'lash' },
  { n: 'eyesmile_R', head: true, z: 0.03, grid: [12, 10], eye: 'R', role: 'smile' },
  { n: 'brow_L', head: true, z: 0.05, grid: [16, 8], brow: 'L' },
  { n: 'brow_R', head: true, z: 0.05, grid: [16, 8], brow: 'R' },
  { n: 'side_L', head: true, z: 0.09, grid: [12, 34], strand: { key: 'sideL', root: 960, len: 620, pow: 1.6, amp: 0.17 } },
  { n: 'side_R', head: true, z: 0.09, grid: [12, 34], strand: { key: 'sideR', root: 960, len: 620, pow: 1.6, amp: 0.17 } },
  { n: 'earring_L', head: true, z: -0.05, grid: [6, 10], earring: 'L' },
  { n: 'earring_R', head: true, z: -0.05, grid: [6, 10], earring: 'R' },
  { n: 'hair_crown', head: true, z: 0.09, grid: [30, 20] },
  { n: 'bangs', head: true, z: 0.09, grid: [30, 22], strand: { key: 'bangs', root: 690, len: 400, pow: 1.5, amp: 0.10, xfade: [1425, 2075, 110] } },
  { n: 'tiara', head: true, z: -0.1, grid: [10, 6] },
];

// ------------------------------------------------------------------ helpers
function makeGrid(w, h, cols, rows, alphaFn) {
  const vx = cols + 1, vy = rows + 1;
  const uv = new Float32Array(vx * vy * 2), rest = new Float32Array(vx * vy * 2);
  for (let j = 0; j < vy; j++) for (let i = 0; i < vx; i++) {
    const k = (j * vx + i) * 2; uv[k] = i / cols; uv[k + 1] = j / rows;
  }
  const tris = [], lines = [];
  for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
    if (alphaFn && !alphaFn(i / cols, j / rows, (i + 1) / cols, (j + 1) / rows)) continue;
    const a = j * vx + i, b = a + 1, c = a + vx, d = c + 1;
    tris.push(a, b, d, a, d, c); lines.push(a, b, b, d, d, c, c, a, a, d);
  }
  return { uv, rest, idx: new Uint32Array(tris), lines: new Uint32Array(lines), n: vx * vy };
}

function alphaOccupancy(img) {
  // downsampled alpha so we can drop empty cells
  const S = 128, c = document.createElement('canvas');
  const sw = Math.max(1, Math.min(S, img.width)), sh = Math.max(1, Math.min(S, img.height));
  c.width = sw; c.height = sh;
  const g = c.getContext('2d', { willReadFrequently: true }); g.drawImage(img, 0, 0, sw, sh);
  const d = g.getImageData(0, 0, sw, sh).data;
  return (u0, v0, u1, v1) => {
    const x0 = Math.max(0, Math.floor(u0 * sw) - 1), x1 = Math.min(sw - 1, Math.ceil(u1 * sw) + 1);
    const y0 = Math.max(0, Math.floor(v0 * sh) - 1), y1 = Math.min(sh - 1, Math.ceil(v1 * sh) + 1);
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) if (d[(y * sw + x) * 4 + 3] > 0) return true;
    return false;
  };
}

function blushCanvas() {
  const c = document.createElement('canvas'); c.width = 512; c.height = 256;
  const g = c.getContext('2d');
  for (const cx of [128, 384]) {
    const grd = g.createRadialGradient(cx, 128, 0, cx, 128, 120);
    grd.addColorStop(0, 'rgba(255,92,120,0.55)'); grd.addColorStop(0.5, 'rgba(255,110,135,0.28)'); grd.addColorStop(1, 'rgba(255,120,140,0)');
    g.fillStyle = grd; g.save(); g.translate(cx, 128); g.scale(1, 0.62); g.translate(-cx, -128);
    g.beginPath(); g.arc(cx, 128, 128, 0, Math.PI * 2); g.fill(); g.restore();
  }
  return c;
}

// ------------------------------------------------------------------ rig
export class Rig {
  constructor(renderer, partsJson, images) {
    this.r = renderer; this.meta = partsJson.meta; this.canvas = partsJson.canvas;
    this.params = {}; this.paramDefs = {};
    for (const [id, mn, mx, df, grp] of PARAMS) { this.params[id] = df; this.paramDefs[id] = { min: mn, max: mx, def: df, group: grp }; }
    this.physics = makePhysics();
    this.physicsOn = true;
    this.parts = [];
    for (const def of PART_DEFS) {
      let info, img;
      if (def.proc === 'blush') {
        img = blushCanvas();
        info = { x: 1470, y: 1100, w: 560, h: 150 };
        def.grid = [16, 6];
      } else {
        info = partsJson.parts[def.n]; img = images[def.n];
        if (!info || !img) { console.warn('missing part', def.n); continue; }
      }
      const [cols, rows] = def.grid;
      const occ = def.proc ? null : alphaOccupancy(img);
      const g = makeGrid(info.w, info.h, cols, rows, occ);
      for (let k = 0; k < g.n; k++) { g.rest[k * 2] = info.x + g.uv[k * 2] * info.w; g.rest[k * 2 + 1] = info.y + g.uv[k * 2 + 1] * info.h; }
      const mesh = this.r.mesh(g.uv, g.idx, g.lines);
      const tex = this.r.texture(img);
      const sheenTex = images['sheen:' + def.n] ? this.r.texture(images['sheen:' + def.n]) : null;
      this.parts.push({ ...def, info, mesh, tex, sheenTex, rest: g.rest, pos: new Float32Array(g.n * 2), nv: g.n, opacity: 1, visible: true });
    }
    this.byName = Object.fromEntries(this.parts.map(p => [p.n, p]));
    this._eyeCurves();
    this.wire = false;
    this.t = 0;
    this.sheenOn = true; this.sheenStrength = 1; this.sheenCol = [1.0, 0.97, 0.93];
  }

  _eyeCurves() {
    // top/bottom lid curves as functions of x from the traced opening polygon
    this.eyeGeo = {};
    for (const s of ['L', 'R']) {
      const e = this.meta.eyes[s]; const P = e.poly;
      let iL = 0, iR = 0;
      P.forEach((p, i) => { if (p[0] < P[iL][0]) iL = i; if (p[0] > P[iR][0]) iR = i; });
      const walk = (a, b) => { const out = []; let i = a; while (true) { out.push(P[i]); if (i === b) break; i = (i + 1) % P.length; } return out; };
      let A = walk(iL, iR), B = walk(iR, iL);
      const meanY = arr => arr.reduce((s, p) => s + p[1], 0) / arr.length;
      let top = meanY(A) < meanY(B) ? A : B, bot = top === A ? B : A;
      top = [...top].sort((a, b) => a[0] - b[0]); bot = [...bot].sort((a, b) => a[0] - b[0]);
      const interp = arr => x => {
        if (x <= arr[0][0]) return arr[0][1]; if (x >= arr[arr.length - 1][0]) return arr[arr.length - 1][1];
        for (let i = 1; i < arr.length; i++) if (x <= arr[i][0]) { const t = (x - arr[i - 1][0]) / (arr[i][0] - arr[i - 1][0]); return lerp(arr[i - 1][1], arr[i][1], t); }
      };
      this.eyeGeo[s] = { top: interp(top), bot: interp(bot), x0: P[iL][0], x1: P[iR][0], cx: (P[iL][0] + P[iR][0]) / 2, ...e };
    }
  }

  // ---------------------------------------------------------------- physics
  updatePhysics(dt) {
    const p = this.params, ph = this.physics;
    if (!this.physicsOn) { for (const k of ['HairFront', 'HairSideL', 'HairSideR', 'HairPony', 'EarringL', 'EarringR']) p[k] = 0; return; }
    const ax = (p.AngleX / 30) * 0.55 + (p.BodyAngleX / 10) * 0.35 + (this._bodySway || 0);
    const tilt = (p.AngleZ + p.BodyAngleZ) * D2R;
    const wind = Math.sin(this.t * 0.7) * 0.15 + Math.sin(this.t * 1.9 + 1.3) * 0.08;
    const ay = (p.AngleY / 30);
    ph.bangs.step(dt, ax * 0.8, tilt, wind * 0.3);
    ph.sideL.step(dt, ax, tilt, wind * 0.5);
    ph.sideR.step(dt, ax, tilt, wind * 0.5);
    ph.pony.step(dt, -ax * 0.9, -tilt, -wind * 0.4);
    ph.earL.step(dt, ax * 0.9, tilt, 0);
    ph.earR.step(dt, ax * 0.9, tilt, 0);
    const out = (pd, i = 0, s = 1) => clamp(pd.angles[i] * s, -1, 1);
    p.HairFront = out(ph.bangs, 1, 1.2) + ay * 0.0;
    p.HairSideL = out(ph.sideL, 1, 1.1); p.HairSideR = out(ph.sideR, 1, 1.1);
    p.HairPony = out(ph.pony, 1, 1.1);
    p.EarringL = out(ph.earL, 0, 1.6); p.EarringR = out(ph.earR, 0, 1.6);
    this._phSeg = { bangs: ph.bangs.angles, sideL: ph.sideL.angles, sideR: ph.sideR.angles, pony: ph.pony.angles };
  }

  // ---------------------------------------------------------------- deformers
  _head(x, y, z, out) {
    const p = this.params, H = HEAD;
    const ax = p.AngleX * D2R * 0.62, ay = p.AngleY * D2R * 0.55, az = p.AngleZ * D2R;
    const dx = (x - H.cx) / H.R, dy = (y - H.cy) / H.R;
    const zf = 1 / (1 + 0.85 * dx * dx + 0.55 * dy * dy) + z;      // pseudo depth of this point
    let X = x + H.R * (zf * Math.sin(ax) - dx * (1 - Math.cos(ax)));
    let Y = y - H.R * (zf * Math.sin(ay) * 0.85 + dy * (1 - Math.cos(ay)) * 0.5);
    // subtle perspective: the near side gets a hint larger
    const persp = 1 + 0.035 * Math.sin(ax) * dx * -1;
    X = H.cx + (X - H.cx) * (1 + 0.0); Y = H.cy + (Y - H.cy) * persp;
    // roll around the neck
    const c = Math.cos(az), s = Math.sin(az), ox = X - H.pivotX, oy = Y - H.pivotY;
    out[0] = H.pivotX + ox * c - oy * s; out[1] = H.pivotY + ox * s + oy * c;
  }

  _body(x, y, out) {
    const p = this.params, B = BODY;
    // head & neck ride rigidly on the top of the torso
    const ey = Math.max(y, B.neckY);
    const bx = p.BodyAngleX * D2R * 0.55, by = p.BodyAngleY * D2R * 0.5, bz = p.BodyAngleZ * D2R * 0.55;
    const dx = (x - B.cx) / B.R, dy = (ey - B.cy) / B.R;
    const zf = 1 / (1 + 1.1 * dx * dx + 0.25 * dy * dy);
    const wUpper = smooth(4250, 2900, ey);                               // legs/hips don't turn
    let X = x + B.R * (zf * Math.sin(bx) - dx * (1 - Math.cos(bx))) * wUpper;
    let Y = y - B.R * zf * Math.sin(by) * 0.35 * wUpper;
    // breathing: shoulders/chest rise a few px
    const br = p.Breath;
    const wChest = smooth(2900, 1900, ey) * smooth(1350, 1700, ey) * 0.6 + smooth(1700, 1500, ey) * 0.0;
    Y -= br * 9 * (smooth(3000, 2000, ey));
    X += (x - B.cx) * br * 0.004 * wChest;
    // sway around the hips
    const wz = smooth(4300, 3200, ey);
    const az = bz * wz, c = Math.cos(az), s = Math.sin(az), ox = X - B.hipX, oy = Y - B.hipY;
    out[0] = B.hipX + ox * c - oy * s; out[1] = B.hipY + ox * s + oy * c;
  }

  _arm(side, x, y, out) {
    const A = ARMS[side], p = this.params;
    const rot = (side === 'L' ? p.ArmL : p.ArmR) * 7 * D2R * A.dir;
    const elb = (side === 'L' ? p.ElbowL : p.ElbowR) * 10 * D2R * A.dir;
    // forearm about elbow
    const ux = A.wx - A.sx, uy = A.wy - A.sy, L2 = ux * ux + uy * uy;
    const t = ((x - A.sx) * ux + (y - A.sy) * uy) / L2;
    const te = ((A.ex - A.sx) * ux + (A.ey - A.sy) * uy) / L2;
    let X = x, Y = y;
    const wf = smooth(te - 0.06, te + 0.08, t);
    if (wf > 0) { const a = elb * wf, c = Math.cos(a), s = Math.sin(a), ox = X - A.ex, oy = Y - A.ey; X = A.ex + ox * c - oy * s; Y = A.ey + ox * s + oy * c; }
    const ws = smooth(0.0, 0.18, t);
    if (ws > 0) { const a = rot * ws, c = Math.cos(a), s = Math.sin(a), ox = X - A.sx, oy = Y - A.sy; X = A.sx + ox * c - oy * s; Y = A.sy + ox * s + oy * c; }
    // shoulders lift with breath
    Y -= p.Breath * 6;
    out[0] = X; out[1] = Y;
  }

  _strand(def, x, y, out) {
    const S = def.strand, seg = this._phSeg && this._phSeg[S.key];
    let a1 = 0, a2 = 0;
    if (seg) { a1 = seg[0]; a2 = seg[1] ?? seg[0]; }
    const up = S.len < 0, L = Math.abs(S.len);
    const d = up ? (S.root - y) : (y - S.root);
    if (d <= 0) { out[0] = x; out[1] = y; return; }
    const t = Math.min(d / L, 1.25);
    let w = Math.pow(t, S.pow);
    if (S.xfade) w *= smooth(S.xfade[0], S.xfade[0] + S.xfade[2], x) * smooth(S.xfade[1], S.xfade[1] - S.xfade[2], x);
    // blend segment angles along the strand, scaled by amplitude
    const tt = Math.min(t, 1);
    const ang = clamp(a1 * (1 - tt * 0.5) + a2 * tt * 0.5, -1, 1) * S.amp * w * (up ? -1 : 1);
    out[0] = x + d * Math.sin(ang); out[1] = y + (up ? d : -d) * (1 - Math.cos(ang));
  }

  _faceWarp(x, y, out) {
    // mouth corners up/down (MouthForm) and pout, local to the mouth
    const p = this.params;
    let X = x, Y = y;
    const form = p.MouthForm, pout = p.MouthPout;
    for (const [cx, cy, sgn] of [[...MOUTH.cornerL, -1], [...MOUTH.cornerR, 1]]) {
      const dx = (x - cx) / 46, dy = (y - cy) / 34; const w = Math.exp(-(dx * dx + dy * dy));
      Y -= form * 15 * w;
      X += sgn * form * 5 * w;
      X -= sgn * pout * 14 * w;          // corners pull in
    }
    // pout: lips push forward (slight upward/downward bulge)
    const dxm = (x - MOUTH.cx) / 70, dym = (y - 1305) / 32; const wm = Math.exp(-(dxm * dxm + dym * dym));
    Y += pout * 3 * wm * Math.sign(y - 1305);
    // cheeks rise with smile eyes
    const sm = (p.EyeLSmile + p.EyeRSmile) * 0.5;
    for (const cx of [1610, 1895]) {
      const dx = (x - cx) / 90, dy = (y - 1150) / 60; const w = Math.exp(-(dx * dx + dy * dy));
      Y -= sm * 6 * w;
    }
    out[0] = X; out[1] = Y;
  }

  _eye(def, x, y, out) {
    const p = this.params, s = def.eye, G = this.eyeGeo[s];
    const open = s === 'L' ? p.EyeLOpen : p.EyeROpen;
    const smile = s === 'L' ? p.EyeLSmile : p.EyeRSmile;
    const close = clamp(1 - open, 0, 1), wide = clamp(open - 1, 0, 0.3);
    const xx = clamp(x, G.x0, G.x1);
    const yt = G.top(xx), yb = G.bot(xx);
    const yc = lerp(yt, yb, 0.72);                              // the line the lids meet on
    const edge = smooth(G.x0 - 55, G.x0 + 6, x) * smooth(G.x1 + 55, G.x1 - 6, x);
    let Y = y;
    if (def.role !== 'iris') {
      if (y < yc) {
        const f = smooth(yt - 95, yt - 30, y) * edge;              // lid + lashes, fading out above the crease
        const target = yc - (yc - y) * 0.28;
        Y = y + (target - y) * close * f;
        Y -= wide * 40 * f * smooth(yc, yt, y);                    // wide eyes: lid lifts
      } else {
        const f = smooth(yb + 50, yb, y) * edge;
        const target = yc + (y - yc) * 0.55;
        Y = y + (target - y) * Math.max(close * 0.6, smile * 0.35) * f;
      }
    }
    let X = x;
    if (def.role === 'iris') {
      const sc = p.EyeBallScale;
      X = G.irisX + (x - G.irisX) * sc + p.EyeBallX * 17;
      Y = G.irisY + (y - G.irisY) * sc - p.EyeBallY * 9;
    }
    out[0] = X; out[1] = Y;
  }

  _brow(def, x, y, out) {
    const p = this.params, s = def.brow, [cx, cy] = BROW_C[s];
    const by = s === 'L' ? p.BrowLY : p.BrowRY, ba = s === 'L' ? p.BrowLAngle : p.BrowRAngle;
    const inner = s === 'L' ? 1 : -1;                          // inner end direction (+x for L)
    const xi = clamp((x - cx) / 140 * inner, -1.2, 1.2);       // +1 at inner end
    let Y = y - by * 24;
    Y -= ba * 13 * xi;                                         // angle: inner end up for +
    Y += p.BrowAngry * 22 * smooth(-0.6, 1, xi);               // angry: inner ends down
    Y -= p.BrowAngry * 4 * smooth(0.2, -1, xi);
    const X = x + p.BrowAngry * 10 * inner * smooth(-0.3, 1, xi);
    out[0] = X; out[1] = Y;
  }

  _mouthPatch(def, x, y, out) {
    const p = this.params;
    let Y = y;
    if (def.mouth !== 'smile' && def.mouth !== 'U') {
      const o = p.MouthOpenY;
      const sc = 0.45 + 0.55 * smooth(0, 1, o);
      Y = MOUTH.upperY + (y - MOUTH.upperY) * (y > MOUTH.upperY ? sc : 1);
    }
    let X = x;
    if (def.mouth === 'E') X = MOUTH.cx + (x - MOUTH.cx) * 0.76;   // the generated "ee" stretched too wide
    out[0] = X; out[1] = Y;
  }

  _earring(def, x, y, out) {
    const [px, py] = EAR_PIV[def.earring];
    const a = (def.earring === 'L' ? this.params.EarringL : this.params.EarringR) * 0.42;
    const c = Math.cos(a), s = Math.sin(a), ox = x - px, oy = y - py;
    out[0] = px + ox * c - oy * s; out[1] = py + ox * s + oy * c;
  }

  // ---------------------------------------------------------------- opacity logic
  _opacities() {
    const p = this.params, op = {};
    for (const s of ['L', 'R']) {
      const open = s === 'L' ? p.EyeLOpen : p.EyeROpen, sm = s === 'L' ? p.EyeLSmile : p.EyeRSmile;
      const vis = smooth(0.06, 0.32, open) * (1 - smooth(0.25, 0.85, sm));
      op['eyewhite_' + s] = vis; op['iris_' + s] = vis; op['lash_' + s] = vis;
      op['eyesmile_' + s] = smooth(0.2, 0.8, sm) * smooth(0.05, 0.4, open + 0.3);
    }
    // mouth: sequential "over" weights so the shapes blend like a crossfade
    const o = smooth(0.0, 0.38, p.MouthOpenY);
    const grin = p.MouthGrin;
    let wa = p.MouthA, wi = p.MouthI, wo = p.MouthO, wu = p.MouthU;
    const sum = wa + wi + wo + wu || 1;
    wa /= sum; wi /= sum; wo /= sum; wu /= sum;
    const W = { mouth_A: o * wa, mouth_E: o * wi, mouth_O: o * wo, mouth_U: Math.max(p.MouthPout, o * wu), mouth_smile: grin };
    // convert to sequential opacities (draw order U, A, E, O, smile)
    const order = ['mouth_U', 'mouth_A', 'mouth_E', 'mouth_O', 'mouth_smile'];
    let rem = 1;
    for (let i = order.length - 1; i >= 0; i--) {
      const w = clamp(W[order[i]], 0, 1) * rem; const k = order[i];
      op[k] = rem > 1e-4 ? clamp(W[k], 0, 1) : 0;
      rem *= 1 - clamp(W[k], 0, 1);
    }
    op.blush = p.Cheek;
    return op;
  }

  // ---------------------------------------------------------------- frame
  update(dt) {
    this.t += dt;
    this.updatePhysics(dt);
    const tmp = [0, 0], tmp2 = [0, 0];
    for (const part of this.parts) {
      const R = part.rest, P = part.pos;
      for (let k = 0; k < part.nv; k++) {
        let x = R[k * 2], y = R[k * 2 + 1];
        if (part.eye) { this._eye(part, x, y, tmp); x = tmp[0]; y = tmp[1]; }
        if (part.brow) { this._brow(part, x, y, tmp); x = tmp[0]; y = tmp[1]; }
        if (part.mouth) { this._mouthPatch(part, x, y, tmp); x = tmp[0]; y = tmp[1]; }
        if (part.faceWarp || part.mouthWarp) { this._faceWarp(x, y, tmp); x = tmp[0]; y = tmp[1]; }
        if (part.strand) { this._strand(part, x, y, tmp); x = tmp[0]; y = tmp[1]; }
        if (part.earring) { this._earring(part, x, y, tmp); x = tmp[0]; y = tmp[1]; }
        if (part.arm) { this._arm(part.arm, x, y, tmp); x = tmp[0]; y = tmp[1]; }
        if (part.head) { this._head(x, y, part.z, tmp); x = tmp[0]; y = tmp[1]; }
        if (part.neck) {
          // neck follows the head partially near the jaw
          const w = smooth(1700, 1380, y) * smooth(300, 140, Math.abs(x - 1750)) * 0.75;
          if (w > 0) { this._head(x, y, -0.15, tmp2); x = lerp(x, tmp2[0], w); y = lerp(y, tmp2[1], w); }
        }
        this._body(x, y, tmp);
        P[k * 2] = tmp[0]; P[k * 2 + 1] = tmp[1];
      }
    }
    this.op = this._opacities();
    // satin sheen: a broad band that follows the body turn + breathing, and a slow travelling glint
    const p = this.params;
    const bandPos = 3300 + p.BodyAngleX * 140 + p.AngleX * 10 + p.Breath * 60 + Math.sin(this.t * 0.35) * 220;
    const gl = (this.t * 0.11) % 1;                                   // glint sweeps every ~9s
    const glintPos = 1500 + gl * 4200;
    this._sheen = { strength: this.sheenStrength, p: [bandPos, glintPos, 700, Math.sin(gl * Math.PI) ** 2], col: this.sheenCol };
  }

  draw() {
    const r = this.r, op = this.op || {};
    for (const part of this.parts) {
      if (!part.visible) continue;
      const o = (op[part.n] ?? 1) * part.opacity;
      r.upload(part.mesh, part.pos);
      if (o <= 0.002) continue;
      if (part.mask) {
        // draw normally, and write the clip mask for the iris
        r.draw(part.mesh, part.tex, { opacity: o });
        r.draw(part.mesh, part.tex, { stencilWrite: true, maskCut: 0.5 });
        continue;
      }
      if (part.clip) {
        r.draw(part.mesh, part.tex, { opacity: o, stencilTest: true });
        r.clearStencil();
        continue;
      }
      r.draw(part.mesh, part.tex, { opacity: o, blend: part.blend || 'normal', sheen: part.sheenTex && this.sheenOn ? { tex: part.sheenTex, ...this._sheen } : null });
    }
    if (this.wire) for (const part of this.parts) if (part.visible) r.drawWire(part.mesh, [0.1, 0.8, 1.0, 0.35]);
  }

  // bounds of the model for camera framing
  get size() { return this.canvas; }
}
