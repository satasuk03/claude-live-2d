// Idle motion, blinking, pointer tracking, expressions and lip-sync -> rig parameters.

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (a, b, t) => a + (b - a) * t;

// smooth 1D value noise
function noise1(seed) {
  const h = n => { const s = Math.sin(n * 127.1 + seed * 311.7) * 43758.5453; return s - Math.floor(s); };
  return t => { const i = Math.floor(t), f = t - i, u = f * f * (3 - 2 * f); return lerp(h(i), h(i + 1), u) * 2 - 1; };
}

export const EXPRESSIONS = {
  neutral: {},
  smile: { MouthForm: 0.75, EyeLSmile: 0.3, EyeRSmile: 0.3, Cheek: 0.25, BrowLY: 0.15, BrowRY: 0.15 },
  happy: { EyeLSmile: 1, EyeRSmile: 1, MouthGrin: 1, MouthForm: 0.6, Cheek: 0.45, BrowLY: 0.35, BrowRY: 0.35 },
  surprised: { EyeLOpen: 1.22, EyeROpen: 1.22, BrowLY: 1, BrowRY: 1, MouthOpenY: 0.75, MouthO: 1, MouthA: 0, EyeBallScale: 0.9 },
  wink: { EyeLOpen: 0, EyeRSmile: 0.25, MouthForm: 0.7, Cheek: 0.2, BrowLY: -0.25, BrowRY: 0.3, AngleZ: 5 },
  pout: { MouthPout: 1, Cheek: 0.55, BrowLAngle: 0.35, BrowRAngle: 0.35, BrowLY: 0.1, BrowRY: 0.1, AngleZ: -4 },
  angry: { BrowAngry: 1, EyeLOpen: 0.78, EyeROpen: 0.78, MouthForm: -0.8, BrowLY: -0.45, BrowRY: -0.45, EyeBallScale: 0.94, AngleY: -5, MouthPout: 0.25 },
  shy: { Cheek: 1, EyeBallY: -0.55, EyeBallX: -0.35, AngleY: -9, AngleX: -7, AngleZ: 6, EyeLOpen: 0.78, EyeROpen: 0.78, MouthForm: 0.35, BrowLAngle: 0.4, BrowRAngle: 0.4 },
};

// critically damped spring towards a target
class Spring { constructor(v = 0, k = 9) { this.v = v; this.vel = 0; this.k = k; }
  step(target, dt, k = this.k) { const w = k, x = this.v - target; const a = -w * w * x - 2 * w * this.vel; this.vel += a * dt; this.v += this.vel * dt; return this.v; } }

export class Controller {
  constructor(rig) {
    this.rig = rig; this.t = 0;
    this.pointer = { x: 0, y: 0, active: false, last: -10 };
    this.follow = true; this.autoBlink = true; this.breath = true; this.idle = true;
    this.expr = 'neutral'; this.exprW = {}; this.exprT = {};
    for (const k of Object.keys(EXPRESSIONS)) this.exprW[k] = k === 'neutral' ? 1 : 0;
    this.overrides = {};             // params pinned by the UI sliders
    this.springs = {};
    this.blink = { next: 1.5, phase: -1, double: false };
    this.nx = noise1(1.3); this.ny = noise1(7.1); this.nz = noise1(3.7); this.nb = noise1(9.9); this.ne = noise1(5.5);
    this.saccade = { x: 0, y: 0, next: 0.5 };
    this.mouth = { open: 0, a: 1, i: 0, u: 0, o: 0, src: null };
    this.face = null;                 // webcam tracking result
    this.talk = null;                 // procedural speech
    this.reaction = null;
    this.gestures = [];               // active timed gestures (presets.js), summed as parameter offsets
    this.preset = null;               // PresetPlayer while a preset is playing
  }
  sp(name, k) { return this.springs[name] || (this.springs[name] = new Spring(this.rig.params[name] ?? 0, k)); }

  setExpression(name, hold = 0) {
    this.expr = name; this.exprUntil = hold ? this.t + hold : 0;
  }

  // reaction when the user pokes the character
  poke(region) {
    const map = { head: 'happy', face: 'shy', body: 'surprised', hair: 'smile' };
    const e = map[region] || 'smile';
    this.prevExpr = this.expr === e ? 'neutral' : this.expr;
    this.setExpression(e, 2.4);
    this.reaction = { t: 0, region };
  }

  // g = { dur, f(u) -> {Param: offset} } from GESTURES
  gesture(g) { this.gestures.push({ ...g, t0: this.t }); }

  speak(durationSec) { this.talk = { t: 0, dur: durationSec, seed: Math.random() * 100 }; }

  update(dt) {
    this.t += dt; const t = this.t, P = this.rig.params;
    this.preset?.tick();
    // ---- gesture offsets
    const G = {};
    this.gestures = this.gestures.filter(g => {
      const u = (t - g.t0) / g.dur; if (u >= 1) return false;
      for (const [k, v] of Object.entries(g.f(Math.max(0, u)))) G[k] = (G[k] || 0) + v;
      return true;
    });
    const g = k => G[k] || 0;
    if (this.exprUntil && t > this.exprUntil) { this.exprUntil = 0; this.expr = this.prevExpr || 'neutral'; }
    // ---- expression weights (smooth crossfade)
    for (const k of Object.keys(EXPRESSIONS)) {
      const target = k === this.expr ? 1 : 0;
      this.exprW[k] = lerp(this.exprW[k], target, 1 - Math.exp(-dt * 7));
    }
    const ex = {};
    for (const [k, w] of Object.entries(this.exprW)) if (w > 1e-3) for (const [pk, pv] of Object.entries(EXPRESSIONS[k])) ex[pk] = (ex[pk] || 0) + pv * w;
    const exw = k => this.exprW[k] || 0;
    const exprOf = (name, base) => {
      // blend: expression value replaces the base value proportionally to the weights that define it
      let wsum = 0, v = 0;
      for (const [k, w] of Object.entries(this.exprW)) if (name in EXPRESSIONS[k]) { wsum += w; v += EXPRESSIONS[k][name] * w; }
      return wsum > 0 ? lerp(base, v / wsum, Math.min(1, wsum)) : base;
    };

    // ---- head target: pointer, webcam or idle drift
    let hx = 0, hy = 0, hz = 0, ebx = 0, eby = 0;
    const idleAmt = this.idle ? 1 : 0;
    const driftX = this.nx(t * 0.23) * 7 * idleAmt, driftY = this.ny(t * 0.19) * 4 * idleAmt, driftZ = this.nz(t * 0.17) * 4 * idleAmt;
    if (this.face) {
      hx = this.face.yaw; hy = this.face.pitch; hz = this.face.roll; ebx = this.face.eyeX; eby = this.face.eyeY;
    } else if (this.follow && this.pointer.active) {
      hx = clamp(this.pointer.x * 32, -30, 30); hy = clamp(-this.pointer.y * 26, -30, 30); hz = clamp(-this.pointer.x * this.pointer.y * 8, -10, 10);
      ebx = clamp(this.pointer.x * 1.4, -1, 1); eby = clamp(-this.pointer.y * 1.4, -1, 1);
      hx += driftX * 0.25; hy += driftY * 0.25; hz += driftZ * 0.5;
    } else {
      hx = driftX; hy = driftY; hz = driftZ;
      // eye saccades
      if (t > this.saccade.next) {
        this.saccade.x = (Math.random() * 2 - 1) * 0.45; this.saccade.y = (Math.random() * 2 - 1) * 0.25;
        this.saccade.next = t + 0.6 + Math.random() * 2.6;
      }
      ebx = this.saccade.x + hx / 40; eby = this.saccade.y + hy / 40;
    }
    // reaction bob
    let bob = 0;
    if (this.reaction) { this.reaction.t += dt; const rt = this.reaction.t; bob = Math.sin(rt * 9) * Math.exp(-rt * 2.5); if (rt > 3) this.reaction = null; }

    const set = (name, v) => { P[name] = name in this.overrides ? this.overrides[name] : v; };
    const k = this.face ? 18 : 6.5;
    const kg = this.gestures.length ? Math.max(k, 11) : k;          // gestures need a snappier head
    set('AngleX', this.sp('AngleX', k).step(exprOf('AngleX', hx) + g('AngleX'), dt, kg));
    set('AngleY', this.sp('AngleY', k).step(exprOf('AngleY', hy) + bob * 6 + g('AngleY'), dt, kg));
    set('AngleZ', this.sp('AngleZ', k).step(exprOf('AngleZ', hz) + bob * 3 + g('AngleZ'), dt, kg));
    set('EyeBallX', this.sp('EyeBallX', 16).step(clamp(exprOf('EyeBallX', ebx) + g('EyeBallX'), -1, 1), dt));
    set('EyeBallY', this.sp('EyeBallY', 16).step(clamp(exprOf('EyeBallY', eby) + g('EyeBallY'), -1, 1), dt));
    set('EyeBallScale', this.sp('EyeBallScale', 8).step(exprOf('EyeBallScale', 1), dt));
    // body follows the head lazily
    this._bx = this.sp('BodyAngleX', 3).step(P.AngleX * 0.28 + this.nb(t * 0.11) * 2.2 * idleAmt + g('BodyAngleX'), dt);
    set('BodyAngleX', this._bx);
    set('BodyAngleY', this.sp('BodyAngleY', 3).step(P.AngleY * 0.15, dt));
    set('BodyAngleZ', this.sp('BodyAngleZ', 2.5).step(P.AngleZ * 0.18 + this.nz(t * 0.09 + 4) * 1.6 * idleAmt + g('BodyAngleZ'), dt));
    // jump / hop translation and squash are already shaped by the gesture curves: no spring
    set('BodyY', g('BodyY'));
    set('BodySquash', clamp(g('BodySquash'), -1, 1));
    set('Breath', this.breath ? (Math.sin(t * Math.PI * 2 / 3.8) * 0.5 + 0.5) : 0);
    set('ArmL', this.sp('ArmL', 2.5).step(clamp(-P.BodyAngleZ * 0.08 + this.ne(t * 0.13) * 0.25 * idleAmt + P.Breath * 0.1 + g('ArmL'), -1, 1), dt, G.ArmL ? 9 : 2.5));
    set('ArmR', this.sp('ArmR', 2.5).step(clamp(P.BodyAngleZ * 0.08 + this.ne(t * 0.13 + 9) * 0.25 * idleAmt + P.Breath * 0.1 + g('ArmR'), -1, 1), dt, G.ArmR ? 9 : 2.5));
    set('ElbowL', this.sp('ElbowL', 2).step(this.ne(t * 0.1 + 3) * 0.3 * idleAmt, dt));
    set('ElbowR', this.sp('ElbowR', 2).step(this.ne(t * 0.1 + 6) * 0.3 * idleAmt, dt));

    // ---- blinking
    let blink = 1;
    if (this.autoBlink && !this.face) {
      const b = this.blink;
      if (b.phase < 0 && t > b.next) { b.phase = 0; }
      if (b.phase >= 0) {
        b.phase += dt; const ph = b.phase;
        const cT = 0.065, hT = 0.04, oT = 0.13;
        if (ph < cT) blink = 1 - ph / cT; else if (ph < cT + hT) blink = 0; else if (ph < cT + hT + oT) blink = (ph - cT - hT) / oT; else {
          blink = 1; b.phase = -1;
          if (!b.double && Math.random() < 0.18) { b.double = true; b.next = t + 0.12; } else { b.double = false; b.next = t + 1.8 + Math.random() * 4.2; }
        }
        blink = blink * blink * (3 - 2 * blink);
      }
    }
    let eyeL = exprOf('EyeLOpen', 1), eyeR = exprOf('EyeROpen', 1);
    if (this.face) { eyeL = this.face.eyeL; eyeR = this.face.eyeR; }
    set('EyeLOpen', Math.min(eyeL, eyeL * blink + (1 - blink) * 0));
    set('EyeROpen', Math.min(eyeR, eyeR * blink));
    set('EyeLSmile', this.sp('EyeLSmile', 10).step(exprOf('EyeLSmile', this.face ? this.face.smile * 0.4 : 0), dt));
    set('EyeRSmile', this.sp('EyeRSmile', 10).step(exprOf('EyeRSmile', this.face ? this.face.smile * 0.4 : 0), dt));

    // ---- brows
    const fb = this.face ? this.face.brow : 0;
    for (const n of ['BrowLY', 'BrowRY']) set(n, this.sp(n, 10).step(exprOf(n, fb), dt));
    for (const n of ['BrowLAngle', 'BrowRAngle', 'BrowAngry']) set(n, this.sp(n, 10).step(exprOf(n, 0), dt));

    // ---- mouth: lipsync source (mic / talk / webcam) or expression
    let mo = 0, ma = 1, mi = 0, mu = 0, moo = 0, form = 0;
    if (this.face) { mo = this.face.jaw; form = this.face.smile * 0.9 - this.face.frown; mu = this.face.pucker; moo = this.face.funnel; ma = 1; }
    if (this.mouth.src) { mo = Math.max(mo, this.mouth.open); ma = this.mouth.a; mi = this.mouth.i; mu = this.mouth.u; moo = this.mouth.o; }
    if (this.talk) {
      this.talk.t += dt; const tt = this.talk.t;
      if (tt > this.talk.dur) this.talk = null; else {
        const syl = Math.abs(Math.sin(tt * 11.5 + Math.sin(tt * 3.1) * 2)) * (0.55 + 0.45 * Math.abs(Math.sin(tt * 1.7 + this.talk.seed)));
        mo = Math.max(mo, syl * 0.85);
        const v = (Math.floor(tt * 6.5 + this.talk.seed) % 5);
        ma = v === 0 || v === 3 ? 1 : 0.2; mi = v === 1 ? 1 : 0; mu = v === 2 ? 0.6 : 0; moo = v === 4 ? 1 : 0;
      }
    }
    const talking = mo > 0.02;
    set('MouthOpenY', this.sp('MouthOpenY', 22).step(talking ? mo : exprOf('MouthOpenY', 0), dt));
    const vs = (n, v) => set(n, this.sp(n, 14).step(v, dt));
    if (talking) { vs('MouthA', ma); vs('MouthI', mi); vs('MouthU', mu); vs('MouthO', moo); }
    else { vs('MouthA', exprOf('MouthA', 1)); vs('MouthI', exprOf('MouthI', 0)); vs('MouthU', exprOf('MouthU', 0)); vs('MouthO', exprOf('MouthO', 0)); }
    set('MouthForm', this.sp('MouthForm', 9).step(exprOf('MouthForm', form), dt));
    set('MouthGrin', this.sp('MouthGrin', 9).step(talking ? 0 : exprOf('MouthGrin', 0), dt));
    set('MouthPout', this.sp('MouthPout', 9).step(talking ? 0 : exprOf('MouthPout', this.face ? this.face.pucker * 0.6 : 0), dt));
    set('Cheek', this.sp('Cheek', 4).step(exprOf('Cheek', 0), dt));

    // pinned overrides win for everything
    for (const [kk, v] of Object.entries(this.overrides)) P[kk] = v;
  }
}

// ------------------------------------------------------------------ audio lip-sync
// Analyses whatever is connected to `an` (mic or a voice clip): RMS -> mouth open, formant bands -> vowel.
export class AudioMouth {
  constructor(ctrl, ac, src = 'voice') {
    this.ctrl = ctrl; this.ac = ac; this.src = src; this.on = false;
    this.an = ac.createAnalyser(); this.an.fftSize = 2048; this.an.smoothingTimeConstant = 0.5;
    this.buf = new Float32Array(this.an.fftSize); this.spec = new Float32Array(this.an.frequencyBinCount);
    this.floor = 0.004;
  }
  start() {
    if (this.on) { this.ctrl.mouth.src = this.src; return; }
    this.on = true; this.ctrl.mouth.src = this.src;
    const loop = () => { if (!this.on) return; this._tick(); requestAnimationFrame(loop); };
    loop();
  }
  stop() { this.on = false; if (this.ctrl.mouth.src === this.src) { this.ctrl.mouth.src = null; this.ctrl.mouth.open = 0; } }
  _tick() {
    this.an.getFloatTimeDomainData(this.buf);
    let rms = 0; for (const v of this.buf) rms += v * v; rms = Math.sqrt(rms / this.buf.length);
    this.floor = Math.min(this.floor * 1.0005 + 1e-6, Math.max(rms, 0.002));
    const level = clamp((rms - this.floor * 1.5) * 14, 0, 1);
    this.an.getFloatFrequencyData(this.spec);
    const hz = this.ac.sampleRate / this.an.fftSize;
    const band = (a, b) => { let s = 0, n = 0; for (let i = Math.floor(a / hz); i < Math.ceil(b / hz); i++) { s += Math.pow(10, this.spec[i] / 20); n++; } return s / Math.max(n, 1); };
    const lo = band(250, 700), mid = band(700, 1600), hi = band(1800, 3800), tot = lo + mid + hi + 1e-9;
    const m = this.ctrl.mouth;
    m.open = lerp(m.open, Math.pow(level, 0.8), 0.6);
    const rl = lo / tot, rm = mid / tot, rh = hi / tot;
    m.a = clamp(rm * 2.2, 0, 1); m.i = clamp((rh - 0.22) * 4, 0, 1); m.o = clamp((rl - 0.55) * 3, 0, 1); m.u = clamp((rl - 0.7) * 3, 0, 1) * 0.6;
  }
}

// ------------------------------------------------------------------ microphone lip-sync
export class MicLipSync {
  constructor(ctrl) { this.ctrl = ctrl; this.on = false; }
  async start() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
    this.stream = stream;
    this.ac = new (window.AudioContext || window.webkitAudioContext)();
    this.mouth = new AudioMouth(this.ctrl, this.ac, 'mic');
    this.ac.createMediaStreamSource(stream).connect(this.mouth.an);
    this.on = true; this.mouth.start();
  }
  stop() { this.on = false; this.mouth?.stop(); this.stream?.getTracks().forEach(t => t.stop()); this.ac?.close(); }
}
