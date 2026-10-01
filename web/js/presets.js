// Animation presets: a generated voice clip (tools/tts.py) + a timeline of expressions and gestures.
// Gestures return additive parameter offsets for a normalised time u in [0, 1]; the Controller sums them.
import { AudioMouth } from './motion.js';

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const ss = x => { const t = clamp(x, 0, 1); return t * t * (3 - 2 * t); };
const env = (u, a = 0.2, r = 0.25) => ss(u / a) * ss((1 - u) / r);     // attack / hold / release
const hump = u => Math.sin(Math.PI * clamp(u, 0, 1));

// ------------------------------------------------------------------ gestures
// each: (args) => { dur, f(u) -> {Param: offset} }.  BodyY is in jump units (1 = JUMP_PX in the rig).
export const GESTURES = {
  nod: ({ amp = 9, n = 1, dur = 0.5 * n }) => ({ dur, f: u => ({ AngleY: -amp * Math.abs(Math.sin(Math.PI * n * u)) }) }),
  shake: ({ amp = 12, n = 2, dur = 0.32 * n }) => ({ dur, f: u => ({ AngleX: amp * Math.sin(Math.PI * 2 * n * u) * (1 - u) }) }),
  tilt: ({ deg = 10, dur = 1.2 }) => ({ dur, f: u => ({ AngleZ: deg * env(u) }) }),
  look: ({ x = 0, y = 0, eyeX = x / 25, eyeY = y / 25, dur = 1.2, a = 0.2, r = 0.25 }) =>
    ({ dur, f: u => { const e = env(u, a, r); return { AngleX: x * e, AngleY: y * e, EyeBallX: eyeX * e, EyeBallY: eyeY * e }; } }),
  lean: ({ x = 0, z = 0, dur = 1.2 }) => ({ dur, f: u => { const e = env(u); return { BodyAngleX: x * e, BodyAngleZ: z * e }; } }),
  sway: ({ amp = 4, n = 2, dur = 1.6 }) => ({ dur, f: u => { const e = env(u, 0.15, 0.3), s = Math.sin(Math.PI * 2 * n * u);
    return { BodyAngleZ: amp * s * e, AngleZ: amp * 1.2 * s * e, AngleX: amp * 0.6 * s * e }; } }),
  arms: ({ amt = -1, dur = 1 }) => ({ dur, f: u => { const e = env(u); return { ArmL: amt * e, ArmR: amt * e }; } }),
  // recoil of surprise: head snaps back, a tiny hop
  jolt: ({ amp = 1, dur = 0.55 }) => ({ dur, f: u => {
    const k = Math.exp(-u * 5) * ss(u / 0.06);
    return { AngleY: 9 * amp * k, BodyY: 0.14 * amp * hump(u / 0.45), BodySquash: -0.35 * amp * k, ArmL: -0.8 * k, ArmR: -0.8 * k };
  } }),
  // small happy hops
  bounce: ({ amp = 0.12, n = 3, dur = 0.34 * n }) => ({ dur, f: u => {
    const ph = (u * n) % 1, y = amp * 4 * ph * (1 - ph);
    return { BodyY: y, BodySquash: (ph < 0.12 || ph > 0.88 ? 0.35 : -0.12) * amp * 4, AngleY: y * 20 };
  } }),
  // a full jump: anticipation squat -> ballistic flight -> landing squash -> settle
  jump: ({ h = 1, air = 0.62, squat = 0.22, land = 0.45 }) => ({ dur: squat + air + land, f: u => {
    const t = u * (squat + air + land);
    if (t < squat) {
      const s = ss(t / squat);
      return { BodyY: -0.07 * s, BodySquash: 0.55 * s, AngleY: -5 * s, ArmL: 0.4 * s, ArmR: 0.4 * s };
    }
    if (t < squat + air) {
      const v = (t - squat) / air, y = h * 4 * v * (1 - v);
      return { BodyY: y, BodySquash: -0.5 * Math.pow(Math.abs(2 * v - 1), 2), AngleY: 7 * (1 - 2 * v), ArmL: -1, ArmR: -1 };
    }
    const w = (t - squat - air) / land, k = Math.exp(-w * 4);
    return { BodyY: -0.06 * hump(w * 1.6), BodySquash: 0.7 * k * Math.cos(w * 9), AngleY: -6 * k, ArmL: -1 + 1.3 * ss(w), ArmR: -1 + 1.3 * ss(w) };
  } }),
};

// ------------------------------------------------------------------ presets
// cue: [time, 'expr', name] | [time, gesture, args]. Times are seconds into the voice clip.
export const PRESETS = [
  { id: 'greet', label: 'Greeting', text: 'こんにちは！はじめまして、よろしくね〜！', en: 'Hi! Nice to meet you!',
    cues: [[0, 'expr', 'smile'], [0.05, 'tilt', { deg: 7, dur: 0.9 }], [0.15, 'nod', { amp: 6 }], [0.9, 'look', { x: -6, y: 3, dur: 0.8 }],
           [1.7, 'expr', 'happy'], [1.75, 'bounce', { amp: 0.06, n: 1 }], [1.9, 'nod', { amp: 10 }], [2.2, 'tilt', { deg: -9, dur: 1.4 }]], tail: 1.2 },
  { id: 'jump', label: 'Jump!', text: 'よーし、いっくよー！せーのっ……えいっ！', en: 'Okay, here I go! Ready… hup!',
    cues: [[0, 'expr', 'smile'], [0.05, 'nod', { amp: 7 }], [1.0, 'expr', 'happy'], [1.05, 'bounce', { amp: 0.05, n: 2, dur: 0.7 }],
           [2.1, 'expr', 'neutral'], [2.15, 'look', { y: -5, dur: 1.2, a: 0.3 }], [2.2, 'lean', { z: -2, dur: 1.2 }],
           [3.35, 'jump', { h: 1.15 }], [3.6, 'expr', 'surprised'], [4.3, 'expr', 'happy'], [4.9, 'bounce', { amp: 0.1, n: 2 }]], tail: 2.2 },
  { id: 'laugh', label: 'Giggle', text: 'えへへっ、ほんとに？うれしいなぁ〜！', en: 'Ehehe, really? I\'m so happy!',
    cues: [[0, 'expr', 'happy'], [0, 'bounce', { amp: 0.05, n: 3, dur: 0.85 }], [0.05, 'look', { x: 5, y: -4, dur: 0.9 }],
           [1.05, 'expr', 'surprised'], [1.08, 'tilt', { deg: 9, dur: 1.0 }], [2.25, 'expr', 'happy'], [2.3, 'sway', { amp: 3.5, n: 1.5, dur: 1.6 }]], tail: 1.0 },
  { id: 'surprised', label: 'Surprised', text: 'えっ！？うそ、ほんとに！？', en: 'Huh!? No way, really!?',
    cues: [[0, 'expr', 'surprised'], [0, 'jolt', {}], [0.75, 'look', { x: 4, y: 4, dur: 0.9 }], [0.8, 'shake', { amp: 5, n: 1 }],
           [1.25, 'expr', 'happy'], [1.3, 'bounce', { amp: 0.09, n: 2 }]], tail: 1.3 },
  { id: 'shy', label: 'Shy', text: 'そ、そんなに見つめないでよ……はずかしいじゃん……', en: 'D-don\'t stare at me like that… it\'s embarrassing…',
    cues: [[0, 'expr', 'shy'], [0, 'jolt', { amp: 0.4 }], [0.45, 'look', { x: -14, y: -6, eyeX: -0.8, eyeY: -0.6, dur: 2.2, a: 0.15 }],
           [0.5, 'lean', { x: -3, z: 2.5, dur: 3.2 }], [2.5, 'shake', { amp: 4, n: 1 }], [2.6, 'look', { x: 10, y: -8, eyeX: 0.6, eyeY: -0.7, dur: 1.6 }],
           [2.7, 'sway', { amp: 2, n: 1, dur: 1.4 }]], tail: 1.4 },
  { id: 'pout', label: 'Sulk', text: 'むぅ〜、なんで待っててくれなかったの？ふんっ！', en: 'Hmph~ why didn\'t you wait for me? Hmph!',
    cues: [[0, 'expr', 'pout'], [0.05, 'look', { y: -4, dur: 1.0 }], [1.2, 'expr', 'angry'], [1.25, 'shake', { amp: 6, n: 2, dur: 0.9 }],
           [2.2, 'nod', { amp: 5, n: 2 }], [3.45, 'expr', 'pout'], [3.45, 'look', { x: 22, y: 6, eyeX: 0.9, eyeY: 0.3, dur: 2.0, a: 0.06 }],
           [3.45, 'lean', { x: 5, dur: 2.0 }], [3.5, 'bounce', { amp: 0.05, n: 1 }]], tail: 1.6 },
  { id: 'think', label: 'Idea!', text: 'うーん……どうしようかなぁ……あっ、そうだ！', en: 'Hmm… what should I do… oh, I know!',
    cues: [[0, 'expr', 'neutral'], [0, 'look', { x: 8, y: 9, eyeX: 0.7, eyeY: 0.9, dur: 3.4, a: 0.12 }], [0.1, 'tilt', { deg: 11, dur: 3.3 }],
           [0.3, 'lean', { z: 2.5, dur: 3.2 }], [1.4, 'expr', 'pout'], [2.0, 'sway', { amp: 1.5, n: 1, dur: 1.3 }],
           [3.8, 'expr', 'surprised'], [3.8, 'jolt', { amp: 0.7 }], [4.3, 'expr', 'happy'], [4.32, 'bounce', { amp: 0.1, n: 2 }], [4.4, 'nod', { amp: 8 }]], tail: 1.2 },
  { id: 'bye', label: 'Goodbye', text: '今日はありがとう！またね〜、ばいばーい！', en: 'Thanks for today! See you, bye-bye!',
    cues: [[0, 'expr', 'smile'], [0.05, 'nod', { amp: 11, dur: 0.7 }], [0.9, 'expr', 'happy'], [0.9, 'sway', { amp: 4.5, n: 2, dur: 1.6 }],
           [1.6, 'bounce', { amp: 0.05, n: 1 }], [2.35, 'expr', 'wink'], [2.35, 'tilt', { deg: 8, dur: 1.4 }]], tail: 1.4 },
];

// ------------------------------------------------------------------ player
export class PresetPlayer {
  constructor(ctrl, { onLine, onEnd } = {}) {
    this.ctrl = ctrl; this.onLine = onLine; this.onEnd = onEnd;
    this.buffers = {}; this.cur = null;
  }
  _ac() {
    if (!this.ac) {
      this.ac = new (window.AudioContext || window.webkitAudioContext)();
      this.mouth = new AudioMouth(this.ctrl, this.ac);
      this.mouth.an.connect(this.ac.destination);
    }
    return this.ac;
  }
  async _buffer(id) {
    if (!this.buffers[id]) this.buffers[id] = fetch(`assets/voice/${id}.mp3`).then(r => r.arrayBuffer()).then(b => this._ac().decodeAudioData(b));
    return this.buffers[id];
  }
  preload() { for (const p of PRESETS) this._buffer(p.id).catch(() => {}); }

  async play(preset) {
    const ac = this._ac(); if (ac.state === 'suspended') await ac.resume();
    this.stop(true);
    let buf = null;
    try { buf = await this._buffer(preset.id); } catch (e) { console.warn('voice missing for', preset.id, e); }
    const cur = { preset, i: 0, t0: ac.currentTime + 0.03, dur: (buf ? buf.duration : 2) + (preset.tail || 1), prevExpr: this.ctrl.expr };
    if (buf) {
      const src = ac.createBufferSource(); src.buffer = buf; src.connect(this.mouth.an); src.start(cur.t0);
      cur.src = src; this.mouth.start();
    }
    this.cur = cur; this.ctrl.preset = this;
    this.onLine?.(preset, cur.dur);
  }

  stop(silent = false) {
    const cur = this.cur; if (!cur) return;
    try { cur.src?.stop(); } catch (e) {}
    this.mouth?.stop(); this.cur = null; this.ctrl.preset = null;
    this.ctrl.gestures.length = 0;
    this.ctrl.setExpression('neutral');
    if (!silent) this.onEnd?.(cur.preset);
  }

  // called by the Controller every frame
  tick() {
    const cur = this.cur; if (!cur) return;
    const t = this.ac.currentTime - cur.t0, cues = cur.preset.cues;
    while (cur.i < cues.length && cues[cur.i][0] <= t) {
      const [, kind, arg] = cues[cur.i++];
      if (kind === 'expr') this.ctrl.setExpression(arg);
      else this.ctrl.gesture(GESTURES[kind](arg || {}));
    }
    if (t > cur.dur) this.stop();
  }
}
