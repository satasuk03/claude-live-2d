import { Renderer } from './gl.js';
import { Rig, PARAMS } from './rig.js';
import { Controller, MicLipSync, EXPRESSIONS } from './motion.js';
import { FaceTracker } from './tracking.js';
import { PRESETS, PresetPlayer, GESTURES } from './presets.js';

const $ = s => document.querySelector(s);
const canvas = $('#stage');
const renderer = new Renderer(canvas);

async function loadImages(parts, onProgress) {
  const names = Object.keys(parts); let done = 0;
  const out = {};
  await Promise.all(names.map(n => new Promise((res, rej) => {
    const im = new Image(); im.decoding = 'async';
    im.onload = () => { out[n] = im; onProgress(++done / names.length); res(); };
    im.onerror = rej; im.src = 'assets/parts/' + parts[n].file;
  })));
  // sheen masks for the satin dress
  await Promise.all(names.filter(n => parts[n].sheen).map(n => new Promise((res) => {
    const im = new Image(); im.onload = () => { out['sheen:' + n] = im; res(); }; im.onerror = res; im.src = 'assets/parts/' + parts[n].sheen;
  })));
  return out;
}

// ------------------------------------------------------------------ camera
const FRAMES = {
  face: { cx: 1750, cy: 1080, h: 1250 },
  bust: { cx: 1750, cy: 1380, h: 2650 },
  full: { cx: 1750, cy: 2480, h: 5150 },
};
const cam = { cx: 1750, cy: 1380, h: 2650, tcx: 1750, tcy: 1380, th: 2650 };
function setFrame(name) { const f = FRAMES[name]; cam.tcx = f.cx; cam.tcy = f.cy; cam.th = f.h; }
function viewMatrix() {
  const w = canvas.width, h = canvas.height, aspect = w / h;
  const panelOffset = window.innerWidth > 900 && !document.body.classList.contains('panel-hidden') ? 0.12 : 0;     // keep the model clear of the side panel
  const sy = 2 / cam.h, sx = sy / aspect;
  const tx = -cam.cx * sx - panelOffset, ty = cam.cy * sy;
  return new Float32Array([sx, 0, 0, 0, -sy, 0, tx, ty, 1]);
}
function screenToModel(px, py) {
  const r = canvas.getBoundingClientRect(); const v = viewMatrix();
  const nx = ((px - r.left) / r.width) * 2 - 1, ny = -(((py - r.top) / r.height) * 2 - 1);
  return [(nx - v[6]) / v[0], (ny - v[7]) / v[4]];
}

// ------------------------------------------------------------------ boot
(async () => {
  const partsJson = await (await fetch('assets/parts.json')).json();
  const images = await loadImages(partsJson.parts, p => { $('#loadbar i').style.width = (p * 100).toFixed(0) + '%'; });
  const rig = new Rig(renderer, partsJson, images);
  const ctrl = new Controller(rig);
  window.rig = rig; window.ctrl = ctrl; window.renderer = renderer;
  $('#loader').classList.add('done');

  // ---------------------------------------------------------------- interaction
  let dragging = null;
  canvas.addEventListener('pointermove', e => {
    const r = canvas.getBoundingClientRect();
    // pointer relative to the face position on screen
    const v = viewMatrix();
    const fx = ((1750 * v[0] + v[6]) + 1) / 2 * r.width + r.left, fy = (1 - (1080 * v[4] + v[7])) / 2 * r.height + r.top;
    ctrl.pointer.x = Math.max(-1, Math.min(1, (e.clientX - fx) / (r.width * 0.45)));
    ctrl.pointer.y = Math.max(-1, Math.min(1, (e.clientY - fy) / (r.height * 0.45)));
    ctrl.pointer.active = true; ctrl.pointer.last = ctrl.t;
    if (dragging) {
      const k = cam.h / r.height;
      cam.tcx = dragging.cx - (e.clientX - dragging.x) * k; cam.tcy = dragging.cy - (e.clientY - dragging.y) * k;
      cam.cx = cam.tcx; cam.cy = cam.tcy;
    }
  });
  canvas.addEventListener('pointerleave', () => { ctrl.pointer.active = false; });
  canvas.addEventListener('pointerdown', e => {
    if (e.button === 1 || e.button === 2 || e.shiftKey) { dragging = { x: e.clientX, y: e.clientY, cx: cam.cx, cy: cam.cy }; return; }
    const [mx, my] = screenToModel(e.clientX, e.clientY);
    let region = null;
    if (my < 900 && Math.abs(mx - 1750) < 420) region = 'head';
    else if (my < 1460 && Math.abs(mx - 1750) < 300) region = 'face';
    else if (my < 1700 && Math.abs(mx - 1750) < 500) region = 'hair';
    else if (my < 4300 && Math.abs(mx - 1750) < 900) region = 'body';
    if (region) { ctrl.poke(region); highlightExpr(ctrl.expr); }
  });
  window.addEventListener('pointerup', () => dragging = null);
  canvas.addEventListener('contextmenu', e => e.preventDefault());
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const f = Math.exp(e.deltaY * 0.0012);
    cam.th = Math.max(700, Math.min(6500, cam.th * f));
  }, { passive: false });

  // ---------------------------------------------------------------- UI: expressions
  const exprBox = $('#exprs');
  const labels = { neutral: 'Neutral', smile: 'Smile', happy: 'Happy', surprised: 'Surprised', wink: 'Wink', pout: 'Pout', angry: 'Angry', shy: 'Shy' };
  for (const k of Object.keys(EXPRESSIONS)) {
    const b = document.createElement('button'); b.textContent = labels[k] || k; b.dataset.expr = k;
    b.onclick = () => { ctrl.setExpression(k); highlightExpr(k); };
    exprBox.appendChild(b);
  }
  function highlightExpr(k) { exprBox.querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.expr === k)); }
  highlightExpr('neutral');

  // ---------------------------------------------------------------- UI: toggles
  const bindToggle = (id, get, set) => { const el = $(id); el.checked = get(); el.onchange = () => set(el.checked); };
  bindToggle('#tFollow', () => ctrl.follow, v => ctrl.follow = v);
  bindToggle('#tBlink', () => ctrl.autoBlink, v => ctrl.autoBlink = v);
  bindToggle('#tBreath', () => ctrl.breath, v => ctrl.breath = v);
  bindToggle('#tIdle', () => ctrl.idle, v => ctrl.idle = v);
  bindToggle('#tPhysics', () => rig.physicsOn, v => rig.physicsOn = v);
  bindToggle('#tWire', () => rig.wire, v => rig.wire = v);
  bindToggle('#tSheen', () => rig.sheenOn, v => rig.sheenOn = v);
  document.querySelectorAll('[data-frame]').forEach(b => b.onclick = () => {
    setFrame(b.dataset.frame); document.querySelectorAll('[data-frame]').forEach(x => x.classList.toggle('on', x === b));
  });
  const TINT = { night: { mul: [0.93, 0.94, 1.0], sheen: [0.95, 0.93, 1.0] }, dusk: { mul: [0.96, 0.9, 0.96], sheen: [1.0, 0.86, 0.9] },
                 studio: { mul: [1, 1, 1], sheen: [1, 0.98, 0.95] }, green: { mul: [1, 1, 1], sheen: [1, 0.98, 0.95] } };
  const applyBg = name => { document.body.dataset.bg = name; renderer.globalMul = TINT[name].mul; rig.sheenCol = TINT[name].sheen;
    document.querySelectorAll('[data-bg]').forEach(x => x.classList.toggle('on', x.dataset.bg === name)); };
  document.querySelectorAll('[data-bg]').forEach(b => b.onclick = () => applyBg(b.dataset.bg));
  applyBg(document.body.dataset.bg || 'night');

  // webcam
  const tracker = new FaceTracker(ctrl, $('#cam'));
  $('#bCam').onclick = async () => {
    const b = $('#bCam');
    if (tracker.on) { tracker.stop(); b.classList.remove('on'); b.textContent = 'Webcam tracking'; $('#camwrap').classList.remove('show'); return; }
    try {
      await tracker.start(s => b.textContent = s);
      b.classList.add('on'); b.textContent = 'Stop tracking'; $('#camwrap').classList.add('show');
    } catch (err) { console.error(err); b.textContent = 'Camera unavailable'; }
  };
  $('#bCalib').onclick = () => tracker.recalibrate();
  // mic
  const mic = new MicLipSync(ctrl);
  $('#bMic').onclick = async () => {
    const b = $('#bMic');
    if (mic.on) { mic.stop(); b.classList.remove('on'); b.textContent = 'Mic lip-sync'; return; }
    try { await mic.start(); b.classList.add('on'); b.textContent = 'Stop mic'; } catch (err) { console.error(err); b.textContent = 'Mic unavailable'; }
  };
  // ---------------------------------------------------------------- voice presets
  const presetBox = $('#presets');
  const player = new PresetPlayer(ctrl, {
    onLine: (p, dur) => {
      const bub = $('#bubble'); bub.innerHTML = '<span class="jp"></span><span class="en"></span>';
      bub.querySelector('.jp').textContent = p.text; bub.querySelector('.en').textContent = p.en;
      bub.classList.add('show');
      clearTimeout(window._bubbleT); window._bubbleT = setTimeout(() => bub.classList.remove('show'), dur * 1000);
      presetBox.querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.preset === p.id));
      $('#presetNow').textContent = '· ' + p.label;
    },
    onEnd: () => { presetBox.querySelectorAll('button').forEach(b => b.classList.remove('on')); $('#presetNow').textContent = ''; highlightExpr(ctrl.expr); },
  });
  window.player = player;
  PRESETS.forEach((p, i) => {
    const b = document.createElement('button'); b.textContent = p.label; b.dataset.preset = p.id; b.title = `${p.text}\n${p.en}  (Shift+${i + 1})`;
    b.onclick = () => playPreset(p);
    presetBox.appendChild(b);
  });
  function playPreset(p) {
    if (player.cur?.preset === p) { player.stop(); $('#bubble').classList.remove('show'); return; }    // click again to stop
    player.play(p);
  }
  // preload once the user has interacted (AudioContext needs a gesture)
  window.addEventListener('pointerdown', () => player.preload(), { once: true });
  $('#bTalk').onclick = () => {
    const pool = PRESETS.filter(p => p !== player.cur?.preset);
    playPreset(pool[Math.floor(Math.random() * pool.length)]);
  };
  $('#bJump').onclick = () => ctrl.gesture(GESTURES.jump({}));

  // ---------------------------------------------------------------- UI: parameter sliders
  const box = $('#params'); let group = null;
  for (const [id, mn, mx, df, grp] of PARAMS) {
    if (grp !== group) { group = grp; const h = document.createElement('div'); h.className = 'pgroup'; h.textContent = grp; box.appendChild(h); }
    const row = document.createElement('label'); row.className = 'prow';
    row.innerHTML = `<span>${id}</span><input type="range" min="${mn}" max="${mx}" step="0.01" value="${df}"><b></b>`;
    const inp = row.querySelector('input'), val = row.querySelector('b');
    inp.oninput = () => { ctrl.overrides[id] = parseFloat(inp.value); row.classList.add('pinned'); };
    row.ondblclick = () => { delete ctrl.overrides[id]; row.classList.remove('pinned'); };
    row._sync = () => { if (!(id in ctrl.overrides)) inp.value = rig.params[id]; val.textContent = (+rig.params[id]).toFixed(2); };
    box.appendChild(row);
  }
  $('#bReset').onclick = () => { ctrl.overrides = {}; box.querySelectorAll('.prow').forEach(r => r.classList.remove('pinned')); };
  const rows = [...box.querySelectorAll('.prow')];
  $('#panelToggle').onclick = () => document.body.classList.toggle('panel-hidden');
  window.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT') return;
    const keys = Object.keys(EXPRESSIONS);
    const dig = /^Digit([1-9])$/.exec(e.code);
    if (dig && e.shiftKey) { const p = PRESETS[+dig[1] - 1]; if (p) playPreset(p); return; }
    if (e.key === 'Escape' && player.cur) { player.stop(); $('#bubble').classList.remove('show'); }
    if (e.key === 'j') $('#bJump').click();
    if (e.key >= '1' && e.key <= String(keys.length)) { const k = keys[+e.key - 1]; ctrl.setExpression(k); highlightExpr(k); }
    if (e.key === 'w') { rig.wire = !rig.wire; $('#tWire').checked = rig.wire; }
    if (e.key === 'h') document.body.classList.toggle('panel-hidden');
    if (e.key === 't') $('#bTalk').click();
  });

  // ---------------------------------------------------------------- loop
  let last = performance.now(), fpsT = 0, frames = 0;
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    const k = 1 - Math.exp(-dt * 6);
    cam.cx += (cam.tcx - cam.cx) * k; cam.cy += (cam.tcy - cam.cy) * k; cam.h += (cam.th - cam.h) * k;
    ctrl.update(dt);
    rig.update(dt);
    renderer.begin(viewMatrix());
    rig.draw();
    if (!document.body.classList.contains('panel-hidden')) for (const r of rows) r._sync();
    frames++; fpsT += dt; if (fpsT > 0.5) { $('#fps').textContent = Math.round(frames / fpsT) + ' fps'; frames = 0; fpsT = 0; }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})().catch(err => { console.error(err); $('#loader').innerHTML = '<p style="color:#f88">Failed to load: ' + err.message + '</p>'; });
