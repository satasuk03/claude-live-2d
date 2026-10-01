// Webcam face tracking with MediaPipe Face Landmarker (blendshapes + head pose).
const VISION = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
const MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

export class FaceTracker {
  constructor(ctrl, videoEl) { this.ctrl = ctrl; this.video = videoEl; this.on = false; this.calib = null; }
  async start(onStatus = () => {}) {
    onStatus('loading tracker…');
    const { FaceLandmarker, FilesetResolver } = await import(`${VISION}/vision_bundle.mjs`);
    const files = await FilesetResolver.forVisionTasks(`${VISION}/wasm`);
    this.lm = await FaceLandmarker.createFromOptions(files, {
      baseOptions: { modelAssetPath: MODEL, delegate: 'GPU' }, runningMode: 'VIDEO', numFaces: 1,
      outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true,
    });
    onStatus('starting camera…');
    this.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } });
    this.video.srcObject = this.stream; await this.video.play();
    this.on = true; onStatus('tracking');
    let last = -1;
    const loop = () => {
      if (!this.on) return;
      if (this.video.currentTime !== last) {
        last = this.video.currentTime;
        const r = this.lm.detectForVideo(this.video, performance.now());
        this._apply(r);
      }
      requestAnimationFrame(loop);
    };
    loop();
  }
  stop() { this.on = false; this.ctrl.face = null; this.stream?.getTracks().forEach(t => t.stop()); }
  recalibrate() { this.calib = null; }
  _apply(r) {
    if (!r.faceBlendshapes?.length) { this.ctrl.face = null; return; }
    const bs = {}; for (const c of r.faceBlendshapes[0].categories) bs[c.categoryName] = c.score;
    const m = r.facialTransformationMatrixes[0].data;   // column-major 4x4
    // rotation -> yaw/pitch/roll (degrees)
    const r00 = m[0], r10 = m[1], r20 = m[2], r21 = m[6], r22 = m[10];
    let yaw = Math.atan2(-r20, Math.hypot(r21, r22)) * 180 / Math.PI;
    let pitch = Math.atan2(r21, r22) * 180 / Math.PI;
    let roll = Math.atan2(r10, r00) * 180 / Math.PI;
    if (!this.calib) this.calib = { yaw, pitch, roll };
    yaw -= this.calib.yaw; pitch -= this.calib.pitch; roll -= this.calib.roll;
    // mirror: the character mirrors the user like a reflection
    const g = n => bs[n] || 0;
    const blinkL = g('eyeBlinkRight'), blinkR = g('eyeBlinkLeft');
    const open = b => clamp(1.15 - b * 1.35, 0, 1.2);
    const lookX = (g('eyeLookOutRight') - g('eyeLookInRight') + g('eyeLookInLeft') - g('eyeLookOutLeft')) * 0.5;
    const lookY = ((g('eyeLookUpLeft') + g('eyeLookUpRight')) - (g('eyeLookDownLeft') + g('eyeLookDownRight'))) * 0.5;
    this.ctrl.face = {
      yaw: clamp(-yaw * 1.3, -30, 30), pitch: clamp(-pitch * 1.3, -30, 30), roll: clamp(-roll, -30, 30),
      eyeL: open(blinkL), eyeR: open(blinkR), eyeX: clamp(-lookX * 1.6, -1, 1), eyeY: clamp(lookY * 1.6, -1, 1),
      jaw: clamp(g('jawOpen') * 1.6, 0, 1), smile: clamp((g('mouthSmileLeft') + g('mouthSmileRight')) * 0.75, 0, 1),
      frown: clamp((g('mouthFrownLeft') + g('mouthFrownRight')) * 0.8, 0, 1), pucker: clamp(g('mouthPucker') * 1.3, 0, 1),
      funnel: clamp(g('mouthFunnel') * 1.5, 0, 1), brow: clamp(g('browInnerUp') * 1.4 - (g('browDownLeft') + g('browDownRight')) * 0.6, -1, 1),
    };
  }
}
