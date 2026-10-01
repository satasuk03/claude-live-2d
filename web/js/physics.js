// Cubism-style pendulum physics: a chain of particles hanging from an anchor that is pushed
// around by the head/body motion. Each chain outputs the swing angle of its segments.

export class Pendulum {
  constructor({ segments = 2, length = 1, gravity = 1, damping = 0.08, stiffness = 0.0, mass = 1, scaleX = 1, scaleAngle = 1 } = {}) {
    this.n = segments; this.len = length / segments;
    this.gravity = gravity; this.damping = damping; this.stiffness = stiffness; this.mass = mass;
    this.scaleX = scaleX; this.scaleAngle = scaleAngle;
    this.p = []; this.q = [];
    for (let i = 0; i <= this.n; i++) { this.p.push([0, i * this.len]); this.q.push([0, i * this.len]); }
    this.angles = new Array(this.n).fill(0);
    this.anchorPrev = 0;
  }
  // anchorX/anchorY: displacement of the root (head translation, y down), tilt: rotation of gravity (radians)
  step(dt, anchorX, tilt, wind = 0, anchorY = 0) {
    const sub = 4, h = Math.min(dt, 1 / 20) / sub;
    if (this.anchorPrevY === undefined) this.anchorPrevY = anchorY;
    for (let s = 0; s < sub; s++) {
      const f = (s + 1) / sub;
      const ax = this.anchorPrev + (anchorX - this.anchorPrev) * f, ay = this.anchorPrevY + (anchorY - this.anchorPrevY) * f;
      this.p[0][0] = ax; this.p[0][1] = ay; this.q[0][0] = ax; this.q[0][1] = ay;
      const gx = Math.sin(tilt) * this.gravity, gy = Math.cos(tilt) * this.gravity;
      for (let i = 1; i <= this.n; i++) {
        const P = this.p[i], Q = this.q[i];
        const vx = (P[0] - Q[0]) * (1 - this.damping), vy = (P[1] - Q[1]) * (1 - this.damping);
        // stiffness pulls each particle towards its rest position directly below its parent
        const par = this.p[i - 1];
        const rx = par[0] + Math.sin(tilt) * this.len * 0, ry = par[1] + this.len;
        const fx = gx * 60 + (rx - P[0]) * this.stiffness * 400 + wind * 30;
        const fy = gy * 60 + (ry - P[1]) * this.stiffness * 400;
        Q[0] = P[0]; Q[1] = P[1];
        P[0] += vx + fx * h * h / this.mass; P[1] += vy + fy * h * h / this.mass;
      }
      for (let it = 0; it < 2; it++) for (let i = 1; i <= this.n; i++) {
        const A = this.p[i - 1], B = this.p[i];
        let dx = B[0] - A[0], dy = B[1] - A[1]; const d = Math.hypot(dx, dy) || 1e-6;
        const k = this.len / d; B[0] = A[0] + dx * k; B[1] = A[1] + dy * k;
      }
    }
    this.anchorPrev = anchorX; this.anchorPrevY = anchorY;
    let prevAng = 0;
    for (let i = 0; i < this.n; i++) {
      const A = this.p[i], B = this.p[i + 1];
      const a = Math.atan2(B[0] - A[0], B[1] - A[1]);   // 0 = hanging straight down
      this.angles[i] = (a - tilt) * this.scaleAngle; prevAng = a;
    }
    return this.angles;
  }
}

// Vertical spring-mass hanging from a moving anchor. Output `lift`: 0 at rest, ~1 when the anchor
// is in free fall (hair floats up), negative when it is pushed up hard (landing, take-off).
export class Lift {
  constructor({ omega = 6, zeta = 0.3, gravity = 60 } = {}) {
    this.k = omega * omega; this.c = 2 * zeta * omega; this.g = gravity;
    this.s0 = gravity / this.k;                       // rest sag
    this.m = null; this.v = 0; this.aPrev = 0; this.lift = 0;
  }
  step(dt, anchorY) {
    if (this.m === null) { this.m = anchorY + this.s0; this.aPrev = anchorY; }
    const sub = 4, h = Math.min(dt, 1 / 20) / sub, av = (anchorY - this.aPrev) / Math.max(dt, 1e-4);
    for (let s = 0; s < sub; s++) {
      const a = this.aPrev + (anchorY - this.aPrev) * ((s + 1) / sub);
      const acc = this.g - this.k * (this.m - a) - this.c * (this.v - av);
      this.v += acc * h; this.m += this.v * h;
    }
    this.aPrev = anchorY;
    this.lift = Math.max(-1, Math.min(1.2, 1 - (this.m - anchorY) / this.s0));
    return this.lift;
  }
}

export function makePhysics() {
  return {
    bangs: new Pendulum({ segments: 2, length: 1.0, damping: 0.11, stiffness: 0.35, scaleAngle: 0.75 }),
    sideL: new Pendulum({ segments: 2, length: 1.4, damping: 0.07, stiffness: 0.15, scaleAngle: 0.9 }),
    sideR: new Pendulum({ segments: 2, length: 1.4, damping: 0.07, stiffness: 0.15, scaleAngle: 0.9 }),
    pony: new Pendulum({ segments: 2, length: 1.3, damping: 0.06, stiffness: 0.2, scaleAngle: 0.8 }),
    earL: new Pendulum({ segments: 1, length: 0.5, damping: 0.035, stiffness: 0.0, scaleAngle: 1.0 }),
    earR: new Pendulum({ segments: 1, length: 0.5, damping: 0.035, stiffness: 0.0, scaleAngle: 1.0 }),
    hairLift: new Lift({ omega: 5.5, zeta: 0.5 }),
    earLift: new Lift({ omega: 8, zeta: 0.35 }),
  };
}
