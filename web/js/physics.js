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
  // anchorX: horizontal displacement of the root (head translation), tilt: rotation of gravity (radians)
  step(dt, anchorX, tilt, wind = 0) {
    const sub = 4, h = Math.min(dt, 1 / 20) / sub;
    for (let s = 0; s < sub; s++) {
      const ax = this.anchorPrev + (anchorX - this.anchorPrev) * ((s + 1) / sub);
      this.p[0][0] = ax; this.p[0][1] = 0; this.q[0][0] = ax; this.q[0][1] = 0;
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
    this.anchorPrev = anchorX;
    let prevAng = 0;
    for (let i = 0; i < this.n; i++) {
      const A = this.p[i], B = this.p[i + 1];
      const a = Math.atan2(B[0] - A[0], B[1] - A[1]);   // 0 = hanging straight down
      this.angles[i] = (a - tilt) * this.scaleAngle; prevAng = a;
    }
    return this.angles;
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
  };
}
