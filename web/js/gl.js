// Minimal WebGL2 mesh renderer with premultiplied alpha, stencil clipping and blend modes.

const VS = `#version 300 es
in vec2 aPos; in vec2 aUV;
uniform mat3 uView;
out vec2 vUV; out vec2 vPos;
void main(){ vec3 p = uView * vec3(aPos,1.0); gl_Position = vec4(p.xy,0.0,1.0); vUV = aUV; vPos = aPos; }`;

const FS = `#version 300 es
precision highp float;
in vec2 vUV; in vec2 vPos; uniform sampler2D uTex; uniform float uOpacity; uniform vec3 uMul; uniform vec3 uScreen;
uniform float uMaskCut;
uniform sampler2D uSheenTex; uniform float uSheen; uniform vec4 uSheenP; // x: band pos, y: glint pos, z: width, w: glint strength
uniform vec3 uSheenCol;
out vec4 o;
void main(){
  vec4 c = texture(uTex, vUV);           // premultiplied
  if (uMaskCut > 0.0 && c.a < uMaskCut) discard;
  vec3 rgb = c.a > 0.0 ? c.rgb / c.a : vec3(0.0);
  if (uSheen > 0.0) {
    float m = texture(uSheenTex, vUV).r;
    float coord = vPos.x * 0.55 + vPos.y;                     // diagonal light sweep in model space
    float band = exp(-pow((coord - uSheenP.x) / uSheenP.z, 2.0));
    float glint = exp(-pow((coord - uSheenP.y) / (uSheenP.z * 0.18), 2.0)) * uSheenP.w;
    rgb += uSheenCol * m * (band * 0.22 + glint * 0.55) * uSheen;
  }
  rgb = rgb * uMul;
  rgb = rgb + uScreen - rgb * uScreen;
  o = vec4(rgb * c.a, c.a) * uOpacity;
}`;

const WIRE_FS = `#version 300 es
precision highp float; uniform vec4 uColor; out vec4 o; void main(){ o = uColor; }`;

function compile(gl, vs, fs) {
  const mk = (t, s) => { const sh = gl.createShader(t); gl.shaderSource(sh, s); gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh)); return sh; };
  const p = gl.createProgram();
  gl.attachShader(p, mk(gl.VERTEX_SHADER, vs)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs));
  gl.bindAttribLocation(p, 0, 'aPos'); gl.bindAttribLocation(p, 1, 'aUV');
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {}; const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) { const a = gl.getActiveUniform(p, i); u[a.name] = gl.getUniformLocation(p, a.name); }
  return { p, u };
}

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', { premultipliedAlpha: true, antialias: true, stencil: true, alpha: true });
    if (!gl) throw new Error('WebGL2 not supported');
    this.gl = gl;
    this.prog = compile(gl, VS, FS);
    this.wire = compile(gl, VS, WIRE_FS);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    this.maxAniso = 1;
    this.anisoExt = gl.getExtension('EXT_texture_filter_anisotropic');
    if (this.anisoExt) this.maxAniso = gl.getParameter(this.anisoExt.MAX_TEXTURE_MAX_ANISOTROPY_EXT);
  }

  texture(img) {
    const gl = this.gl, t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    if (this.anisoExt) gl.texParameterf(gl.TEXTURE_2D, this.anisoExt.TEXTURE_MAX_ANISOTROPY_EXT, Math.min(8, this.maxAniso));
    return t;
  }

  mesh(uv, idx, lineIdx) {
    const gl = this.gl;
    const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    const pos = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, pos);
    gl.bufferData(gl.ARRAY_BUFFER, uv.length * 4, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    const uvb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, uvb);
    gl.bufferData(gl.ARRAY_BUFFER, uv, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 0, 0);
    const ib = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
    gl.bindVertexArray(null);
    // separate VAO for wireframe
    const wvao = gl.createVertexArray(); gl.bindVertexArray(wvao);
    gl.bindBuffer(gl.ARRAY_BUFFER, pos); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    const lb = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, lb);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, lineIdx, gl.STATIC_DRAW);
    gl.bindVertexArray(null);
    return { vao, pos, count: idx.length, wvao, lineCount: lineIdx.length };
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    const w = Math.round(this.canvas.clientWidth * dpr), h = Math.round(this.canvas.clientHeight * dpr);
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
    this.gl.viewport(0, 0, w, h);
  }

  begin(view) {
    const gl = this.gl;
    this.resize();
    gl.clearColor(0, 0, 0, 0); gl.clearStencil(0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.STENCIL_BUFFER_BIT);
    gl.enable(gl.BLEND);
    this.view = view;
  }

  _blend(mode) {
    const gl = this.gl;
    if (mode === 'add') gl.blendFuncSeparate(gl.ONE, gl.ONE, gl.ZERO, gl.ONE);
    else if (mode === 'multiply') gl.blendFuncSeparate(gl.DST_COLOR, gl.ONE_MINUS_SRC_ALPHA, gl.ZERO, gl.ONE);
    else gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  }

  upload(m, positions) {
    const gl = this.gl; gl.bindBuffer(gl.ARRAY_BUFFER, m.pos); gl.bufferSubData(gl.ARRAY_BUFFER, 0, positions);
  }

  draw(m, tex, { opacity = 1, mul = this.globalMul || [1, 1, 1], screen = [0, 0, 0], blend = 'normal', stencilTest = false, stencilWrite = false, maskCut = 0, sheen = null } = {}) {
    const gl = this.gl, P = this.prog;
    gl.useProgram(P.p);
    if (sheen) {
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, sheen.tex); gl.uniform1i(P.u.uSheenTex, 1);
      gl.uniform1f(P.u.uSheen, sheen.strength); gl.uniform4fv(P.u.uSheenP, sheen.p); gl.uniform3fv(P.u.uSheenCol, sheen.col);
    } else gl.uniform1f(P.u.uSheen, 0);
    gl.uniformMatrix3fv(P.u.uView, false, this.view);
    gl.uniform1f(P.u.uOpacity, opacity); gl.uniform3fv(P.u.uMul, mul); gl.uniform3fv(P.u.uScreen, screen);
    gl.uniform1f(P.u.uMaskCut, maskCut);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tex); gl.uniform1i(P.u.uTex, 0);
    if (stencilWrite) {
      gl.enable(gl.STENCIL_TEST); gl.stencilFunc(gl.ALWAYS, 1, 0xff); gl.stencilOp(gl.KEEP, gl.KEEP, gl.REPLACE);
      gl.colorMask(false, false, false, false);
    } else if (stencilTest) {
      gl.enable(gl.STENCIL_TEST); gl.stencilFunc(gl.EQUAL, 1, 0xff); gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
    } else gl.disable(gl.STENCIL_TEST);
    this._blend(blend);
    gl.bindVertexArray(m.vao);
    gl.drawElements(gl.TRIANGLES, m.count, gl.UNSIGNED_INT, 0);
    gl.bindVertexArray(null);
    if (stencilWrite) gl.colorMask(true, true, true, true);
  }

  clearStencil() { const gl = this.gl; gl.clear(gl.STENCIL_BUFFER_BIT); gl.disable(gl.STENCIL_TEST); }

  drawWire(m, color) {
    const gl = this.gl, P = this.wire;
    gl.disable(gl.STENCIL_TEST);
    gl.useProgram(P.p); gl.uniformMatrix3fv(P.u.uView, false, this.view); gl.uniform4fv(P.u.uColor, color);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindVertexArray(m.wvao); gl.drawElements(gl.LINES, m.lineCount, gl.UNSIGNED_INT, 0); gl.bindVertexArray(null);
  }
}
