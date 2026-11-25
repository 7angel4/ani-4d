// Ani4D project page: outputs drawn live with WebGL, so viewers can turn them. One canvas holds one viewport per shown
// method (one row; the two-row layout, the input camera's view above the turnable view, is no longer used by the page),
// all sharing one orbit camera. Data (render/make_3d.py): media/3d/<id>.bin holds the input mesh's faces and colours, the
// midpoint subdivision the renders use, and the ground truth; media/3d/<id>_v<kk>.bin holds the five outputs for camera
// kk. The look follows the Blender renders (render/bl_video.py): three lights placed relative to the camera and a floor
// that only carries the shadow. Two colourings: the asset's own colours, or the error map (ERRMAP below). The edge-stretch
// colouring (neighbouring surface patches' change of distance against the ground truth's) was removed on 2026-11-23.
import * as THREE from "./three.module.min.js";
import { OrbitControls } from "./OrbitControls.js";

const BG = 0xf2f3f5;
const FX = 50 / 36;                                  // the renders' 50 mm lens on a 36 mm sensor
const FOV = 2 * Math.atan(18 / 50) * 180 / Math.PI;
const lin = v => (v /= 255) <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
// light levels relative to the first version, matched to the rendered videos (render/bl_video.py, 2026-11-09): the
// ground truth seen from the input camera is as bright in this live view as in the video row above it
export const LOOK = { world: 1.3, light: 1.25 };
// A card can pass `tone` (2026-11-23; render/tone_match.py -> data/look_fit.json -> build_page.py): the input video's own
// look, fitted on its frame 0 against the input video, as its rendered panels have it. The dataset renderer's light
// (scripts/renderer/render_env.py): a uniform grey world of radiance IN_WORLD (here an environment map, which lights
// both the diffuse and the specular part, as Blender's world does) and one sun of IN_SUN W/m2 (three.js's lights are in
// physical units: the same numbers) held at IN_SUN_REL in the camera's (right, forward, up), as bl_video.aim_input_sun
// does; the fitted material (rough, metal, unlit, emit) and a per-channel gain on the linear colours.
const IN_WORLD = 0.9 * 0.45, IN_SUN = 4.0, _A = 50 * Math.PI / 180, _B = 60 * Math.PI / 180;
const IN_SUN_REL = [Math.sin(_B) * Math.sin(_A), -Math.cos(_B) * Math.sin(_A), Math.cos(_A)];
const TOLIN = new Float32Array(256);                 // sRGB byte -> linear
for (let i = 0; i < 256; i++) TOLIN[i] = lin(i);
// The error map (2026-11-23): each output vertex's distance from the ground truth's same vertex (every output poses the
// input mesh, so vertex i is the same point of the asset in all of them), as a fraction of the truth's frame-0 size; drawn
// over the asset's own colours, desaturated and lightened: nothing below ERRMAP.lo, then yellow, orange and red, opaque
// from ERRMAP.full and darkest from ERRMAP.hi.
export const ERRMAP = {
  lo: 0.02, full: 0.1, hi: 0.2,
  stops: [[0.02, [255, 214, 102]], [0.05, [253, 141, 60]], [0.1, [222, 33, 38]], [0.2, [120, 0, 40]]],
};
const NS = 1024;
function heatTable(spec) {                           // 0 .. spec.hi -> sRGB heat colour (0..1) and opacity
  const H = new Float32Array(NS * 4);
  for (let i = 0; i < NS; i++) {
    const s = i / (NS - 1) * spec.hi, st = spec.stops;
    let k = 0;
    while (k < st.length - 2 && s > st[k + 1][0]) k++;
    const f = Math.max(0, Math.min(1, (s - st[k][0]) / (st[k + 1][0] - st[k][0])));
    for (let c = 0; c < 3; c++) H[4 * i + c] = (st[k][1][c] * (1 - f) + st[k + 1][1][c] * f) / 255;
    H[4 * i + 3] = s <= spec.lo ? 0 : Math.pow(Math.min(1, (s - spec.lo) / (spec.full - spec.lo)), 0.75);
  }
  return H;
}
const HEATE = heatTable(ERRMAP);

// a material that also emits its vertex colours at strength k (bl_video.vcol_material's `emit`: assets whose material
// emits a copy of its base colour); three.js has no per-vertex emission, so it is added to the shader's emission term
function emitVertexColours(mat, k) {
  mat.onBeforeCompile = sh => {
    sh.fragmentShader = sh.fragmentShader.replace("#include <emissivemap_fragment>",
      `#include <emissivemap_fragment>\n\ttotalEmissiveRadiance += vColor.rgb * ${Number(k).toFixed(4)};`);
  };
  mat.customProgramCacheKey = () => "emit" + k;
}

async function fetchBin(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  let ab = await r.arrayBuffer(), u8 = new Uint8Array(ab);
  if (u8[0] === 0x1f && u8[1] === 0x8b) {
    ab = await new Response(new Blob([u8]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
    u8 = new Uint8Array(ab);
  }
  if (String.fromCharCode(u8[0], u8[1], u8[2], u8[3]) !== "A4D1") throw new Error(`${url}: not an A4D1 file`);
  const hl = new DataView(ab).getUint32(4, true);
  const hdr = JSON.parse(new TextDecoder().decode(u8.subarray(8, 8 + hl)));
  const base = 8 + hl, blobs = {};
  for (const b of hdr.blobs) {
    const n = b.shape.reduce((a, x) => a * x, 1), off = base + b.off;
    const T = { u1: Uint8Array, u2: Uint16Array, u4: Uint32Array, i2: Int16Array, f4: Float32Array }[b.dtype];
    blobs[b.name] = Object.assign({ a: new T(ab, off, n) }, b);
  }
  return { hdr, blobs };
}

// one method's 16 frames (make_3d.encode_positions, undone): [Float32Array(3 n) per frame], xyz interleaved
function decodePositions(b, T, origin, step) {
  const N = b.n, S = 3 * N, M = T * S, by = b.a, r = new Int32Array(M);
  if (b.enc === "i2") for (let i = 0; i < M; i++) r[i] = ((by[i] | (by[M + i] << 8)) << 16) >> 16;
  else for (let i = 0; i < M; i++) r[i] = by[i] | (by[M + i] << 8) | (by[2 * M + i] << 16) | (by[3 * M + i] << 24);
  for (let i = 0; i < S && T > 1; i++) r[S + i] += r[i];
  for (let t = 2; t < T; t++) { const o = t * S; for (let i = 0; i < S; i++) r[o + i] += 2 * r[o - S + i] - r[o - 2 * S + i]; }
  const out = [];
  for (let t = 0; t < T; t++) {
    const P = new Float32Array(S), o = t * S;
    for (let a = 0; a < 3; a++) { const oa = o + a * N, g = origin[a]; for (let v = 0; v < N; v++) P[3 * v + a] = g + r[oa + v] * step; }
    out.push(P);
  }
  return out;
}

function camAxes(az, el) {                           // vidlib.cam_axes
  const a = az * Math.PI / 180, e = el * Math.PI / 180;
  const fwd = new THREE.Vector3(-Math.cos(e) * Math.cos(a), -Math.cos(e) * Math.sin(a), -Math.sin(e));
  const right = new THREE.Vector3(-Math.sin(a), Math.cos(a), 0);
  return { fwd, right, up: new THREE.Vector3().crossVectors(right, fwd) };
}

function frameCamera(P, az, el, margin = 0.07) {     // vidlib.frame_camera: target and distance that fit every point
  const { fwd, right, up } = camAxes(az, el), n = P.length / 3, c = new THREE.Vector3();
  for (let i = 0; i < n; i++) c.x += P[3 * i], c.y += P[3 * i + 1], c.z += P[3 * i + 2];
  c.multiplyScalar(1 / n);
  const proj = (ax, i) => (P[3 * i] - c.x) * ax.x + (P[3 * i + 1] - c.y) * ax.y + (P[3 * i + 2] - c.z) * ax.z;
  for (let it = 0; it < 4; it++) {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < n; i++) { const x = proj(right, i), y = proj(up, i); x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    c.addScaledVector(right, 0.5 * (x0 + x1)).addScaledVector(up, 0.5 * (y0 + y1));
  }
  let need = 0;
  for (let i = 0; i < n; i++) {
    const x = Math.abs(proj(right, i)), y = Math.abs(proj(up, i)), z = proj(fwd, i);
    need = Math.max(need, Math.max(x, y) * 2 * FX / (1 - 2 * margin) - z);
  }
  return { c, d: need, fwd };
}

// the shading normals a Turn3D keeps per frame, at most this many megabytes (Turn3D.normals, 2026-11-25)
const NORMALS_MB = 40;

export class Turn3D {
  // home: the view a load and a double-click go to ("turn" = 55 degrees from the input camera, "focus" = opts.focus)
  constructor(canvas, { root = "media/3d/", onState = () => {}, home = "turn", focus = null, tone = null } = {}) {
    this.canvas = canvas; this.root = root; this.onState = onState; this.home = home; this.focus = focus;
    this.tone = tone;                                // the input video's look (a card's), or null: the page's
    const r = this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    r.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFSoftShadowMap;
    r.setScissorTest(true);
    this.camera = new THREE.PerspectiveCamera(FOV, 1, 0.01, 200);
    this.camera.up.set(0, 0, 1);
    this.camTop = new THREE.PerspectiveCamera(FOV, 1, 0.01, 200);    // the input camera's view, for the two-row layout
    this.camTop.up.set(0, 0, 1);
    const ctl = this.controls = new OrbitControls(this.camera, canvas);
    ctl.enablePan = false;
    ctl.rotateSpeed = 0.85;
    ctl.zoomSpeed = 0.8;
    ctl.minPolarAngle = 0.05;
    ctl.maxPolarAngle = Math.PI * 0.6;             // down to a little below the floor, never under it
    ctl.addEventListener("change", () => this.render());
    canvas.style.touchAction = "pan-y";              // on a touchscreen a sideways drag turns, a vertical swipe still scrolls
    // the wheel zooms only with ctrl or cmd (a trackpad pinch sends ctrlKey), so the page still scrolls over the canvas;
    // in the two-row layout the top row (the input camera) does not turn
    canvas.addEventListener("wheel", e => { if ((!e.ctrlKey && !e.metaKey) || this.inTop(e)) e.stopImmediatePropagation(); }, { capture: true });
    canvas.addEventListener("pointerdown", e => { if (this.inTop(e)) e.stopImmediatePropagation(); }, { capture: true });
    canvas.addEventListener("dblclick", e => { if (!this.inTop(e)) this.preset(this.home); });
    this.keys = []; this.rows = 1; this.frame = 0; this.mode = "app"; this.skel = false; this.spin = false;
    this.panels = {}; this.files = new Map(); this.gt = null; this.token = 0;
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
  }

  inTop(e) { return this.rows === 2 && e.offsetY < this.canvas.clientHeight / 2; }

  get(url) {                                         // a small cache of fetched files (base files are shared by cameras)
    if (!this.files.has(url)) {
      const p = fetchBin(url);
      p.catch(() => this.files.delete(url));
      this.files.set(url, p);
      if (this.files.size > 6) this.files.delete(this.files.keys().next().value);
    }
    return this.files.get(url);
  }

  async load(id, k) {
    const tok = ++this.token, kk = String(k).padStart(2, "0");
    this.onState("loading");
    let base, cam;
    try { [base, cam] = await Promise.all([this.get(`${this.root}${id}.bin`), this.get(`${this.root}${id}_v${kk}.bin`)]); }
    catch (e) { if (tok === this.token) this.onState("error", e); return false; }
    if (tok !== this.token) return false;
    if (!this.gt || this.gt.id !== id) this.setBase(id, base);
    this.setCamera(cam);
    this.preset(this.home);
    this.update();
    this.render();
    this.onState("ready");
    return true;
  }

  // a mesh of the base file (suffix "" = the outputs', "_gt" = a two-mesh file's ground truth): faces, the subdivision the
  // page applies, that level's colours and their pale version under the error map's colours
  meshOf(hdr, blobs, sfx) {
    const nv = sfx ? hdr.nv_gt : hdr.nv, nl = sfx ? hdr.nv_view_gt : hdr.nv_view, levels = sfx ? hdr.levels_gt : hdr.levels;
    const F0 = blobs["F0" + sfx].a, sub = [];
    for (let i = 0; i < levels; i++) sub.push(blobs[`sub${i}${sfx}`].a);
    const F = blobs["F" + sfx] ? blobs["F" + sfx].a : F0, cb = blobs["col" + sfx].a;
    const col = new Float32Array(3 * nl), pale = new Float32Array(3 * nl);
    const gain = this.tone && this.tone.gain;        // the fitted per-channel gain on the linear colours
    for (let v = 0; v < nl; v++) {
      const c = [0, 1, 2].map(a => cb[3 * v + a]), g = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
      for (let a = 0; a < 3; a++) {
        col[3 * v + a] = gain ? Math.min(1, TOLIN[c[a]] * gain[a]) : TOLIN[c[a]];
        pale[3 * v + a] = (0.42 + 0.58 * (0.3 * c[a] + 0.7 * g) / 255) * 0.97;
      }
    }
    return { nv, nl, F0, F, sub, col, pale };
  }

  setBase(id, { hdr, blobs }) {
    const T = hdr.frames, two = !!hdr.two_mesh;
    const gt = decodePositions(blobs.V_gt, T, hdr.origin, hdr.step);
    this.gt = { id, hdr, T, two, size: hdr.step * 8191, V: gt, J: blobs.J_gt ? blobs.J_gt.a : null, mesh: {},
                parentsGt: two ? blobs.parents_gt.a : blobs.parents.a, njGt: two ? hdr.joints_gt : hdr.joints, outFrom: null,
                baseFile: { hdr, blobs } };
    if (two) this.gt.mesh.gt = this.meshOf(hdr, blobs, "_gt");          // the outputs' mesh comes with each camera
    else { this.setOut(hdr, blobs); this.gt.mesh.gt = this.gt.mesh.out; }
    this.disposePanels();
  }

  // the outputs' mesh and rig tree, and the error map's reference: one mesh = the base file's, the ground truth itself;
  // two meshes (a video-only clip, 2026-11-22) = the camera file's generated mesh and the truth carried onto it (V_gtc)
  setOut(hdr, blobs) {
    const g = this.gt, T = g.T, origin = g.hdr.origin, step = g.hdr.step, two = g.two;
    const out = this.meshOf(hdr, blobs, "");
    const ref = two ? decodePositions(blobs.V_gtc, T, origin, step) : g.V;
    Object.assign(g, { F0: out.F0, F: out.F, sub: out.sub, col: out.col, pale: out.pale, ref,
                       parents: blobs.parents.a, nj: hdr.joints });
    g.mesh.out = out;
  }

  // the mesh, skeleton tree and joint count a panel draws: the ground truth's own in a two-mesh file, else the shared ones
  look(key) {
    const g = this.gt, isGt = key === "gt";
    return { m: isGt ? g.mesh.gt : g.mesh.out, parents: isGt ? g.parentsGt : g.parents, nj: isGt ? g.njGt : g.nj };
  }

  setCamera(cam) {
    const { hdr, blobs } = cam, g = this.gt;
    // a two-mesh clip's camera brings its own generated mesh; a camera file without one (the first layout, 2026-11-22,
    // where the base held camera 0's) takes the base file's
    const src = blobs.F0 || !g.baseFile.blobs.F0 ? cam : g.baseFile;
    if (g.two && g.outFrom !== src) { this.setOut(src.hdr, src.blobs); g.outFrom = src; this.disposePanels(); }
    const seqs = { gt: { V: g.V, J: g.J } };
    for (const m of hdr.methods) seqs[m] = { V: decodePositions(blobs[`V_${m}`], g.T, g.hdr.origin, g.hdr.step), J: blobs[`J_${m}`] ? blobs[`J_${m}`].a : null };
    this.seqs = seqs;
    this.missing = new Set(hdr.missing || []);
    // the input camera's azimuth and elevation (it looks along -z of cam_R), for the presets
    const R = hdr.cam_R, f = [-R[0][2], -R[1][2], -R[2][2]];
    this.camAz = Math.atan2(-f[1], -f[0]) * 180 / Math.PI;
    this.camEl = Math.asin(Math.max(-1, Math.min(1, -f[2]))) * 180 / Math.PI;
    // a point sample of every output over all frames, to frame the shared camera on all of them (as the videos do)
    const nall = Object.values(seqs).reduce((n, s) => n + s.V[0].length / 3, 0);    // every output's vertices (+ the truth's)
    const pts = [], per = Math.max(1, Math.floor(nall * g.T / 24000));
    for (const s of Object.values(seqs)) for (let t = 0; t < g.T; t++) {
      const P = s.V[t];
      for (let i = (t * 7) % per; i < P.length / 3; i += per) pts.push(P[3 * i], P[3 * i + 1], P[3 * i + 2]);
    }
    this.pts = new Float32Array(pts);
    let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < this.pts.length; i += 3) for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], this.pts[i + a]); hi[a] = Math.max(hi[a], this.pts[i + a]); }
    this.box = { lo, hi, size: Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) };
    this.place(this.camTop, frameCamera(this.pts, this.camAz, this.camEl));
    this.buildPanels();
  }

  disposePanels() {
    for (const p of Object.values(this.panels)) p.scene.traverse(o => { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); });
    this.panels = {};
  }

  buildPanels() {
    const g = this.gt, size = g.size;
    const want = Object.keys(this.seqs);
    for (const k of Object.keys(this.panels)) if (!want.includes(k)) delete this.panels[k];
    for (const key of want) {
      if (this.panels[key]) continue;
      const scene = new THREE.Scene();
      const hemi = new THREE.HemisphereLight(0xffffff, 0xe6e8ec, 2.05 * LOOK.world);
      scene.add(hemi);
      const key1 = new THREE.DirectionalLight(0xffffff, 2.9 * LOOK.light), fill = new THREE.DirectionalLight(0xffffff, 0.85 * LOOK.light),
            rim = new THREE.DirectionalLight(0xffffff, 0.9 * LOOK.light), sun = new THREE.DirectionalLight(0xffffff, 0.5 * LOOK.light);
      sun.castShadow = true;
      sun.shadow.mapSize.set(1024, 1024);
      sun.shadow.bias = -0.0004;
      sun.shadow.normalBias = 0.01 * size;
      sun.shadow.radius = 4;
      for (const L of [key1, fill, rim, sun]) { scene.add(L); scene.add(L.target); }
      const tone = this.tone;
      let insun = null;
      if (tone) {                                    // the input video's light instead of the page's on the mesh
        for (const L of [hemi, key1, fill, rim, sun]) L.intensity = 0;   // `sun` still casts the floor's shadow
        insun = new THREE.DirectionalLight(0xffffff, IN_SUN);
        scene.add(insun); scene.add(insun.target);
        scene.environment = this.uniformEnv();
      }
      const L = this.look(key), nv = L.m.nl;
      const geom = new THREE.BufferGeometry();
      geom.setIndex(new THREE.BufferAttribute(L.m.F, 1));
      geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3 * nv), 3));
      geom.setAttribute("color", new THREE.BufferAttribute(L.m.col.slice(), 3));
      const mat = tone && tone.unlit ? new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide })
        : new THREE.MeshStandardMaterial({ vertexColors: true, roughness: tone ? tone.rough : 0.55, metalness: tone ? tone.metal : 0,
                                           side: THREE.DoubleSide });
      if (tone && tone.emit && !tone.unlit) emitVertexColours(mat, tone.emit);
      const mesh = new THREE.Mesh(geom, mat);
      mesh.castShadow = true;
      mesh.frustumCulled = false;
      scene.add(mesh);
      const floor = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.ShadowMaterial({ opacity: 0.2 }));
      floor.receiveShadow = true;
      scene.add(floor);
      const p = { scene, mesh, geom, floor, lights: { key: key1, fill, rim, sun, insun }, skel: null };
      if (this.seqs[key].J) p.skel = this.makeSkeleton(scene, L.nj, size, L.parents);
      this.panels[key] = p;
    }
    // floor under the ground truth's lowest point, sized to every output's extent
    for (const p of Object.values(this.panels)) {
      const b = this.box, cx = (b.lo[0] + b.hi[0]) / 2, cy = (b.lo[1] + b.hi[1]) / 2;
      p.floor.position.set(cx, cy, g.hdr.floor - 0.002 * b.size);
      p.floor.scale.set(b.size * 8, b.size * 8, 1);
      const s = p.lights.sun.shadow.camera, h = b.size * 1.2;
      s.left = -h; s.right = h; s.top = h; s.bottom = -h; s.near = 0.01; s.far = b.size * 12; s.updateProjectionMatrix();
    }
  }

  makeSkeleton(scene, nj, size, parents) {
    const mk = (geo, color) => {
      const m = new THREE.InstancedMesh(geo, new THREE.MeshBasicMaterial({ color, depthTest: false, depthWrite: false, transparent: true }), nj);
      m.renderOrder = 10; m.frustumCulled = false; m.visible = false; scene.add(m); return m;
    };
    const joints = mk(new THREE.SphereGeometry(1, 14, 10), 0xffffff), bones = mk(new THREE.CylinderGeometry(1, 1, 1, 10), 0x3b6be6);
    const red = new THREE.Color(0xe0312b), green = new THREE.Color(0x2ba44e);
    for (let j = 0; j < nj; j++) joints.setColorAt(j, parents[j] < 0 ? green : red);
    return { joints, bones, rj: 0.014 * size, rb: 0.0065 * size };
  }

  // positions at the viewing level, subdivided in place (a level's vertices are a prefix of the next's); sub = the
  // panel's own mesh's subdivision
  fill(out, P0, sub = this.gt.sub) {
    out.set(P0);
    let n = P0.length / 3;
    for (const pr of sub) for (let e = 0; e < pr.length; e += 2, n++) {
      const a = 3 * pr[e], b = 3 * pr[e + 1], o = 3 * n;
      out[o] = 0.5 * (out[a] + out[b]); out[o + 1] = 0.5 * (out[a + 1] + out[b + 1]); out[o + 2] = 0.5 * (out[a + 2] + out[b + 2]);
    }
  }

  // per-vertex values s through the error map's heat colours, blended over the pale colours
  errorColours(out, s, pale = this.gt.pale) {
    const k = (NS - 1) / ERRMAP.hi;
    for (let v = 0; v < s.length; v++) {
      const i = 4 * Math.min(NS - 1, Math.round(s[v] * k)), a = HEATE[i + 3];
      for (let c = 0; c < 3; c++) out[3 * v + c] = TOLIN[Math.round(255 * (pale[3 * v + c] * (1 - a) + HEATE[i + c] * a))];
    }
  }

  // the error map's values: each vertex's distance from the ground truth's same vertex (a two-mesh clip: from the truth
  // carried onto the output's mesh, setBase's `ref`), over the truth's size; the subdivided vertices take their edge's mean
  error(Pm, Pg) {
    const g = this.gt, nv = g.mesh.out.nv, nl = g.mesh.out.nl;
    const s = this._e && this._e.length === nl ? this._e : (this._e = new Float32Array(nl));
    for (let v = 0; v < nv; v++) {
      const a = 3 * v;
      s[v] = Math.hypot(Pm[a] - Pg[a], Pm[a + 1] - Pg[a + 1], Pm[a + 2] - Pg[a + 2]) / g.size;
    }
    let n = nv;
    for (const pr of g.sub) for (let e = 0; e < pr.length; e += 2, n++) s[n] = 0.5 * (s[pr[e]] + s[pr[e + 1]]);
    return s;
  }

  update() {
    if (!this.gt || !this.seqs) return;
    const g = this.gt, f = Math.min(this.frame, g.T - 1), Pg = g.ref[f];     // the error map's reference (setOut)
    const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), yAxis = new THREE.Vector3(0, 1, 0);
    const a = new THREE.Vector3(), b = new THREE.Vector3(), d = new THREE.Vector3(), sc = new THREE.Vector3();
    for (const key of this.keys) {
      const p = this.panels[key], s = this.seqs[key];
      if (!p || !s) continue;
      const L = this.look(key);
      const pos = p.geom.attributes.position;
      this.fill(pos.array, s.V[f], L.m.sub);
      pos.needsUpdate = true;
      this.normals(p, s, f);
      const col = p.geom.attributes.color;
      if (this.mode !== "err") col.array.set(L.m.col);
      else if (key === "gt" && g.two) {               // the ground truth on its own mesh: its zero, the pale colours
        const z = this._zero && this._zero.length === L.m.nl ? this._zero : (this._zero = new Float32Array(L.m.nl));
        this.errorColours(col.array, z, L.m.pale);
      } else this.errorColours(col.array, this.error(s.V[f], Pg), L.m.pale);
      col.needsUpdate = true;
      if (p.skel) {
        const on = this.skel && !!s.J;
        p.skel.joints.visible = p.skel.bones.visible = on;
        if (on) {
          const J = s.J, nj = L.nj, o = 3 * f * nj, par = L.parents;
          for (let j = 0; j < nj; j++) {
            a.set(J[o + 3 * j], J[o + 3 * j + 1], J[o + 3 * j + 2]);
            p.skel.joints.setMatrixAt(j, m4.compose(a, q.identity(), sc.setScalar(p.skel.rj * (par[j] < 0 ? 1.25 : 1))));
            const pj = par[j];
            if (pj < 0) { p.skel.bones.setMatrixAt(j, m4.makeScale(0, 0, 0)); continue; }
            b.set(J[o + 3 * pj], J[o + 3 * pj + 1], J[o + 3 * pj + 2]);
            d.subVectors(a, b);
            const len = d.length();
            q.setFromUnitVectors(yAxis, len > 1e-9 ? d.clone().divideScalar(len) : yAxis);
            p.skel.bones.setMatrixAt(j, m4.compose(a.clone().add(b).multiplyScalar(0.5), q, sc.set(p.skel.rb, len, p.skel.rb)));
          }
          p.skel.joints.instanceMatrix.needsUpdate = p.skel.bones.instanceMatrix.needsUpdate = true;
        }
      }
    }
  }

  // a panel's shading normals at frame f (2026-11-25): each face's normal (the cross product of two edges, so larger
  // faces weigh more) summed at its vertices and normalised, exactly as three.js's computeVertexNormals computes them
  // (bit for bit), in plain typed arrays (~3x faster); each frame's are kept, up to NORMALS_MB, so a looping clip
  // computes them once. On a slow computer these normals were most of the page's work while a clip played.
  normals(p, s, f) {
    const g = p.geom, P = g.attributes.position.array, I = g.index.array;
    let na = g.attributes.normal;
    if (!na) { na = new THREE.BufferAttribute(new Float32Array(P.length), 3); g.setAttribute("normal", na); }
    const N = na.array;
    let c = p.ncache;
    if (!c || c.V !== s.V || c.n !== N.length) c = p.ncache = { V: s.V, n: N.length, f: [], bytes: 0 };
    if (c.f[f]) { N.set(c.f[f]); na.needsUpdate = true; return; }
    N.fill(0);
    for (let i = 0; i < I.length; i += 3) {
      const a = 3 * I[i], b = 3 * I[i + 1], q = 3 * I[i + 2];
      const cbx = P[q] - P[b], cby = P[q + 1] - P[b + 1], cbz = P[q + 2] - P[b + 2];
      const abx = P[a] - P[b], aby = P[a + 1] - P[b + 1], abz = P[a + 2] - P[b + 2];
      const x = cby * abz - cbz * aby, y = cbz * abx - cbx * abz, z = cbx * aby - cby * abx;
      N[a] += x; N[a + 1] += y; N[a + 2] += z;
      N[b] += x; N[b + 1] += y; N[b + 2] += z;
      N[q] += x; N[q + 1] += y; N[q + 2] += z;
    }
    for (let i = 0; i < N.length; i += 3) {
      const x = N[i], y = N[i + 1], z = N[i + 2], k = 1 / (Math.sqrt(x * x + y * y + z * z) || 1);
      N[i] = x * k; N[i + 1] = y * k; N[i + 2] = z * k;
    }
    na.needsUpdate = true;
    const used = Object.values(this.panels).reduce((n, o) => n + (o.ncache ? o.ncache.bytes : 0), 0);
    if (used + N.byteLength <= NORMALS_MB * 1048576) { c.f[f] = N.slice(); c.bytes += N.byteLength; }
  }

  setFrame(f) { if (f === this.frame) return; this.frame = f; this.update(); this.render(); }
  setMode(m) { this.mode = m; this.update(); this.render(); }
  setSkeleton(on) { this.skel = on; this.update(); this.render(); }
  setCols(keys) { this.keys = keys.slice(); this.update(); this.render(); }
  setRows(n) { this.rows = n; this.render(); }

  setSpin(on) {
    this.spin = on;
    this.controls.autoRotate = on;
    this.controls.autoRotateSpeed = 2.4;
    const tick = () => { if (!this.spin) return; this.controls.update(); requestAnimationFrame(tick); };
    if (on) requestAnimationFrame(tick);
  }

  place(cam, { c, d, fwd }) {
    cam.position.copy(c).addScaledVector(fwd, -d);
    cam.near = Math.max(0.002, d / 200); cam.far = d * 20 + this.box.size * 10;
    cam.lookAt(c);
    cam.updateProjectionMatrix();
    return c;
  }

  // turn: 55 degrees from the input camera; input: the input camera's direction; top: from above; focus: opts.focus =
  // {target, radius, az, el} in the clip's world frame (az and el absolute, or daz / del relative to the input camera)
  preset(name) {
    if (!this.pts) return;
    let fit;
    if (name === "focus" && this.focus) {
      const fo = this.focus, az = fo.az !== undefined ? fo.az : this.camAz + (fo.daz || 0), el = fo.el !== undefined ? fo.el : this.camEl + (fo.del || 0);
      const { fwd } = camAxes(az, el), c = new THREE.Vector3(...fo.target);
      fit = { c, d: fo.radius * 2 * FX / (1 - 2 * 0.06) + fo.radius * 0.4, fwd };
    } else {
      const az = name === "input" ? this.camAz : this.camAz + 55;
      const el = name === "input" ? this.camEl : name === "top" ? 62 : Math.max(this.camEl, 18);
      fit = frameCamera(this.pts, az, el);
    }
    this.controls.target.copy(this.place(this.camera, fit));
    this.controls.update();
    this.render();
  }

  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.render();
  }

  placeLights(p, cam, target) {                      // bl_video.area_lights / aim_sun, relative to the camera's azimuth
    const v = cam.position.clone().sub(target), az = Math.atan2(v.y, v.x) * 180 / Math.PI;
    const r = Math.max(1, this.box.size * 4);
    const put = (L, daz, el) => {
      const a = (az + daz) * Math.PI / 180, e = el * Math.PI / 180;
      L.position.set(target.x + r * Math.cos(e) * Math.cos(a), target.y + r * Math.cos(e) * Math.sin(a), target.z + r * Math.sin(e));
      L.target.position.copy(target);
      L.target.updateMatrixWorld();
    };
    put(p.lights.key, -35, 45); put(p.lights.fill, 60, 15); put(p.lights.rim, 170, 35); put(p.lights.sun, -20, 64);
    const s = p.lights.insun;
    if (s) {                                         // the input video's sun at its place relative to this camera
      const fwd = target.clone().sub(cam.position).normalize();
      const right = fwd.clone().cross(new THREE.Vector3(0, 0, 1)).normalize(), up = right.clone().cross(fwd);
      const d = right.multiplyScalar(IN_SUN_REL[0]).addScaledVector(fwd, IN_SUN_REL[1]).addScaledVector(up, IN_SUN_REL[2]);
      s.position.copy(target).addScaledVector(d, r);
      s.target.position.copy(target);
      s.target.updateMatrixWorld();
    }
  }

  // the input video's world as an environment map: one uniform radiance (bl_video.py's look "input", Blender's world)
  uniformEnv() {
    if (!this.env) {
      const pm = new THREE.PMREMGenerator(this.renderer), s = new THREE.Scene();
      s.background = new THREE.Color().setRGB(IN_WORLD, IN_WORLD, IN_WORLD, THREE.LinearSRGBColorSpace);
      this.env = pm.fromScene(s).texture;
      pm.dispose();
    }
    return this.env;
  }

  render() {
    const n = this.keys.length, r = this.renderer;
    const W = this.canvas.clientWidth, H = this.canvas.clientHeight;
    if (!n || !W || !H || !this.box) return;
    const tw = W / n, th = H / this.rows;
    const topTarget = new THREE.Vector3();
    this.camTop.getWorldDirection(topTarget).multiplyScalar(this.camTop.position.distanceTo(this.controls.target)).add(this.camTop.position);
    for (let row = 0; row < this.rows; row++) {
      const cam = this.rows === 2 && row === 0 ? this.camTop : this.camera;
      const target = cam === this.camera ? this.controls.target : topTarget;
      cam.aspect = tw / th;
      cam.updateProjectionMatrix();
      const y = Math.round((this.rows - 1 - row) * th);
      for (let i = 0; i < n; i++) {
        const x = Math.round(i * tw), w = Math.round((i + 1) * tw) - x, p = this.panels[this.keys[i]];
        r.setViewport(x, y, w, Math.round(th)); r.setScissor(x, y, w, Math.round(th));
        r.setClearColor(BG, 1);
        if (!p || !this.seqs || !this.seqs[this.keys[i]]) { r.clear(); continue; }
        this.placeLights(p, cam, target);
        r.render(p.scene, cam);
      }
    }
  }
}
