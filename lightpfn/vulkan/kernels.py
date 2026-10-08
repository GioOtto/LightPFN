"""WGSL compute shaders of the Vulkan backend (compiled to SPIR-V by wgpu when a pipeline is created).

Every kernel reads its parameters from a u32 storage buffer P at offset 0. Tensors are float32 rows
addressed through views (see engine.View): row r = (z, i) with z = r / L, i = r % L lives at float offset
base + (z / zin) * so + (z % zin) * si + i * ss. Views let one kernel read a column of cells, a row of
cells, the first tokens of every row or a broadcast parameter without copies or transposes.

Precision: everything is float32. sin/cos use a Cody-Waite range reduction with minimax polynomials and
erf the Abramowitz-Stegun 7.1.26 formula (float32 error below 5e-7 on [-32,32]), so results do not depend on the precision
of the driver's transcendental functions.
"""

import re

COMMON = """
@group(0) @binding(0) var<storage, read> P: array<u32>;

struct View { base: u32, L: u32, zin: u32, so: u32, si: u32, ss: u32 }

fn view(o: u32) -> View { return View(P[o], P[o + 1u], P[o + 2u], P[o + 3u], P[o + 4u], P[o + 5u]); }
fn voff(v: View, z: u32, i: u32) -> u32 { return v.base + (z / v.zin) * v.so + (z % v.zin) * v.si + i * v.ss; }
fn roff(v: View, r: u32) -> u32 { return voff(v, r / v.L, r % v.L); }
fn pf(o: u32) -> f32 { return bitcast<f32>(P[o]); }

fn erf_(x: f32) -> f32 {
  let z = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * z);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-z * z);
  return select(-y, y, x >= 0.0);
}
fn gelu(x: f32) -> f32 { return 0.5 * x * (1.0 + erf_(x * 0.7071067811865476)); }
fn gelu4(v: vec4<f32>) -> vec4<f32> { return vec4<f32>(gelu(v.x), gelu(v.y), gelu(v.z), gelu(v.w)); }
fn tanh_(x: f32) -> f32 {
  let e = exp(2.0 * min(abs(x), 15.0));
  let t = 1.0 - 2.0 / (e + 1.0);
  return select(-t, t, x >= 0.0);
}
fn sincos(x: f32) -> vec2<f32> {
  // x = k * pi/2 + r with |r| <= pi/4 (pi/2 split in three parts, Cody-Waite), cephes sinf/cosf polynomials
  let k = round(x * 0.6366197723675814);
  var r = x - k * 1.5703125;
  r = r - k * 4.837512969970703125e-4;
  r = r - k * 7.54978995489188216e-8;
  let r2 = r * r;
  let s = r + r * r2 * (-1.6666654611e-1 + r2 * (8.3321608736e-3 + r2 * -1.9515295891e-4));
  let c = 1.0 - 0.5 * r2 + r2 * r2 * (4.166664568298827e-2 + r2 * (-1.388731625493765e-3 + r2 * 2.443315711809948e-5));
  let q = i32(k - 4.0 * floor(k * 0.25));
  if (q == 0) { return vec2<f32>(s, c); }
  if (q == 1) { return vec2<f32>(c, -s); }
  if (q == 2) { return vec2<f32>(-s, -c); }
  return vec2<f32>(-c, s);
}
// P[1]: workgroups along x (2D grids past 65535); P[63]: first workgroup of this slice of the dispatch
fn wgid(wg: vec3u) -> u32 { return P[63] + wg.x + wg.y * P[1]; }
"""

# Y[r, :O] = epilogue((X[r, :K] @ W^T)), W (O, K) row-major. 64 x 64 output tiles, 256 threads with 4 x 4
# outputs each. Options (compile time): NORM scales row r by 1 / rms(X[r]) (a folded RMSNorm), BIAS adds
# b, GELU applies gelu, RES = "self" adds the old Y, "r" adds rows of a separate view R.
# P: [0] n_tiles, [1] grid x, [2] N, [3] K, [4] O, [5] eps, [6..11] X view, [12..17] Y view, [18..23] R view
GEMM = """
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> W: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Y: array<vec4<f32>>;
#if BIAS
@group(0) @binding(4) var<storage, read> Bv: array<vec4<f32>>;
#endif
#if RES_R
@group(0) @binding(5) var<storage, read> R: array<vec4<f32>>;
#endif
var<workgroup> xs: array<vec4<f32>, 256>;  // [k 16][row 64 / 4]
var<workgroup> ws: array<vec4<f32>, 256>;  // [k 16][col 64 / 4]
var<workgroup> xo: array<u32, 64>;
var<workgroup> yo: array<u32, 64>;
var<workgroup> ro: array<u32, 64>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(local_invocation_index) li: u32) {
  let id = wgid(wg);
  if (id >= P[0]) { return; }
  let N = P[2]; let K = P[3]; let O = P[4];
  let tiles_c = (O + 63u) / 64u;
  let row0 = (id / tiles_c) * 64u;
  let col0 = (id % tiles_c) * 64u;
  if (li < 64u) {
    let r = min(row0 + li, N - 1u);
    xo[li] = roff(view(6u), r) / 4u;
    yo[li] = roff(view(12u), r) / 4u;
    ro[li] = roff(view(18u), r) / 4u;
  }
  workgroupBarrier();
  var acc: array<vec4<f32>, 4>;
  var ssq = vec4<f32>(0.0);
  let lr = li / 4u;  // row (and column) loaded by this thread
  let lk = li % 4u;  // vec4 of k loaded by this thread
  let K4 = K / 4u;
  for (var k4 = 0u; k4 < K4; k4 += 4u) {
    var xv = vec4<f32>(0.0);
    if (row0 + lr < N && k4 + lk < K4) { xv = X[xo[lr] + k4 + lk]; }
    var wv = vec4<f32>(0.0);
    if (col0 + lr < O && k4 + lk < K4) { wv = W[(col0 + lr) * K4 + k4 + lk]; }
    for (var c = 0u; c < 4u; c++) {
      xs[(lk * 4u + c) * 16u + lr / 4u][lr % 4u] = xv[c];
      ws[(lk * 4u + c) * 16u + lr / 4u][lr % 4u] = wv[c];
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk++) {
      let a = xs[kk * 16u + lid.y];
      let b = ws[kk * 16u + lid.x];
      acc[0] += a.x * b;
      acc[1] += a.y * b;
      acc[2] += a.z * b;
      acc[3] += a.w * b;
#if NORM
      ssq += a * a;
#endif
    }
    workgroupBarrier();
  }
  let c4 = col0 / 4u + lid.x;
  if (col0 + lid.x * 4u >= O) { return; }
  for (var i = 0u; i < 4u; i++) {
    let rl = lid.y * 4u + i;
    if (row0 + rl >= N) { continue; }
    var v = acc[i];
#if NORM
    v *= inverseSqrt(ssq[i] / f32(K) + pf(5u));
#endif
#if BIAS
    v += Bv[c4];
#endif
#if GELU
    v = gelu4(v);
#endif
#if RES_SELF
    v += Y[yo[rl] + c4];
#endif
#if RES_R
    v += R[ro[rl] + c4];
#endif
    Y[yo[rl] + c4] = v;
  }
}
"""

# softmax(q k^T / sqrt(D)) v for every (z, head, query), float32 online softmax (flash attention).
# Tiled: a workgroup holds 64 queries of one (z, head) and walks the keys in shared-memory tiles of KT.
# P: [0] n_groups, [1] grid x, [2] Lq, [3] Lk, [4] H, [5] scale, [6..11] q, [12..17] k, [18..23] v,
#    [24..29] o views, [30] q_h, [31] k_h, [32] v_h, [33] o_h head strides (floats)
ATTN_TILED = """
const D4: u32 = {D4}u;
const KT: u32 = {KT}u;
@group(0) @binding(1) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> K: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> V: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> O: array<vec4<f32>>;
var<workgroup> ks: array<vec4<f32>, {KT} * {D4}>;
var<workgroup> vs: array<vec4<f32>, {KT} * {D4}>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let id = wgid(wg);
  if (id >= P[0]) { return; }
  let Lq = P[2]; let Lk = P[3]; let H = P[4];
  let tiles = (Lq + 63u) / 64u;
  let zh = id / tiles;
  let z = zh / H;
  let h = zh % H;
  let qi = (id % tiles) * 64u + li;
  let qv = view(6u); let kv = view(12u); let vv = view(18u); let ov = view(24u);
  let qb = (voff(qv, z, min(qi, Lq - 1u)) + h * P[30]) / 4u;
  let scale = pf(5u);
  var q: array<vec4<f32>, {D4}>;
  var acc: array<vec4<f32>, {D4}>;
  for (var d = 0u; d < D4; d++) { q[d] = Q[qb + d] * scale; acc[d] = vec4<f32>(0.0); }
  var m = -3.0e38;
  var l = 0.0;
  for (var t0 = 0u; t0 < Lk; t0 += KT) {
    for (var e = li; e < KT * D4; e += 64u) {
      let key = min(t0 + e / D4, Lk - 1u);
      ks[e] = K[(voff(kv, z, key) + h * P[31]) / 4u + e % D4];
      vs[e] = V[(voff(vv, z, key) + h * P[32]) / 4u + e % D4];
    }
    workgroupBarrier();
    let nk = min(KT, Lk - t0);
    var s: array<f32, {KT}>;
    var mt = m;
    for (var j = 0u; j < KT; j++) {
      var dot4 = vec4<f32>(0.0);
      for (var d = 0u; d < D4; d++) { dot4 += q[d] * ks[j * D4 + d]; }
      let sj = select(-3.0e38, dot4.x + dot4.y + dot4.z + dot4.w, j < nk);
      s[j] = sj;
      mt = max(mt, sj);
    }
    let corr = exp(m - mt);
    l *= corr;
    for (var d = 0u; d < D4; d++) { acc[d] *= corr; }
    for (var j = 0u; j < KT; j++) {
      let p = select(0.0, exp(s[j] - mt), j < nk);
      l += p;
      for (var d = 0u; d < D4; d++) { acc[d] += p * vs[j * D4 + d]; }
    }
    m = mt;
    workgroupBarrier();
  }
  if (qi < Lq) {
    let ob = (voff(ov, z, qi) + h * P[33]) / 4u;
    for (var d = 0u; d < D4; d++) { O[ob + d] = acc[d] / l; }
  }
}
"""

# Same function, one thread per (z, head, query) reading keys from global memory: for few queries per
# sequence (summary tokens) or few keys. Same parameter layout as ATTN_TILED, [0] = Z * H * Lq.
ATTN_SMALL = """
const D4: u32 = {D4}u;
@group(0) @binding(1) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> K: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> V: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> O: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let id = wgid(wg) * 64u + li;
  if (id >= P[0]) { return; }
  let Lq = P[2]; let Lk = P[3]; let H = P[4];
  let qi = id % Lq;
  let zh = id / Lq;
  let z = zh / H;
  let h = zh % H;
  let qv = view(6u); let kv = view(12u); let vv = view(18u); let ov = view(24u);
  let qb = (voff(qv, z, qi) + h * P[30]) / 4u;
  let scale = pf(5u);
  var q: array<vec4<f32>, {D4}>;
  var acc: array<vec4<f32>, {D4}>;
  for (var d = 0u; d < D4; d++) { q[d] = Q[qb + d] * scale; acc[d] = vec4<f32>(0.0); }
  var m = -3.0e38;
  var l = 0.0;
  for (var j = 0u; j < Lk; j++) {
    let kb = (voff(kv, z, j) + h * P[31]) / 4u;
    var dot4 = vec4<f32>(0.0);
    for (var d = 0u; d < D4; d++) { dot4 += q[d] * K[kb + d]; }
    let s = dot4.x + dot4.y + dot4.z + dot4.w;
    let mt = max(m, s);
    let corr = exp(m - mt);
    let p = exp(s - mt);
    l = l * corr + p;
    let vb = (voff(vv, z, j) + h * P[32]) / 4u;
    for (var d = 0u; d < D4; d++) { acc[d] = acc[d] * corr + p * V[vb + d]; }
    m = mt;
  }
  let ob = (voff(ov, z, qi) + h * P[33]) / 4u;
  for (var d = 0u; d < D4; d++) { O[ob + d] = acc[d] / l; }
}
"""

# dst row r = src row r (W4 vec4 per row). P: [0] rows * W4, [1] grid x, [2] W4, [6..11] src, [12..17] dst
COPY = """
@group(0) @binding(1) var<storage, read> S: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> Dst: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let id = wgid(wg) * 64u + li;
  if (id >= P[0]) { return; }
  let r = id / P[2];
  let c = id % P[2];
  Dst[roff(view(12u), r) / 4u + c] = S[roff(view(6u), r) / 4u + c];
}
"""

# In place on rows of W4 vec4: optional LayerNorm (weight, bias, eps), then optional + emb[slot[z]], where
# slot index = r / P[4] (the training row of a cell).
# P: [0] rows, [1] grid x, [2] W4, [3] eps, [4] rows per slot, [6..11] view
ROWNORM = """
const W4: u32 = {W4}u;
@group(0) @binding(1) var<storage, read_write> X: array<vec4<f32>>;
#if LN
@group(0) @binding(2) var<storage, read> Lw: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> Lb: array<vec4<f32>>;
#endif
#if EMB
@group(0) @binding(4) var<storage, read> Emb: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> Slot: array<u32>;
#endif

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let r = wgid(wg) * 64u + li;
  if (r >= P[0]) { return; }
  let v = view(6u);
  let b = roff(v, r) / 4u;
#if LN
  var x: array<vec4<f32>, {W4}>;
  var s = vec4<f32>(0.0);
  for (var c = 0u; c < W4; c++) { x[c] = X[b + c]; s += x[c]; }
  let mean = (s.x + s.y + s.z + s.w) / f32(W4 * 4u);
  var q = vec4<f32>(0.0);
  for (var c = 0u; c < W4; c++) { let d = x[c] - mean; q += d * d; }
  let rstd = inverseSqrt((q.x + q.y + q.z + q.w) / f32(W4 * 4u) + pf(3u));
  for (var c = 0u; c < W4; c++) { x[c] = (x[c] - mean) * rstd * Lw[c] + Lb[c]; }
#if EMB
  let e = Slot[r / P[4]] * W4;
  for (var c = 0u; c < W4; c++) { x[c] += Emb[e + c]; }
#endif
  for (var c = 0u; c < W4; c++) { X[b + c] = x[c]; }
#else
  let e = Slot[r / P[4]] * W4;
  for (var c = 0u; c < W4; c++) { X[b + c] += Emb[e + c]; }
#endif
}
"""

# Cell features for the embedding GEMM, one thread per cell of a block of the (B, n, m) table: cells c in
# (b, ii, jj) order over B x nr rows (from row i0) x mc columns (from column j0). Inputs z, r, nan are the
# normalized values of the whole table; neighbors are the group offsets (j + o) % m.
# Output row c, KP floats: fourier [sum_g sin(z_g f), sum_g cos(z_g f)] (2F) or rbf kernels (64), then
# [z_g, nan_g, sin(r_g pi 2^e), cos(r_g pi 2^e)], zero padded.
# P: [0] cells, [1] grid x, [2] n, [3] m, [4] i0, [5] nr, [6] j0, [7] mc
FEATURES = """
const G: u32 = {G}u;
const NF: u32 = {NF}u;
const NE: u32 = {NE}u;
const KP: u32 = {KP}u;
const OFFS = array<u32, {G}>({OFFS});
@group(0) @binding(1) var<storage, read> Zt: array<f32>;
@group(0) @binding(2) var<storage, read> Rt: array<f32>;
@group(0) @binding(3) var<storage, read> Nt: array<f32>;
@group(0) @binding(4) var<storage, read> Fq: array<f32>;
@group(0) @binding(5) var<storage, read_write> Out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let c = wgid(wg) * 64u + li;
  if (c >= P[0]) { return; }
  let n = P[2]; let m = P[3]; let nr = P[5]; let mc = P[7];
  let b = c / (nr * mc);
  let rem = c % (nr * mc);
  let i = P[4] + rem / mc;
  let j = P[6] + rem % mc;
  let rowbase = (b * n + i) * m;
  var z: array<f32, {G}>;
  var r: array<f32, {G}>;
  var nn: array<f32, {G}>;
  for (var g = 0u; g < G; g++) {
    let jg = (j + OFFS[g]) % m;
    z[g] = Zt[rowbase + jg];
    r[g] = Rt[rowbase + jg];
    nn[g] = Nt[rowbase + jg];
  }
  let o = c * KP;
  var k = 0u;
#if FOURIER
  for (var f = 0u; f < NF; f++) {
    var s = 0.0;
    var co = 0.0;
    for (var g = 0u; g < G; g++) {
      let sc = sincos(z[g] * Fq[g * NF + f]);
      s += sc.x;
      co += sc.y;
    }
    Out[o + f] = s;
    Out[o + NF + f] = co;
  }
  k = 2u * NF;
#else
  for (var t = 0u; t < 64u; t++) {
    var s = 0.0;
    for (var g = 0u; g < G; g++) { let d = z[g] - Fq[t]; s += exp(-0.5 * d * d); }
    Out[o + t] = s;
  }
  k = 64u;
#endif
  for (var g = 0u; g < G; g++) { Out[o + k + g] = z[g]; Out[o + k + G + g] = nn[g]; }
  k += 2u * G;
  for (var g = 0u; g < G; g++) {
    for (var e = 0u; e < NE; e++) {
      let sc = sincos(r[g] * 3.141592653589793 * f32(1u << e));
      Out[o + k + g * NE + e] = sc.x;
      Out[o + k + G * NE + g * NE + e] = sc.y;
    }
  }
  k += 2u * G * NE;
  for (; k < KP; k++) { Out[o + k] = 0.0; }
}
"""

# Rotary embedding in place on adjacent pairs (2p, 2p + 1) of each head (the layout of LightPFN.folded()),
# for tokens i >= p0 of each sequence at position i - p0; cos/sin from a table T (pos, D / 2, 2).
# P: [0] rows * H * D4, [1] grid x, [2] H, [3] p0, [4] head stride, [6..11] view
ROPE = """
const D4: u32 = {D4}u;
@group(0) @binding(1) var<storage, read_write> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> T: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let id = wgid(wg) * 64u + li;
  if (id >= P[0]) { return; }
  let v = view(6u);
  let per_row = P[2] * D4;
  let r = id / per_row;
  let i = r % v.L;
  if (i < P[3]) { return; }
  let h = (id % per_row) / D4;
  let d = id % D4;
  let a = (roff(v, r) + h * P[4]) / 4u + d;
  let x = X[a];
  let cs = T[(i - P[3]) * D4 + d];  // (cos, sin) of pairs 2d and 2d + 1
  X[a] = vec4<f32>(x.x * cs.x - x.y * cs.y, x.x * cs.y + x.y * cs.x,
                   x.z * cs.z - x.w * cs.w, x.z * cs.w + x.w * cs.z);
}
"""

# Softmax scaling of attention queries in place: q[k] *= base[k % HD] * (1 + tanh(mod[k])) (contiguous).
# P: [0] total vec4, [1] grid x, [2] HD / 4
QSCALE = """
@group(0) @binding(1) var<storage, read_write> Q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> Md: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> Bs: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let id = wgid(wg) * 64u + li;
  if (id >= P[0]) { return; }
  let mv = Md[id];
  let t = vec4<f32>(tanh_(mv.x), tanh_(mv.y), tanh_(mv.z), tanh_(mv.w));
  Q[id] = Q[id] * Bs[id % P[2]] * (1.0 + t);
}
"""


def render(src, flags=(), **consts):
    """Source with `#if NAME` / `#else` / `#endif` blocks resolved (NAME in flags) and {KEY} replaced."""
    out, stack = [], []
    for line in (COMMON + src).splitlines():
        s = line.strip()
        if s.startswith("#if "):
            stack.append(s[4:].strip() in flags)
        elif s == "#else":
            if not stack:
                raise ValueError("unmatched #else in shader template")
            stack[-1] = not stack[-1]
        elif s == "#endif":
            if not stack:
                raise ValueError("unmatched #endif in shader template")
            stack.pop()
        elif all(stack):
            out.append(line)
    if stack:
        raise ValueError("unclosed #if in shader template")
    code = "\n".join(out)
    for k, v in consts.items():
        code = code.replace("{" + k + "}", str(v))
    if re.search(r"\{[A-Z][A-Z0-9_]*\}", code):
        raise ValueError("missing shader template constant")
    return code
