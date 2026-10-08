"""LightPFN inference on a Vulkan GPU: the computation of LightPFN.folded().encode / predict_logits with
WGSL kernels. Column statistics and value normalization (sort, searchsorted) stay on the host in torch;
everything after the normalized values runs on the GPU, and the training-set context (column caches, ICL
key/value cache, decoder keys) stays in GPU memory between fit and predict.
"""

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch

from lightpfn.model.layers import Block
from lightpfn.model.lightpfn import CellEmbedder
from lightpfn.vulkan import kernels as K
from lightpfn.vulkan.engine import Engine, View, f32bits, rows

GPU_CHUNK_CELLS = 1 << 20


def _eps(norm):
    return torch.finfo(torch.float32).eps if norm.eps is None else norm.eps


def _pad4(n):
    return (n + 3) // 4 * 4


@dataclass
class VulkanContext:
    """What prediction needs from the training rows; the caches are GPU buffers."""

    stats: dict
    col_kv: list
    col_refine_kv: list | None
    icl_kv: list
    dec_k: object
    onehot: object
    B: int
    n: int
    m: int
    n_classes: int


class _Block:
    """GPU weights of a folded Block (RMSNorm weights already inside the linear layers)."""

    def __init__(self, eng, blk):
        a = blk.attn
        self.H, self.hd = a.n_heads, a.head_dim
        self.E = a.q.weight.shape[1]
        self.eps = (_eps(blk.norm_q), _eps(blk.norm_kv), _eps(blk.norm_ff))
        w = lambda t: eng.upload(t.detach().float().cpu().numpy())  # noqa: E731
        self.wq, self.wkv, self.wo = w(a.q.weight), w(a.kv.weight), w(a.out.weight)
        hid = blk.mlp.fc1.weight.shape[0]
        self.hid = _pad4(hid)  # zero rows/columns: gelu(0) = 0 adds nothing
        w1 = torch.zeros(self.hid, self.E)
        w1[:hid] = blk.mlp.fc1.weight
        w2 = torch.zeros(self.E, self.hid)
        w2[:, :hid] = blk.mlp.fc2.weight
        self.w1, self.w2 = w(w1), w(w2)
        self.scaling = None
        if a.scaling is not None:
            self.scaling = _Scaling(eng, a.scaling)


class _Scaling:
    """SoftmaxScaling: base(log n) on the host (a vector per key count), mod(q) on the GPU."""

    def __init__(self, eng, sc):
        self.base_module = copy.deepcopy(sc.base).float().cpu().eval()
        self.H, self.hd = sc.n_heads, sc.head_dim
        w = lambda t: eng.upload(t.detach().float().cpu().numpy())  # noqa: E731
        lin1, lin2 = sc.mod[0], sc.mod[2]
        self.hidden = lin1.weight.shape[0]
        assert self.hidden % 4 == 0
        self.w1, self.b1, self.w2, self.b2 = w(lin1.weight), w(lin1.bias), w(lin2.weight), w(lin2.bias)
        self.eng, self.bases = eng, {}

    def base(self, n_keys):
        buf = self.bases.get(n_keys)
        if buf is None:
            with torch.no_grad():
                b = self.base_module(torch.full((1, 1), math.log(max(n_keys, 2)))).view(-1)
            buf = self.bases[n_keys] = self.eng.upload(b.numpy())
        return buf


class _ColumnStage:
    def __init__(self, eng, st):
        w = lambda t: eng.upload(t.detach().float().cpu().numpy())  # noqa: E731
        self.y_emb = w(st.y_emb.weight)
        self.ind = [w(p) for p in st.inducing]
        self.ind_blocks = [_Block(eng, b) for b in st.ind_blocks]
        self.cell_blocks = [_Block(eng, b) for b in st.cell_blocks]
        with torch.no_grad():  # the inducing queries do not depend on the data
            self.ind_q = [w(b.attn.q(b.norm_q(p))) for b, p in zip(st.ind_blocks, st.inducing)]
        self.n_ind = st.inducing[0].shape[0]


class VulkanLightPFN:
    """LightPFN on a Vulkan GPU with the encode / predict_logits interface of LightPFN (inputs and logits
    are CPU torch tensors). model: a LightPFN, folded or not. adapter: see engine.pick_adapter."""

    def __init__(self, model, adapter=None):
        cfg = model.cfg
        for dim, heads in ((cfg.col_dim, cfg.col_heads), (cfg.col_dim, cfg.row_heads),
                           (cfg.icl_dim, cfg.icl_heads), (cfg.icl_dim, cfg.decoder_heads)):
            if heads <= 0 or dim <= 0 or dim % (4 * heads):
                raise NotImplementedError("Vulkan attention head dimensions must be positive multiples of four")
        if not 0 <= cfg.n_ecdf_freq < 32:
            raise NotImplementedError("Vulkan n_ecdf_freq must be between 0 and 31")
        if cfg.n_inducing <= 0:
            raise NotImplementedError("Vulkan inducing token count must be positive")
        m = model.folded()
        self.cfg = cfg = m.cfg
        self.eng = eng = Engine(adapter)
        self.E, self.D, self.C = cfg.col_dim, cfg.icl_dim, cfg.n_cls
        if cfg.icl_kv_heads_test not in (1, cfg.icl_heads):
            raise NotImplementedError("icl_kv_heads_test must be 1 or icl_heads")
        w = lambda t: eng.upload(t.detach().float().cpu().numpy())  # noqa: E731
        ce = m.cells
        self.offsets, self.n_ecdf = tuple(ce.offsets), ce.n_ecdf_freq
        G = len(self.offsets)
        meta = G * (2 + 2 * self.n_ecdf)
        if cfg.cell_embed == "fourier":
            self.fourier, self.n_freq = True, ce.freq.shape[1]
            value_w, self.fq = ce.fourier.weight, w(ce.freq)
        else:
            self.fourier, self.n_freq = False, 64
            value_w, self.fq = ce.rbf.weight, w(ce.rbf_centers)
        k = value_w.shape[1] + meta
        self.kp = _pad4(k)
        wc = torch.zeros(self.E, self.kp)
        wc[:, :k] = torch.cat([value_w, ce.meta.weight], 1)
        self.w_cells, self.ln_w, self.ln_b, self.ln_eps = w(wc), w(ce.norm.weight), w(ce.norm.bias), ce.norm.eps
        self.col = _ColumnStage(eng, m.col)
        self.row_mode, self.rope_base = m.row.mode, m.row.rope_base
        self.cls = w(m.row.cls)
        self.row_blocks = [_Block(eng, b) for b in m.row.blocks]
        self.refine = None
        if cfg.row_refine:
            self.refine = dict(summary=w(m.refine.summary), n_summary=m.refine.summary.shape[0],
                               broadcast=[_Block(eng, b) for b in m.refine.broadcast],
                               gather=[_Block(eng, b) for b in m.refine.gather])
            self.col_refine = _ColumnStage(eng, m.col_refine)
        icl = m.icl
        self.icl_y_emb, self.thinking = w(icl.y_emb.weight), w(icl.thinking)
        self.n_thinking = icl.thinking.shape[0]
        self.icl_blocks = [_Block(eng, b) for b in icl.blocks]
        self.kv_heads = icl.kv_heads_test
        dec = m.decoder
        self.dec_H, self.dec_hd = dec.n_heads, dec.head_dim
        self.dec_q, self.dec_k = w(dec.q.weight), w(dec.k.weight)
        self.dec_eps = _eps(icl.norm)
        self.dec_scaling = _Scaling(eng, dec.scaling)
        self.rope_tables = {}
        # Cell limits are upper bounds; train_capacity and the piece costs also bound caches and scratch.
        self.max_cells = eng.max_binding // (4 * 2 * self.E)
        self.max_train_cells = eng.max_binding // (4 * self.E)

    def _piece_costs(self, n, m):
        """Floats per estimator in one column / one row piece."""
        stages = [self.col] + ([self.col_refine] if self.refine else [])
        col_width = max(self.kp, 2 * self.E,
                        *(b.hid for s in stages for b in s.ind_blocks + s.cell_blocks))
        row_blocks = self.row_blocks + (self.refine["broadcast"] + self.refine["gather"] if self.refine else [])
        row_width = max(2 * self.E, *(b.hid for b in row_blocks))
        S = self.refine["n_summary"] if self.refine else 0
        return max(n, self.col.n_ind) * col_width, max(max(m + self.C, S) * row_width, m * self.kp)

    def _icl_width(self):
        return max(2 * self.D, self.dec_H * self.dec_scaling.hidden,
                   *(b.hid for b in self.icl_blocks),
                   *(b.H * b.scaling.hidden for b in self.icl_blocks))

    def train_capacity(self, n, m):
        """Estimators whose persistent buffers and smallest pieces fit the binding limit."""
        col_piece, row_piece = self._piece_costs(n, m)
        largest = max(n * m * self.E, m * self.col.n_ind * 2 * self.E,
                      (n + self.n_thinking) * self._icl_width(), col_piece, row_piece)
        return min(self.max_train_cells // max(1, n * m), self.eng.max_binding // (4 * max(1, largest)))

    # kernels -------------------------------------------------------------------------------------
    def gemm(self, x, y, w, N, Kd, O, norm_eps=None, bias=None, gelu=False, res=None):
        """y rows = epilogue(x rows @ w^T) for N rows; res: None, "self" or a View added to the result."""
        flags = []
        if norm_eps is not None:
            flags.append("NORM")
        if bias is not None:
            flags.append("BIAS")
        if gelu:
            flags.append("GELU")
        if res == "self":
            flags.append("RES_SELF")
        elif res is not None:
            flags.append("RES_R")
        assert Kd % 4 == 0 and O % 4 == 0, (Kd, O)
        tiles = math.ceil(N / 64) * math.ceil(O / 64)
        rv = res if isinstance(res, View) else y
        params = [tiles, 0, N, Kd, O, f32bits(norm_eps or 0.0)] + x.params() + y.params() + rv.params()
        bufs = {1: x.buf, 2: w, 3: y.buf}
        if bias is not None:
            bufs[4] = bias
        if isinstance(res, View):
            bufs[5] = res.buf
        self.eng.dispatch(self.eng.pipeline("gemm", K.GEMM, flags), bufs, params, tiles, 2.0 * N * Kd * O)

    def attention(self, q, k, v, o, Z, H, Lq, Lk, hd, q_h, k_h, v_h, o_h):
        if Z == 0 or Lq == 0:
            return
        if Lk == 0:
            width = H * hd
            zero = self.eng.upload(np.zeros((Z * Lq, width), np.float32))
            self.copy(rows(zero, width), o, Z * Lq, width)
            return
        small = Lq < 32 or Lk < 16
        D4 = hd // 4
        if small:
            pipe = self.eng.pipeline("attn_small", K.ATTN_SMALL, D4=D4)
            total = Z * H * Lq
            groups = math.ceil(total / 64)
        else:
            kt = min(64, 16384 // (2 * hd * 4))
            pipe = self.eng.pipeline("attn_tiled", K.ATTN_TILED, D4=D4, KT=kt)
            total = math.ceil(Lq / 64) * Z * H
            groups = total
        params = [total, 0, Lq, Lk, H, f32bits(hd ** -0.5)] + q.params() + k.params() + v.params() + o.params()
        params += [q_h, k_h, v_h, o_h]
        self.eng.dispatch(pipe, {1: q.buf, 2: k.buf, 3: v.buf, 4: o.buf}, params, groups, 4.0 * Z * H * Lq * Lk * hd)

    def copy(self, src, dst, N, width):
        W4 = width // 4
        params = [N * W4, 0, W4, 0, 0, 0] + src.params() + dst.params()
        self.eng.dispatch(self.eng.pipeline("copy", K.COPY), {1: src.buf, 2: dst.buf}, params, math.ceil(N * W4 / 64))

    def rownorm(self, x, N, width, ln=None, emb=None, slots=None, slot_div=1):
        """In place on N rows: LayerNorm (ln = (w, b, eps)) then + emb[slots[r // slot_div]]."""
        flags = (["LN"] if ln else []) + (["EMB"] if emb is not None else [])
        params = [N, 0, width // 4, f32bits(ln[2] if ln else 0.0), slot_div, 0] + x.params()
        bufs = {1: x.buf}
        if ln:
            bufs[2], bufs[3] = ln[0], ln[1]
        if emb is not None:
            bufs[4], bufs[5] = emb, slots
        self.eng.dispatch(self.eng.pipeline("rownorm", K.ROWNORM, flags, W4=width // 4), bufs, params, math.ceil(N / 64))

    def rope(self, x, N, H, hd, head_stride, p0, L):
        tab = self.rope_table(L, hd)
        D4 = hd // 4
        params = [N * H * D4, 0, H, p0, head_stride, 0] + x.params()
        self.eng.dispatch(self.eng.pipeline("rope", K.ROPE, D4=D4), {1: x.buf, 2: tab}, params, math.ceil(N * H * D4 / 64))

    def rope_table(self, L, hd):
        """cos/sin of positions 0..L-1 for the adjacent pairs, computed as torch's rope() does."""
        key = (hd, L)
        tab = self.rope_tables.get(key)
        if tab is None:
            for (d, n), t in self.rope_tables.items():  # a longer table of the same head dim serves too
                if d == hd and n >= L:
                    return t
            inv_freq = self.rope_base ** (-torch.arange(0, hd, 2, dtype=torch.float32) / hd)
            ang = torch.arange(max(L, 1), dtype=torch.float32)[:, None] * inv_freq[None]
            tab = self.rope_tables[key] = self.eng.upload(torch.stack([ang.cos(), ang.sin()], -1).numpy())
        return tab

    def qscale(self, q, N, sc, n_keys):
        """Queries q (N rows of H * hd, contiguous) times base(log n_keys) * (1 + tanh(mod(q)))."""
        rows_h = N * sc.H
        mh = self.eng.temp("mod_h", rows_h * sc.hidden)
        md = self.eng.temp("mod_o", rows_h * sc.hd)
        self.gemm(rows(q, sc.hd), rows(mh, sc.hidden), sc.w1, rows_h, sc.hd, sc.hidden, bias=sc.b1, gelu=True)
        self.gemm(rows(mh, sc.hidden), rows(md, sc.hd), sc.w2, rows_h, sc.hidden, sc.hd, bias=sc.b2)
        total = rows_h * sc.hd // 4
        params = [total, 0, sc.H * sc.hd // 4]
        self.eng.dispatch(self.eng.pipeline("qscale", K.QSCALE), {1: q, 2: md, 3: sc.base(n_keys)}, params,
                          math.ceil(total / 64))

    def mlp(self, blk, x, N):
        hid = self.eng.temp("hid", N * blk.hid)
        self.gemm(x, rows(hid, blk.hid), blk.w1, N, blk.E, blk.hid, norm_eps=blk.eps[2], gelu=True)
        self.gemm(rows(hid, blk.hid), x, blk.w2, N, blk.hid, blk.E, res="self")

    # stages --------------------------------------------------------------------------------------
    def embed(self, zrn, x, B, n, m, i0, nr, j0, mc):
        """Cell embeddings of rows i0..i0+nr, columns j0..j0+mc of the (B, n, m) table into view x."""
        N = B * nr * mc
        feat = self.eng.temp("feat", N * self.kp)
        G = len(self.offsets)
        consts = dict(G=G, NF=self.n_freq, NE=self.n_ecdf, KP=self.kp,
                      OFFS=", ".join(f"{o % m}u" for o in self.offsets))
        pipe = self.eng.pipeline("features", K.FEATURES, ["FOURIER"] if self.fourier else [], **consts)
        params = [N, 0, n, m, i0, nr, j0, mc]
        self.eng.dispatch(pipe, {1: zrn[0], 2: zrn[1], 3: zrn[2], 4: self.fq, 5: feat}, params, math.ceil(N / 64))
        self.gemm(rows(feat, self.kp), x, self.w_cells, N, self.kp, self.E)

    def column_context(self, st, main, B, n, m, cols, slots, zrn=None):
        """ColumnStage.context on pieces of `cols` columns of the (B, n, m, E) buffer `main` (embedded
        first when zrn is given); returns the (B * m, n_ind, 2E) key/value caches of the cell blocks."""
        E, I = self.E, st.n_ind
        caches = [self.eng.empty(B * m * I * 2 * E) for _ in st.cell_blocks]
        for j0 in range(0, m, cols):
            mc = min(cols, m - j0)
            N = B * n * mc
            x = View(main, j0 * E, L=mc, zin=1, so=m * E, si=0, ss=E)  # rows (b * n + i, jj)
            ln = None
            if zrn is not None:
                self.embed(zrn, x, B, n, m, 0, n, j0, mc)
                ln = (self.ln_w, self.ln_b, self.ln_eps)
            self.rownorm(x, N, E, ln=ln, emb=st.y_emb, slots=slots, slot_div=mc)
            for ind, ind_q, ib, cb, cache in zip(st.ind, st.ind_q, st.ind_blocks, st.cell_blocks, caches):
                H, hd = ib.H, ib.hd
                # inducing points read the cells of their column: Z = B * mc sequences of n keys
                kv = self.eng.temp("kv", N * 2 * E)
                self.gemm(x, rows(kv, 2 * E), ib.wkv, N, E, 2 * E, norm_eps=ib.eps[1])
                kview = lambda base: View(kv, base, L=n, zin=mc, so=n * mc * 2 * E, si=2 * E, ss=mc * 2 * E)  # noqa: E731
                o = self.eng.temp("o_ind", B * mc * I * E)
                self.attention(View(ind_q, 0, L=I, zin=1, so=0, si=0, ss=E), kview(0), kview(E),
                               View(o, 0, L=I, zin=1, so=I * E, ss=E), B * mc, H, I, n, hd, hd, hd, hd, hd)
                h = self.eng.temp("h_ind", B * mc * I * E)
                self.gemm(rows(o, E), rows(h, E), ib.wo, B * mc * I, E, E,
                          res=View(ind, 0, L=I, zin=1, so=0, si=0, ss=E))
                self.mlp(ib, rows(h, E), B * mc * I)
                cview = lambda base: View(cache, j0 * I * 2 * E + base, L=I, zin=mc, so=m * I * 2 * E,  # noqa: E731
                                          si=I * 2 * E, ss=2 * E)
                self.gemm(rows(h, E), cview(0), cb.wkv, B * mc * I, E, 2 * E, norm_eps=cb.eps[1])
                self.cell_block(cb, x, N, B * mc, n, mc, n * mc, cview(0), cview(E), I)
        return caches

    def cell_block(self, cb, x, N, Z, L, zin, so_rows, kview, vview, I):
        """Cells attend to the inducing states of their column (Block.forward with cached keys). x rows
        are in (z_outer * L + i, jj) order with zin columns per sequence group."""
        E = self.E
        q = self.eng.temp("q", N * E)
        self.gemm(x, rows(q, E), cb.wq, N, E, E, norm_eps=cb.eps[0])
        qv = View(q, 0, L=L, zin=zin, so=so_rows * E, si=E, ss=zin * E)
        o = self.eng.temp("o", N * E)
        ov = View(o, 0, L=L, zin=zin, so=so_rows * E, si=E, ss=zin * E)
        self.attention(qv, kview, vview, ov, Z, cb.H, L, I, cb.hd, cb.hd, cb.hd, cb.hd, cb.hd)
        self.gemm(rows(o, E), x, cb.wo, N, E, E, res="self")
        self.mlp(cb, x, N)

    def column_query(self, st, caches, x, B, nr, m):
        """ColumnStage.query on test cells x: rows (b * nr + ii, j), contiguous (B, nr, m, E)."""
        E, I = self.E, st.n_ind
        N = B * nr * m
        for cb, cache in zip(st.cell_blocks, caches):
            cview = lambda base: View(cache, base, L=I, zin=1, so=I * 2 * E, si=0, ss=2 * E)  # noqa: E731
            self.cell_block(cb, x, N, B * m, nr, m, nr * m, cview(0), cview(E), I)

    def refine_rows(self, x, R, m):
        """RowRefinement on R rows of m cells, x a view with L = m (row z, cell i)."""
        E, rf = self.E, self.refine
        S = rf["n_summary"]
        N = R * m
        s = self.eng.temp("s", R * S * E)
        self.copy(View(rf["summary"], 0, L=S, zin=1, so=0, si=0, ss=E), rows(s, E), R * S, E)
        for i, bc in enumerate(rf["broadcast"]):
            H, hd = bc.H, bc.hd
            kvs = self.eng.temp("kvs", R * S * 2 * E)
            self.gemm(rows(s, E), rows(kvs, 2 * E), bc.wkv, R * S, E, 2 * E, norm_eps=bc.eps[1])
            q = self.eng.temp("q", N * E)
            self.gemm(x, rows(q, E), bc.wq, N, E, E, norm_eps=bc.eps[0])
            qv = View(q, 0, L=m, zin=1, so=m * E, ss=E)
            self.rope(qv, N, H, hd, hd, 0, m)
            o = self.eng.temp("o", N * E)
            kv_s = lambda base: View(kvs, base, L=S, zin=1, so=S * 2 * E, ss=2 * E)  # noqa: E731
            self.attention(qv, kv_s(0), kv_s(E), View(o, 0, L=m, zin=1, so=m * E, ss=E), R, H, m, S, hd, hd, hd, hd, hd)
            self.gemm(rows(o, E), x, bc.wo, N, E, E, res="self")
            self.mlp(bc, x, N)
            if i < len(rf["gather"]):
                g = rf["gather"][i]
                kvx = self.eng.temp("kv", N * 2 * E)
                self.gemm(x, rows(kvx, 2 * E), g.wkv, N, E, 2 * E, norm_eps=g.eps[1])
                kx = lambda base: View(kvx, base, L=m, zin=1, so=m * 2 * E, ss=2 * E)  # noqa: E731
                self.rope(kx(0), N, H, hd, hd, 0, m)
                qs = self.eng.temp("qs", R * S * E)
                self.gemm(rows(s, E), rows(qs, E), g.wq, R * S, E, E, norm_eps=g.eps[0])
                os_ = self.eng.temp("os", R * S * E)
                sv = lambda buf: View(buf, 0, L=S, zin=1, so=S * E, ss=E)  # noqa: E731
                self.attention(sv(qs), kx(0), kx(E), sv(os_), R, H, S, m, hd, hd, hd, hd, hd)
                self.gemm(rows(os_, E), rows(s, E), g.wo, R * S, E, E, res="self")
                self.mlp(g, rows(s, E), R * S)

    def row_stage(self, x, R, m, out):
        """RowStage on R rows of m cells (x a view with L = m); writes the R rows of C * E floats to view out."""
        E, C = self.E, self.C
        T = C + m
        xr = self.eng.temp("xr", R * T * E)
        self.copy(View(self.cls, 0, L=C, zin=1, so=0, si=0, ss=E), View(xr, 0, L=C, zin=1, so=T * E, ss=E), R * C, E)
        self.copy(x, View(xr, C * E, L=m, zin=1, so=T * E, ss=E), R * m, E)
        tok = lambda buf, w, base=0: View(buf, base, L=T, zin=1, so=T * w, ss=w)  # noqa: E731
        for blk in self.row_blocks:
            H, hd = blk.H, blk.hd
            kv = self.eng.temp("kv", R * T * 2 * E)
            self.gemm(rows(xr, E), rows(kv, 2 * E), blk.wkv, R * T, E, 2 * E, norm_eps=blk.eps[1])
            self.rope(tok(kv, 2 * E), R * T, H, hd, hd, C, m)
            if self.row_mode == "summary":
                sq = View(xr, 0, L=C, zin=1, so=T * E, ss=E)
                q = self.eng.temp("q", R * C * E)
                self.gemm(sq, rows(q, E), blk.wq, R * C, E, E, norm_eps=blk.eps[0])
                o = self.eng.temp("o", R * C * E)
                cv = lambda buf: View(buf, 0, L=C, zin=1, so=C * E, ss=E)  # noqa: E731
                self.attention(cv(q), tok(kv, 2 * E), tok(kv, 2 * E, E), cv(o), R, H, C, T, hd, hd, hd, hd, hd)
                self.gemm(rows(o, E), sq, blk.wo, R * C, E, E, res="self")
                self.mlp(blk, sq, R * C)
            else:
                q = self.eng.temp("q", R * T * E)
                self.gemm(rows(xr, E), rows(q, E), blk.wq, R * T, E, E, norm_eps=blk.eps[0])
                self.rope(tok(q, E), R * T, H, hd, hd, C, m)
                o = self.eng.temp("o", R * T * E)
                self.attention(tok(q, E), tok(kv, 2 * E), tok(kv, 2 * E, E), tok(o, E), R, H, T, T, hd, hd, hd, hd, hd)
                self.gemm(rows(o, E), rows(xr, E), blk.wo, R * T, E, E, res="self")
                self.mlp(blk, rows(xr, E), R * T)
        self.copy(View(xr, 0, L=1, zin=1, so=T * E), out, R, C * E)

    def icl_block(self, blk, x, B, Lq, kview, vview, Lk, k_h, v_h, kv_heads_cache=None):
        D = self.D
        N = B * Lq
        q = self.eng.temp("q", N * D)
        self.gemm(rows(x, D), rows(q, D), blk.wq, N, D, D, norm_eps=blk.eps[0])
        self.qscale(q, N, blk.scaling, Lk)
        o = self.eng.temp("o", N * D)
        qv = lambda buf: View(buf, 0, L=Lq, zin=1, so=Lq * D, ss=D)  # noqa: E731
        self.attention(qv(q), kview, vview, qv(o), B, blk.H, Lq, Lk, blk.hd, blk.hd, k_h, v_h, blk.hd)
        self.gemm(rows(o, D), rows(x, D), blk.wo, N, D, D, res="self")
        self.mlp(blk, rows(x, D), N)

    # interface -----------------------------------------------------------------------------------
    def _upload_normalized(self, X, stats):
        z, r, nan = CellEmbedder.normalize(X, stats)
        return tuple(self.eng.upload(t.contiguous().numpy()) for t in (z, r, nan))

    def _chunk(self, chunk_cells):
        return max(1, min(chunk_cells or GPU_CHUNK_CELLS, self.max_cells))

    @torch.inference_mode()
    def encode(self, X_train, y_train, d=None, slots=None, n_classes=None, has_padding=None, chunk_cells=None):
        if d is not None:
            raise NotImplementedError("the Vulkan backend does not take zero-padded feature counts (d)")
        X = torch.as_tensor(X_train).float().cpu()
        y = torch.as_tensor(y_train).long().cpu()
        B, n, m = X.shape
        if B == 0 or m == 0:
            raise ValueError("Vulkan encode needs a nonempty batch and at least one feature")
        if n == 0 and n_classes is None:
            raise ValueError("n_classes is required for an empty training context")
        n_classes = int(y.max()) + 1 if n_classes is None else n_classes
        if n_classes > self.cfg.label_slots:
            raise ValueError(f"{n_classes} classes, the model supports {self.cfg.label_slots}")
        slots = torch.arange(self.cfg.label_slots).expand(B, -1) if slots is None else torch.as_tensor(slots).long().cpu()
        y_slots = torch.gather(slots, 1, y)
        if B > self.train_capacity(n, m):
            raise MemoryError(f"{B} x {n} x {m} training cells exceed a Vulkan cache/scratch buffer limit")
        eng, E, D = self.eng, self.E, self.D
        stats = CellEmbedder.stats(X)
        zrn = self._upload_normalized(X, stats)
        slot_buf = eng.upload(y_slots.numpy().reshape(-1), np.uint32)
        chunk = self._chunk(chunk_cells)
        col_cost, row_cost = self._piece_costs(n, m)
        cols = min(max(1, chunk // (B * max(n, 1))), eng.max_binding // (4 * B * col_cost))
        nrows = min(max(1, chunk // (B * m)), eng.max_binding // (4 * B * row_cost))
        main = eng.empty(B * n * m * E)
        col_kv = self.column_context(self.col, main, B, n, m, cols, slot_buf, zrn)
        row_view = lambda i0, nr: View(main, i0 * m * E, L=m, zin=nr, so=n * m * E, si=m * E, ss=E)  # noqa: E731
        col_refine_kv = None
        if self.refine is not None:
            for i0 in range(0, n, nrows):
                nr = min(nrows, n - i0)
                self.refine_rows(row_view(i0, nr), B * nr, m)
            col_refine_kv = self.column_context(self.col_refine, main, B, n, m, cols, slot_buf)
        rows_buf = eng.empty(B * n * D)
        for i0 in range(0, n, nrows):
            nr = min(nrows, n - i0)
            self.row_stage(row_view(i0, nr), B * nr, m, View(rows_buf, i0 * D, L=1, zin=nr, so=n * D, si=D))
        del main
        # ICL over thinking rows + training rows
        T = self.n_thinking
        L = T + n
        x = eng.empty(B * L * D)
        self.copy(View(self.thinking, 0, L=T, zin=1, so=0, si=0, ss=D), View(x, 0, L=T, zin=1, so=L * D, ss=D), B * T, D)
        train = View(x, T * D, L=n, zin=1, so=L * D, ss=D)
        self.copy(rows(rows_buf, D), train, B * n, D)
        self.rownorm(train, B * n, D, emb=self.icl_y_emb, slots=slot_buf, slot_div=1)
        del rows_buf
        kvh = self.kv_heads * self.icl_blocks[0].hd
        icl_kv = []
        for blk in self.icl_blocks:
            kv = eng.temp("kv_icl", B * L * 2 * D)
            self.gemm(rows(x, D), rows(kv, 2 * D), blk.wkv, B * L, D, 2 * D, norm_eps=blk.eps[1])
            cache = eng.empty(B * L * 2 * kvh)
            self.copy(rows(kv, 2 * D), rows(cache, 2 * kvh), B * L, kvh)
            self.copy(rows(kv, 2 * D, D), rows(cache, 2 * kvh, kvh), B * L, kvh)
            kview = lambda base: View(kv, base, L=L, zin=1, so=L * 2 * D, ss=2 * D)  # noqa: E731
            self.icl_block(blk, x, B, L, kview(0), kview(D), L, blk.hd, blk.hd)
            icl_kv.append(cache)
        dec_k = eng.empty(B * n * D)
        self.gemm(train, rows(dec_k, D), self.dec_k, B * n, D, D, norm_eps=self.dec_eps)
        onehot = torch.nn.functional.one_hot(y, self.dec_hd).float()
        ctx = VulkanContext(stats, col_kv, col_refine_kv, icl_kv, dec_k, eng.upload(onehot.numpy()), B, n, m, n_classes)
        eng.flush()
        return ctx

    @torch.inference_mode()
    def predict_logits(self, ctx, X_test, chunk_cells=None):
        X = torch.as_tensor(X_test).float().cpu()
        B, M, m = X.shape
        if (B, m) != (ctx.B, ctx.m):
            raise ValueError(f"test batch {(B, m)} does not match the context {(ctx.B, ctx.m)}")
        if M == 0:
            return torch.zeros(B, 0, ctx.n_classes)
        eng, E, D = self.eng, self.E, self.D
        col_cost, row_cost = self._piece_costs(1, m)
        row_cost = max(row_cost, m * col_cost // self.col.n_ind)
        if 4 * B * max(M * m, M * self._icl_width(), row_cost) > eng.max_binding:
            raise MemoryError("query cache/scratch buffer exceeds the Vulkan limit; use fewer query rows")
        zrn = self._upload_normalized(X, ctx.stats)
        step = max(1, self._chunk(chunk_cells) // (B * m))
        step = min(step, eng.max_binding // (4 * B * row_cost))
        rows_buf = eng.empty(B * M * D)
        for i0 in range(0, M, step):
            nr = min(step, M - i0)
            cells = eng.temp("test_cells", B * nr * m * E)
            x = View(cells, 0, L=m, zin=1, so=m * E, ss=E)
            self.embed(zrn, x, B, M, m, i0, nr, 0, m)
            self.rownorm(x, B * nr * m, E, ln=(self.ln_w, self.ln_b, self.ln_eps))
            self.column_query(self.col, ctx.col_kv, x, B, nr, m)
            if self.refine is not None:
                self.refine_rows(x, B * nr, m)
                self.column_query(self.col_refine, ctx.col_refine_kv, x, B, nr, m)
            self.row_stage(x, B * nr, m, View(rows_buf, i0 * D, L=1, zin=nr, so=M * D, si=D))
        Lc = self.n_thinking + ctx.n
        kvh = self.kv_heads * self.icl_blocks[0].hd
        for blk, cache in zip(self.icl_blocks, ctx.icl_kv):
            kview = lambda base: View(cache, base, L=Lc, zin=1, so=Lc * 2 * kvh, ss=2 * kvh)  # noqa: E731
            head = 0 if self.kv_heads == 1 else blk.hd
            self.icl_block(blk, rows_buf, B, M, kview(0), kview(kvh), Lc, head, head)
        H, hd = self.dec_H, self.dec_hd
        q = eng.temp("q", B * M * D)
        self.gemm(rows(rows_buf, D), rows(q, D), self.dec_q, B * M, D, D, norm_eps=self.dec_eps)
        self.qscale(q, B * M, self.dec_scaling, ctx.n)
        o = eng.temp("o", B * M * D)
        qv = lambda buf: View(buf, 0, L=M, zin=1, so=M * D, ss=D)  # noqa: E731
        self.attention(qv(q), View(ctx.dec_k, 0, L=ctx.n, zin=1, so=ctx.n * D, ss=D),
                       View(ctx.onehot, 0, L=ctx.n, zin=1, so=ctx.n * hd, ss=hd), qv(o), B, H, M, ctx.n, hd, hd, hd, 0, hd)
        p = torch.from_numpy(eng.download(o, (B, M, H, hd))).mean(2)[..., : ctx.n_classes]
        return torch.log(p.clamp(min=1e-5) + 3e-5)
