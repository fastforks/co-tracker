"""General fast inference for CoTracker3 offline (exp 007, attempt/general): any video length / point count / image size,
nothing compiled or tuned per size, plain PyTorch ops only (exportable).

    from cotracker.fast import FastCoTrackerPredictor
    pred = FastCoTrackerPredictor("scaled_offline.pth")   # same call contract as CoTrackerPredictor (offline)
    tracks, vis = pred(video, grid_size=30)               # video [B,T,3,H,W] float 0..255 on the GPU

Changes vs upstream (fp32 everywhere; network math unchanged, precision / layout / op choice only):
- feature CNN (BasicEncoder) in fp16 channels-last; L2 feature normalisation, the feature pyramid and all bilinear
  sampling stay fp32 (sub-pixel coordinates)
- correlation (corr="gather", default): the 7x7 window points share one fractional offset, so the window is gathered at
  the enclosing 8x8 integer positions from fp16 features, correlated with the 49 support features in one fp16 batched
  GEMM per level in (B, N, T) order, and the correlations are bilinearly interpolated (exact algebra; border padding =
  index clamp). Upstream: 3D grid_sample of 128-d features at 49 sub-pixel points + einsum + permutes in fp32.
  corr="sample": fp32 2D grid_sample + fp16 GEMM (first version). Corr MLP in fp16, 2401-wide input zero-padded to 2408
- update transformer in fp16 with fused SDPA (upstream: explicit softmax(QK^T)V in fp32); coordinates, visibility and
  confidence accumulate in fp32
- `gacc`: optional fp16-accumulation GEMMs, and (`hacc`, default = gacc) feature-CNN convs through fastbb.cudnn_conv
  (off for the friendly variant)
"""
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from cotracker.models.build_cotracker import build_cotracker
from cotracker.models.core.cotracker.blocks import Attention
from cotracker.models.core.cotracker.cotracker3_online import posenc
from cotracker.predictor import CoTrackerPredictor


def _sdpa_attention_forward(self, x, context=None, attn_bias=None):
    B, N1, C = x.shape
    h = self.heads
    q = self.to_q(x).view(B, N1, h, C // h).transpose(1, 2)
    context = x if context is None else context
    N2 = context.shape[1]
    kv = self.to_kv(context).view(B, N2, 2, h, C // h)
    k, v = kv.unbind(2)
    o = F.scaled_dot_product_attention(q, k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_bias, scale=self.scale)
    return self.to_out(o.transpose(1, 2).reshape(B, N1, C))


@contextlib.contextmanager
def fp16_accumulate(enabled):
    old = torch.backends.cuda.matmul.allow_fp16_accumulation
    torch.backends.cuda.matmul.allow_fp16_accumulation = bool(enabled)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation = old


def _inorm_forward(x):
    """InstanceNorm2d (affine=False, eps 1e-5) for fp16 channels-last maps without the NCHW round trip of
    F.instance_norm (fp32 accumulation inside var_mean)"""
    v, m = torch.var_mean(x, dim=(2, 3), keepdim=True, unbiased=False)
    a = (v.float() + 1e-5).rsqrt()
    return torch.addcmul((-m.float() * a).half(), x, a.half())  # (x - m) * a in one pass


def _support_offsets(r, device):
    d = torch.linspace(-r, r, 2 * r + 1, device=device)
    a, b = torch.meshgrid(d, d, indexing="ij")
    # upstream get_support_points: x offset varies along the slow axis, y offset along the fast axis
    return torch.stack([a, b], -1).reshape(-1, 2)  # [(2r+1)^2, (x_off, y_off)]


def _sample2d(fm, xy):
    """fm [M, C, H, W] fp32, xy [M, P, Q, 2] pixel coords (x, y), align_corners=True, border -> [M, C, P, Q]"""
    H, W = fm.shape[-2:]
    g = xy * xy.new_tensor([2 / max(W - 1, 1), 2 / max(H - 1, 1)]) - 1
    return F.grid_sample(fm, g, mode="bilinear", padding_mode="border", align_corners=True)


class FastCoTrackerThreeOffline(nn.Module):
    def __init__(self, model, gacc=False, corr="gather", hacc=False):
        super().__init__()
        self.corr = corr
        self.m = m = model.eval()
        self.gacc = gacc
        m.fnet.half().to(memory_format=torch.channels_last)
        if hacc:  # fp16-accumulating cuDNN convs in the feature CNN (fastbb.cudnn_conv: precompiled plans)
            from fastbb.cudnn_conv import patch_convs
            patch_convs(m.fnet)
        for mod in m.fnet.modules():
            if isinstance(mod, nn.InstanceNorm2d):
                assert not mod.affine and not mod.track_running_stats and mod.eps == 1e-5
                mod.forward = _inorm_forward
        m.updateformer.half()
        for mod in m.updateformer.modules():
            if isinstance(mod, Attention):
                mod.forward = _sdpa_attention_forward.__get__(mod)
        # corr MLP fc1: pad K 2401 -> 2408 (zero columns); input built with the same padding
        fc1 = m.corr_mlp.fc1
        self.k_pad = (-fc1.in_features) % 8
        w = F.pad(fc1.weight.data, (0, self.k_pad)).half()
        self.c1w, self.c1b = w, fc1.bias.data.half()
        self.c2w, self.c2b = m.corr_mlp.fc2.weight.data.half(), m.corr_mlp.fc2.bias.data.half()
        # update-former input transform: pad K 1110 -> 1112
        it = m.updateformer.input_transform
        self.in_pad = (-it.in_features) % 8
        self.itw = F.pad(it.weight.data, (0, self.in_pad))
        self.itb = it.bias.data
        self.settings = f"fp16 fnet (channels-last) / corr GEMM + MLP / update transformer (SDPA), corr={corr}, fp32 coords, " \
                        f"gemm_acc16={gacc}, conv_acc16={hacc}"

    def _corr_gather(self, f16, c, sup, B, T, N, h, w, kbuf):
        """local correlation without sampling 128-d features at 49 sub-pixel points: all 49 points of the 7x7 window share
        one fractional offset, so bilinear sampling of the window = bilinear interpolation over the enclosing 8x8 integer
        window. Gather the 8x8 integer window (fp16, border = index clamp, identical to grid_sample border padding),
        correlate with the 49 support features in one fp16 GEMM, then interpolate the correlations (linear in the features).
        f16 [B*T*h*w, D]; c [B, N, T, 2] level coords (x, y); sup [B, N, D, RR] -> kbuf [B*N*T, 2401 + pad] (filled)"""
        r = self.m.corr_radius
        R = 2 * r + 1
        c0 = c.floor()
        fr = (c - c0).half()  # [B, N, T, 2]
        k = torch.arange(-r, r + 2, device=c.device)  # 8 integer offsets
        xs = (c0[..., 0:1].long() + k).clamp_(0, w - 1)  # [B, N, T, 8]
        ys = (c0[..., 1:2].long() + k).clamp_(0, h - 1)
        base = (torch.arange(B, device=c.device)[:, None, None] * T + torch.arange(T, device=c.device)) * (h * w)  # [B, 1, T]
        idx = base[..., None, None] + ys[..., None, :] * w + xs[..., :, None]  # [B, N, T, 8 (x), 8 (y)]
        g = f16[idx]  # [B, N, T, 8, 8, D] fp16
        D = g.shape[-1]
        cw = torch.matmul(g.view(B, N, T * 64, D), sup).view(B, N, T, 8, 8, R * R)  # correlations at integer positions
        cy = torch.lerp(cw[..., :, :R, :], cw[..., :, 1:, :], fr[..., 1, None, None, None])  # along y -> [.., 8, 7, RR]
        out = kbuf[:, : R ** 4].view(B, N, T, R, R, R * R)
        torch.lerp(cy[..., :R, :, :], cy[..., 1:, :, :], fr[..., 0, None, None, None], out=out)  # along x
        return kbuf

    def updateformer(self, x, add_space_attn=True):
        uf = self.m.updateformer
        tokens = F.linear(x, self.itw, self.itb)
        B, _, T, _ = tokens.shape
        tokens = torch.cat([tokens, uf.virual_tracks.repeat(B, 1, T, 1)], dim=1)
        _, N, _, _ = tokens.shape
        nv = uf.num_virtual_tracks
        j = 0
        for i in range(len(uf.time_blocks)):
            tokens = uf.time_blocks[i](tokens.reshape(B * N, T, -1)).view(B, N, T, -1)
            if add_space_attn and hasattr(uf, "space_virtual_blocks") and \
                    i % (len(uf.time_blocks) // len(uf.space_virtual_blocks)) == 0:
                st = tokens.permute(0, 2, 1, 3).reshape(B * T, N, -1)
                pt, vt = st[:, : N - nv], st[:, N - nv:]
                vt = uf.space_virtual2point_blocks[j](vt, pt)
                vt = uf.space_virtual_blocks[j](vt)
                pt = uf.space_point2virtual_blocks[j](pt, vt)
                tokens = torch.cat([pt, vt], dim=1).view(B, T, N, -1).permute(0, 2, 1, 3)
                j += 1
        tokens = tokens[:, : N - nv]
        flow = uf.flow_head(tokens)
        if uf.linear_layer_for_vis_conf:
            flow = torch.cat([flow, uf.vis_conf_head(tokens)], dim=-1)
        return flow

    @torch.no_grad()
    def forward(self, video, queries, iters=4, is_train=False, add_space_attn=True, fmaps_chunk_size=200):
        with fp16_accumulate(self.gacc):
            return self._forward(video, queries, iters, add_space_attn, fmaps_chunk_size)

    def _forward(self, video, queries, iters, add_space_attn, fmaps_chunk_size):
        m = self.m
        B, T, C, H, W = video.shape
        assert H % m.stride == 0 and W % m.stride == 0
        N = queries.shape[1]
        s, D, r = m.stride, m.latent_dim, m.corr_radius
        R = 2 * r + 1
        H4, W4 = H // s, W // s
        x = (video.reshape(B * T, C, H, W) * (2 / 255.0) - 1.0).half().contiguous(memory_format=torch.channels_last)
        fm = torch.cat([m.fnet(x[i:i + fmaps_chunk_size]) for i in range(0, B * T, fmaps_chunk_size)])
        fm = fm.float()
        fm = (fm * fm.square().sum(1, keepdim=True).clamp_min(1e-12).rsqrt()).contiguous()  # [BT, D, H4, W4] fp32 (as upstream)
        pyr = [fm]
        for _ in range(m.corr_levels - 1):
            pyr.append(F.avg_pool2d(pyr[-1], 2, stride=2))
        if self.corr == "gather":  # fp16 channels-last copies, flattened [B*T*h*w, D], for the integer window gathers
            pyr16 = [f.half().permute(0, 2, 3, 1).reshape(-1, D) for f in pyr]
            kbuf = torch.zeros(B * N * T, R ** 4 + self.k_pad, device=video.device, dtype=torch.float16)

        qf = queries[:, :, 0].long()
        qxy = queries[..., 1:3] / s  # [B, N, 2]
        off = _support_offsets(r, video.device)  # [R*R, 2]
        # track support features per level (3D sampling at the query frame, as upstream): [B, N, D, R*R] fp16
        sup = []
        for i, f in enumerate(pyr):
            h, w = f.shape[-2:]
            pts = (qxy / 2 ** i)[:, :, None, :] + off  # [B, N, RR, 2] (x, y)
            g = torch.cat([pts * pts.new_tensor([2 / max(w - 1, 1), 2 / max(h - 1, 1)]) - 1,
                           (qf.float() * (2 / max(T - 1, 1)) - 1)[:, :, None, None].expand(B, N, R * R, 1)], -1)
            sf = F.grid_sample(f.view(B, T, D, h, w).transpose(1, 2), g[:, :, :, None], mode="bilinear",
                               padding_mode="border", align_corners=True)  # [B, D, N, RR, 1]
            sup.append(sf[..., 0].permute(0, 2, 1, 3).half().contiguous())  # [B, N, D, RR]

        coords = qxy.reshape(B, 1, N, 2).expand(B, T, N, 2).float()
        vis = torch.zeros((B, T, N), device=video.device)
        conf = torch.zeros((B, T, N), device=video.device)
        scale = torch.tensor([m.model_resolution[1], m.model_resolution[0]], device=video.device) / s
        temb = None
        out = None
        for it in range(iters):
            cn = coords.permute(0, 2, 1, 3)  # [B, N, T, 2]
            embs = []
            for i, f in enumerate(pyr):
                if self.corr == "gather":
                    h, w = pyr[i].shape[-2:]
                    cv = self._corr_gather(pyr16[i], cn / 2 ** i, sup[i], B, T, N, h, w, kbuf)
                else:
                    pts = (coords.reshape(B * T, N, 1, 2) / 2 ** i) + off  # [BT, N, RR, 2]
                    cf = _sample2d(f, pts)  # [BT, D, N, RR] fp32
                    cf = cf.view(B, T, D, N, R * R).permute(0, 3, 1, 4, 2).half()  # [B, N, T, RR, D]
                    # local correlation: [B,N,T*RR,D] @ [B,N,D,RR] -> [B,N,T*RR,RR] -> rows (b,n,t) x (hw, ij)
                    cv = torch.matmul(cf.reshape(B, N, T * R * R, D), sup[i])
                    cv = cv.view(B * N * T, R ** 4)
                    if self.k_pad:
                        cv = F.pad(cv, (0, self.k_pad))
                e = F.linear(F.gelu(F.linear(cv, self.c1w, self.c1b)), self.c2w, self.c2b)
                embs.append(e.view(B, N, T, -1))
            rel_f = F.pad(cn[:, :, :-1] - cn[:, :, 1:], (0, 0, 0, 1))
            rel_b = F.pad(cn[:, :, 1:] - cn[:, :, :-1], (0, 0, 1, 0))
            pe = posenc(torch.cat([rel_f, rel_b], -1) / scale.repeat(2), min_deg=0, max_deg=10)
            parts = [vis.permute(0, 2, 1)[..., None].half(), conf.permute(0, 2, 1)[..., None].half()] + embs + [pe.half()]
            if self.in_pad:
                parts.append(torch.zeros(B, N, T, self.in_pad, device=video.device, dtype=torch.float16))
            xin = torch.cat(parts, -1)  # [B, N, T, 1112]
            if temb is None:
                temb = m.interpolate_time_embed(xin[..., : m.input_dim].reshape(B * N, T, -1).float(), T)
                temb = F.pad(temb, (0, self.in_pad)).half().view(1, 1, T, -1)
            delta = self.updateformer(xin + temb, add_space_attn=add_space_attn).float()  # [B, N, T, 4]
            coords = coords + delta[..., :2].permute(0, 2, 1, 3)
            vis = vis + delta[..., 2].permute(0, 2, 1)
            conf = conf + delta[..., 3].permute(0, 2, 1)
        return coords * float(s), torch.sigmoid(vis), torch.sigmoid(conf), None


class FastCoTrackerPredictor(CoTrackerPredictor):
    """CoTrackerPredictor (offline) with the fast model; preprocessing / query handling unchanged (upstream code)."""

    def __init__(self, checkpoint, window_len=60, gacc=False, hacc=None, device="cuda"):
        nn.Module.__init__(self)
        self.v2 = False
        self.support_grid_size = 6
        model = build_cotracker(checkpoint, v2=False, offline=True, window_len=window_len).to(device)
        self.interp_shape = model.model_resolution
        self.model = FastCoTrackerThreeOffline(model, gacc=gacc, hacc=gacc if hacc is None else hacc)
        self.settings = self.model.settings
