"""D1 harness: per-channel mixed-precision weight quantization of a small CIFAR-10 ResNet.

- CIFAR ResNet (He et al. option-A shortcuts), widths 16/32/64, `n` blocks per stage
  (n=3 -> ResNet-20, n=1 -> ResNet-8). Weight-only quantization (activations stay float).
- Symmetric per-output-channel uniform quantizer, bits in [2, 8]; b bits -> integer levels
  in [-(2^(b-1)-1), 2^(b-1)-1] (2 bits = ternary); per-channel scale = MSE-optimal clipping.
- Budget = fan-in-weighted average bits over ALL conv + fc weights (first and last layer included).
- Sensitivity: diagonal empirical Fisher x dw^2.
- Allocators: uniform, per_layer, hawq_style, wf, baq, wf_global (definitions in the functions).
- Evaluation: (1) no retraining: round-to-nearest + BatchNorm re-estimation ("rtn_bn");
  (2) short quantization-aware fine-tune with straight-through estimator ("ft").
"""
import os, math, time, heapq, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BITS = list(range(2, 9))
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
_DATA = None


def load_cifar(val_n=5000, split_seed=1234):
    """Train 45k / validation 5k (fixed split carved from train) / test 10k, normalised float tensors."""
    global _DATA
    if _DATA is None:
        d = f"{ROOT}/data/cifar"
        f = lambda a: ((torch.from_numpy(a).permute(0, 3, 1, 2).float() / 255.0) - MEAN) / STD
        X = np.load(f"{d}/train_x.npy"); y = np.load(f"{d}/train_y.npy")
        perm = np.random.RandomState(split_seed).permutation(len(y))
        va, tr = perm[:val_n], perm[val_n:]
        _DATA = dict(xtr=f(X[tr]), ytr=torch.from_numpy(y[tr]), xva=f(X[va]), yva=torch.from_numpy(y[va]),
                     xte=f(np.load(f"{d}/test_x.npy")), yte=torch.from_numpy(np.load(f"{d}/test_y.npy")))
    return _DATA


def augment(x, g):
    """Random crop (pad 4) + horizontal flip; 8 crop offsets per batch for speed."""
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
    out = torch.empty_like(x)
    for idx in torch.arange(len(x)).chunk(8):
        i, j = torch.randint(0, 9, (2,), generator=g).tolist()
        out[idx] = xp[idx, :, i:i + 32, j:j + 32]
    flip = torch.rand(len(x), generator=g) < 0.5
    out[flip] = out[flip].flip(3)
    return out


# ----------------------------------------------------------------------------- model
class QMixin:
    """Per-output-channel fake quantization. bits[c] == 0 means channel c stays float."""
    def init_q(self):
        self.q_on = False
        self.register_buffer("q_bits", torch.zeros(self.weight.shape[0], dtype=torch.long))
        self.register_buffer("q_scale", torch.ones(self.weight.shape[0]))

    def qweight(self):
        if not self.q_on:
            return self.weight
        w = self.weight
        sh = (-1,) + (1,) * (w.dim() - 1)
        n = (2.0 ** (self.q_bits.clamp(min=1).float() - 1) - 1).view(sh)
        s = self.q_scale.view(sh)
        wq = torch.maximum(torch.minimum(torch.round(w / s), n), -n) * s
        wq = torch.where((self.q_bits > 0).view(sh), wq, w)
        return w + (wq - w).detach()                            # straight-through estimator


class QConv2d(nn.Conv2d, QMixin):
    def forward(self, x):
        return F.conv2d(x, self.qweight(), None, self.stride, self.padding)


class QLinear(nn.Linear, QMixin):
    def forward(self, x):
        return F.linear(x, self.qweight(), self.bias)


def conv3(cin, cout, stride=1):
    m = QConv2d(cin, cout, 3, stride, 1, bias=False); m.init_q(); return m


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = conv3(cin, cout, stride); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = conv3(cout, cout); self.b2 = nn.BatchNorm2d(cout)
        self.pad = (cout - cin) // 2 if (stride != 1 or cin != cout) else None
        self.stride = stride

    def forward(self, x):
        o = self.b2(self.c2(F.relu(self.b1(self.c1(x)))))
        if self.pad is not None:
            x = F.pad(x[:, :, ::self.stride, ::self.stride], (0, 0, 0, 0, self.pad, self.pad))
        return F.relu(o + x)


class ResNet(nn.Module):
    def __init__(self, n=1, width=16, classes=10):
        super().__init__()
        self.stem = conv3(3, width); self.bn = nn.BatchNorm2d(width)
        blocks, cin = [], width
        for i, w in enumerate([width, 2 * width, 4 * width]):
            for j in range(n):
                blocks.append(Block(cin, w, 2 if (i > 0 and j == 0) else 1)); cin = w
        self.blocks = nn.Sequential(*blocks)
        self.fc = QLinear(cin, classes); self.fc.init_q()

    def forward(self, x):
        x = self.blocks(F.relu(self.bn(self.stem(x))))
        return self.fc(x.mean((2, 3)))


def qlayers(model):
    """Ordered dict name -> quantizable module."""
    return {n: m for n, m in model.named_modules() if isinstance(m, QMixin)}


def fan_in(m):
    return m.weight[0].numel()


@torch.no_grad()
def evaluate(model, x, y, bs=1000):
    model.eval(); correct = 0; loss = 0.0
    for i in range(0, len(y), bs):
        o = model(x[i:i + bs])
        correct += (o.argmax(1) == y[i:i + bs]).sum().item()
        loss += F.cross_entropy(o, y[i:i + bs], reduction="sum").item()
    return 100.0 * correct / len(y), loss / len(y)


def train_float(seed, n=1, width=16, epochs=15, lr=0.1, bs=128, wd=5e-4, log=print):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    D = load_cifar(); model = ResNet(n, width)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    N = len(D["ytr"]); spe = N // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * spe, pct_start=0.2)
    t0 = time.time()
    for ep in range(epochs):
        model.train(); perm = torch.randperm(N, generator=g)
        for i in range(spe):
            idx = perm[i * bs:(i + 1) * bs]
            loss = F.cross_entropy(model(augment(D["xtr"][idx], g)), D["ytr"][idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        log(f"seed {seed} ep {ep} val {evaluate(model, D['xva'], D['yva'])[0]:.2f} t {time.time()-t0:.0f}s")
    return model


# ----------------------------------------------------------------------------- quantizer tables
@torch.no_grad()
def quant_tables(model, ratios=40):
    """For each layer and b in BITS: MSE-optimal per-channel scale, and per-weight error (w - Q_b(w))."""
    T = {}
    for name, m in qlayers(model).items():
        w = m.weight.flatten(1); mx = w.abs().max(1).values.clamp(min=1e-8)
        scale, dw = {}, {}
        for b in BITS:
            n = 2 ** (b - 1) - 1
            best_e = torch.full((w.shape[0],), float("inf")); best_s = mx / n; best_q = w.clone()
            for r in torch.linspace(0.2, 1.0, ratios):
                s = (r * mx / n).unsqueeze(1)
                q = (w / s).round().clamp(-n, n) * s
                e = ((w - q) ** 2).sum(1)
                better = e < best_e
                best_e = torch.where(better, e, best_e); best_s = torch.where(better, s.squeeze(1), best_s)
                best_q[better] = q[better]
            scale[b] = best_s; dw[b] = w - best_q
        T[name] = dict(scale=scale, dw=dw, max=mx, fan_in=w.shape[1], n_ch=w.shape[0])
    return T


def set_bits(model, tables, alloc):
    """alloc: name -> integer array of bits per channel (0 = float). alloc None -> quantization off."""
    for name, m in qlayers(model).items():
        if alloc is None:
            m.q_on = False; continue
        b = torch.as_tensor(np.asarray(alloc[name]), dtype=torch.long)
        m.q_bits.copy_(b)
        s = torch.ones(len(b))
        for bb in BITS:
            s[b == bb] = tables[name]["scale"][bb][b == bb]
        m.q_scale.copy_(s); m.q_on = True


def avg_bits(tables, alloc):
    num = sum(float(np.sum(np.asarray(alloc[k]))) * tables[k]["fan_in"] for k in tables)
    den = sum(tables[k]["n_ch"] * tables[k]["fan_in"] for k in tables)
    return num / den


# ----------------------------------------------------------------------------- sensitivities
def fisher_diag(model, x, y, bs=8):
    """Diagonal empirical Fisher (mean of squared mini-batch gradients, bs=8) per quantizable weight."""
    model.eval(); L = qlayers(model)
    Fd = {k: torch.zeros_like(m.weight) for k, m in L.items()}
    nb = 0
    for i in range(0, len(y), bs):
        model.zero_grad()
        F.cross_entropy(model(x[i:i + bs]), y[i:i + bs]).backward()
        for k, m in L.items():
            Fd[k] += m.weight.grad ** 2
        nb += 1
    return {k: (v / nb).flatten(1) for k, v in Fd.items()}


def hessian_trace(model, x, y, probes=16):
    """Hutchinson estimate of the Hessian trace per quantizable layer (HAWQ-V2 style)."""
    model.eval(); L = qlayers(model); ws = [m.weight for m in L.values()]
    loss = F.cross_entropy(model(x), y)
    gs = torch.autograd.grad(loss, ws, create_graph=True)
    tr = [0.0] * len(ws)
    for _ in range(probes):
        vs = [torch.randint_like(w, 2) * 2 - 1 for w in ws]
        hv = torch.autograd.grad(gs, ws, vs, retain_graph=True)
        for j in range(len(ws)):
            tr[j] += (hv[j] * vs[j]).sum().item() / probes
    return dict(zip(L.keys(), tr))


def input_second_moment(model, x):
    """Mean squared input activation per quantizable layer (diag of the layer-wise X X^T proxy)."""
    L = qlayers(model); out = {}; hooks = []
    for k, m in L.items():
        hooks.append(m.register_forward_hook(lambda mod, inp, o, k=k: out.__setitem__(
            k, (inp[0].pow(2).mean().item(), o.pow(2).mean().item()))))
    model.eval()
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return out


def sensitivities(model, tables, D, n_fisher=2048, n_hess=512, seed=0):
    idx = torch.randperm(len(D["ytr"]), generator=torch.Generator().manual_seed(seed))[:n_fisher]
    x, y = D["xtr"][idx], D["ytr"][idx]
    Fd = fisher_diag(model, x, y)
    tr = hessian_trace(model, x[:n_hess], y[:n_hess])
    sm = input_second_moment(model, x[:n_hess])
    S = {}
    for k, t in tables.items():
        f = Fd[k]
        S[k] = dict(
            c=(f.sum(1) * (2 * t["max"]) ** 2 / 12.0).numpy(),                       # law: loss_i(b) = c_i 4^-b
            table=np.stack([(f * t["dw"][b] ** 2).sum(1).numpy() for b in BITS], 1),  # measured additive
            hawq=np.array([max(tr[k], 0.0) / f.numel() * (t["dw"][b] ** 2).sum().item() for b in BITS]),
            # layer-local output-reconstruction proxy (relative output error), no task-loss information
            c_local=(sm[k][0] * t["fan_in"] * (2 * t["max"]) ** 2 / 12.0 / max(sm[k][1], 1e-12)).numpy(),
            hess_trace=tr[k])
    return S


# ----------------------------------------------------------------------------- allocators
def _greedy(cost, weight, budget_units):
    """Minimise sum_i cost[i, b_i] s.t. sum_i b_i*weight[i] <= budget_units, b in [2, 8]; greedy on
    marginal gain per unit cost (items that no longer fit are skipped)."""
    n = len(weight); b = np.full(n, 2); used = 2 * weight.sum()
    heap = [(-(cost[i, 0] - cost[i, 1]) / weight[i], i) for i in range(n)]
    heapq.heapify(heap)
    while heap:
        _, i = heapq.heappop(heap)
        if used + weight[i] > budget_units + 1e-9:
            continue
        b[i] += 1; used += weight[i]
        if b[i] < 8:
            j = b[i] - 2
            heapq.heappush(heap, (-(cost[i, j] - cost[i, j + 1]) / weight[i], i))
    return b


def _flat(tables, S, key):
    names = list(tables.keys())
    v = np.concatenate([S[k][key] for k in names], 0)
    f = np.concatenate([np.full(tables[k]["n_ch"], tables[k]["fan_in"], float) for k in names])
    return names, v, f


def _unflat(tables, names, b):
    out, i = {}, 0
    for k in names:
        out[k] = b[i:i + tables[k]["n_ch"]].astype(int); i += tables[k]["n_ch"]
    return out


def _bisect_rule(score, f, budget):
    """b_i = clip(round(0.5*log2(score_i/theta)), 2, 8); largest allocation with avg bits <= budget."""
    ls = 0.5 * np.log2(np.maximum(score, 1e-300))
    rule = lambda t: np.clip(np.round(ls - t), 2, 8)
    lo, hi = ls.min() - 10, ls.max() + 10                      # t = 0.5*log2(theta)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if (rule(mid) * f).sum() / f.sum() > budget:
            lo = mid
        else:
            hi = mid
    return rule(hi)


def alloc_uniform(tables, S, budget):
    assert float(budget).is_integer()
    return {k: np.full(t["n_ch"], int(budget)) for k, t in tables.items()}


def alloc_wf(tables, S, budget):
    """Fisher water-filling, closed form, one global theta: b_i = clip(round(.5 log2(c_i/(theta f_i))),2,8)."""
    names, c, f = _flat(tables, S, "c")
    return _unflat(tables, names, _bisect_rule(c / f, f, budget))


def alloc_baq(tables, S, budget):
    """BAQ-style closed form R_j = round(max(0, .5 log2(C_j/lambda))), floor 2, lambda by bisection, with a
    LAYER-LOCAL sensitivity C_j (input second moment x range^2/12, relative to layer output power; per weight).
    Our reading of BAQ (arXiv 2506.05664): closed-form rule on a layer-wise Hessian proxy, no task-loss Fisher."""
    names, c, f = _flat(tables, S, "c_local")
    return _unflat(tables, names, _bisect_rule(c / f, f, budget))


def alloc_wf_global(tables, S, budget):
    """Additive allocation free to move budget anywhere: exact-table greedy on the MEASURED per-channel
    Fisher x dw(b)^2 table (no 4^-b law, no rounding of a closed form)."""
    names, tab, f = _flat(tables, S, "table")
    return _unflat(tables, names, _greedy(tab, f, budget * f.sum()))


def alloc_per_layer(tables, S, budget):
    """One bit-width per layer; greedy on the layer-summed Fisher x dw(b)^2 table."""
    names = list(tables.keys())
    tab = np.stack([S[k]["table"].sum(0) for k in names]); w = np.array([tables[k]["n_ch"] * tables[k]["fan_in"] for k in names], float)
    b = _greedy(tab, w, budget * w.sum())
    return {k: np.full(tables[k]["n_ch"], int(b[i])) for i, k in enumerate(names)}


def alloc_hawq(tables, S, budget):
    """HAWQ-style: one bit-width per layer; cost = mean Hessian trace x ||Q_b(W) - W||^2 (Hutchinson)."""
    names = list(tables.keys())
    tab = np.stack([S[k]["hawq"] for k in names]); w = np.array([tables[k]["n_ch"] * tables[k]["fan_in"] for k in names], float)
    b = _greedy(tab, w, budget * w.sum())
    return {k: np.full(tables[k]["n_ch"], int(b[i])) for i, k in enumerate(names)}


ALLOCATORS = dict(uniform=alloc_uniform, per_layer=alloc_per_layer, hawq_style=alloc_hawq, wf=alloc_wf,
                  baq=alloc_baq, wf_global=alloc_wf_global)


# ----------------------------------------------------------------------------- evaluation of an allocation
@torch.no_grad()
def bn_recalibrate(model, x, bs=500):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats(); m.momentum = None
    model.train()
    for i in range(0, len(x), bs):
        model(x[i:i + bs])
    model.eval()


def finetune(model, D, steps, seed, lr=0.01, bs=128):
    """Short QAT with straight-through estimator, scales fixed, bits fixed."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    model.train(); N = len(D["ytr"])
    for _ in range(steps):
        idx = torch.randint(0, N, (bs,), generator=g)
        loss = F.cross_entropy(model(augment(D["xtr"][idx], g)), D["ytr"][idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval()


def eval_alloc(float_state, tables, alloc, D, seed, n=1, width=16, ft_steps=100, calib_n=2000):
    """Returns accuracies (val and test) for rtn (raw), rtn_bn (BN re-estimated) and ft (short QAT)."""
    model = ResNet(n, width); model.load_state_dict(float_state)
    set_bits(model, tables, alloc)
    r = dict(avg_bits=avg_bits(tables, alloc))
    r["rtn_val"], _ = evaluate(model, D["xva"], D["yva"]); r["rtn_test"], _ = evaluate(model, D["xte"], D["yte"])
    cal = torch.randperm(len(D["ytr"]), generator=torch.Generator().manual_seed(seed))[:calib_n]
    bn_recalibrate(model, D["xtr"][cal])
    r["rtn_bn_val"], _ = evaluate(model, D["xva"], D["yva"]); r["rtn_bn_test"], _ = evaluate(model, D["xte"], D["yte"])
    if ft_steps:
        finetune(model, D, ft_steps, seed)
        r["ft_val"], _ = evaluate(model, D["xva"], D["yva"]); r["ft_test"], _ = evaluate(model, D["xte"], D["yte"])
    return r
