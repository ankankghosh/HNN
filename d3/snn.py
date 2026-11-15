"""D3 harness: recurrent spiking network (LIF, surrogate gradient) on SHD, plain PyTorch.

- dt = 10 ms, T = 100 steps (first 1.0 s), 700 channels binned to 140 inputs (spike counts).
- One recurrent layer of H LIF units with a PER-NEURON membrane time constant and threshold,
  leaky-integrator readout, cross-entropy on the time-averaged readout.
- Validation split: fixed stratified 10% of the SHD training set (the SHD test set is never
  used for model selection). Reported test accuracy = test accuracy at the best-validation epoch.
- Counters: hidden spikes per neuron per sample; SynOps = spikes x fan-out.

A run is described by a config dict (see `train`). Allocators only have to supply
`tau0` (per-neuron initial time constant), `learn_mask` (which taus are trainable) and `thr`.
"""
import os, json, time, math
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DT = 0.010
TAU_MIN, TAU_MAX = 0.005, 0.5
E_SYNOP_PJ = 0.9  # pJ per synaptic accumulate (45 nm convention)

_DATA = None


def load_shd(val_frac=0.1, split_seed=1234):
    """Returns dict of uint8 tensors. Validation = fixed stratified split of the training set."""
    global _DATA
    if _DATA is not None:
        return _DATA
    d = f"{ROOT}/data/shd"
    X = np.load(f"{d}/train_x.npy"); y = np.load(f"{d}/train_y.npy")
    Xt = np.load(f"{d}/test_x.npy"); yt = np.load(f"{d}/test_y.npy")
    rng = np.random.RandomState(split_seed)
    val_idx = []
    for c in range(20):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        val_idx.extend(idx[: int(round(val_frac * len(idx)))])
    val_mask = np.zeros(len(y), bool); val_mask[val_idx] = True
    t = torch.from_numpy
    _DATA = dict(xtr=t(X[~val_mask]), ytr=t(y[~val_mask]), xva=t(X[val_mask]), yva=t(y[val_mask]),
                 xte=t(Xt), yte=t(yt))
    return _DATA


class SpikeFn(torch.autograd.Function):
    """Heaviside with fast-sigmoid surrogate gradient (slope 10)."""
    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)
        return (v > 0).float()

    @staticmethod
    def backward(ctx, g):
        (v,) = ctx.saved_tensors
        return g / (10.0 * v.abs() + 1.0) ** 2


class RSNN(nn.Module):
    def __init__(self, n_in=140, H=128, n_out=20, tau0=0.02, learn_mask=None, thr=1.0, tau_out=0.02,
                 w_scale=1.0):
        super().__init__()
        self.H = H
        self.w_in = nn.Parameter(torch.randn(H, n_in) * w_scale / math.sqrt(n_in))
        self.w_rec = nn.Parameter(torch.randn(H, H) * w_scale / math.sqrt(H))
        self.w_out = nn.Parameter(torch.randn(n_out, H) / math.sqrt(H))
        tau0 = torch.as_tensor(np.broadcast_to(np.asarray(tau0, dtype=np.float32), (H,)).copy())
        self.log_tau = nn.Parameter(torch.log(tau0))          # per-neuron membrane time constant
        lm = np.zeros(H, bool) if learn_mask is None else np.broadcast_to(np.asarray(learn_mask, bool), (H,))
        self.register_buffer("learn_mask", torch.as_tensor(lm.copy()).float())
        self.register_buffer("thr", torch.as_tensor(np.broadcast_to(np.asarray(thr, np.float32), (H,)).copy()))
        self.beta_out = math.exp(-DT / tau_out)
        self.log_tau.register_hook(lambda g: g * self.learn_mask)

    def tau(self):
        return self.log_tau.exp().clamp(TAU_MIN, TAU_MAX)

    def forward(self, x, return_spikes=False):
        """x: (B, T, n_in) float. Returns logits (B, n_out), spike counts per neuron (B, H)."""
        B, T, _ = x.shape
        alpha = torch.exp(-DT / self.tau())
        v = x.new_zeros(B, self.H); s = x.new_zeros(B, self.H)
        u = x.new_zeros(B, self.w_out.shape[0]); usum = 0.0
        cnt = x.new_zeros(B, self.H)
        inp = x @ self.w_in.t()                                # (B, T, H)
        spikes = []
        for t in range(T):
            i_t = inp[:, t] + s @ self.w_rec.t()
            v = alpha * v + (1 - alpha) * i_t - self.thr * s   # soft reset
            s = SpikeFn.apply(v - self.thr)
            u = self.beta_out * u + (1 - self.beta_out) * (s @ self.w_out.t())
            usum = usum + u
            cnt = cnt + s
            if return_spikes:
                spikes.append(s)
        out = usum / T
        if return_spikes:
            return out, cnt, torch.stack(spikes, 1)
        return out, cnt

    def fan_out(self):
        """Outgoing synapses per hidden neuron (non-zero recurrent + readout weights)."""
        return ((self.w_rec != 0).sum(0) + (self.w_out != 0).sum(0)).float()


def evaluate(model, x, y, bs=512):
    model.eval()
    correct = 0; cnt = torch.zeros(model.H); loss = 0.0
    with torch.no_grad():
        for i in range(0, len(y), bs):
            out, c = model(x[i:i + bs].float())
            correct += (out.argmax(1) == y[i:i + bs]).sum().item()
            loss += nn.functional.cross_entropy(out, y[i:i + bs], reduction="sum").item()
            cnt += c.sum(0)
    n = len(y)
    spikes_per_neuron = cnt / n                                # mean spikes per sample, per neuron
    synops = (spikes_per_neuron * model.fan_out()).sum().item()
    return dict(acc=100.0 * correct / n, loss=loss / n, spikes=spikes_per_neuron.sum().item(),
                synops=synops, spikes_per_neuron=spikes_per_neuron.numpy().round(3).tolist())


def make_tau(mode, H, seed, tau_h=0.02, gamma_k=3.0):
    """Time-constant initialisations for the baselines."""
    if mode in ("homogeneous", "learned_tau"):
        return np.full(H, tau_h, np.float32)
    if mode == "random_heterogeneous":                          # Gamma(k, tau_h/k), as Perez-Nieves et al.
        rng = np.random.RandomState(10_000 + seed)
        return np.clip(rng.gamma(gamma_k, tau_h / gamma_k, H), TAU_MIN, TAU_MAX).astype(np.float32)
    raise ValueError(mode)


def channel_shift(x, max_shift, g):
    """Shift each sample by a random integer in [-max_shift, max_shift] input channels (zero fill)."""
    B, T, C = x.shape
    sh = torch.randint(-max_shift, max_shift + 1, (B, 1), generator=g)
    src = torch.arange(C).unsqueeze(0) - sh                    # (B, C) source channel for each target
    ok = (src >= 0) & (src < C)
    out = torch.gather(x, 2, src.clamp(0, C - 1).unsqueeze(1).expand(B, T, C))
    return out * ok.unsqueeze(1)


def train(cfg):
    """cfg keys: name, seed, shift (channel-shift augmentation, in input channels), order_seed, H, epochs, lr, bs, tau0 (float or list[H]), learn_mask (bool or list[H]),
    thr (float or list[H]), rate_penalty (lambda on mean firing rate), tau_lr_mult, save (path or None),
    init_from (state-dict path or None; weights only unless load_tau)."""
    torch.set_num_threads(cfg.get("threads", 1))
    seed = cfg["seed"]
    torch.manual_seed(seed); np.random.seed(seed)
    D = load_shd()
    H = cfg.get("H", 128)
    model = RSNN(H=H, tau0=cfg.get("tau0", 0.02), learn_mask=cfg.get("learn_mask", False),
                 thr=cfg.get("thr", 1.0), w_scale=cfg.get("w_scale", 1.0))
    if cfg.get("init_from"):
        sd = torch.load(cfg["init_from"])
        keep = ["w_in", "w_rec", "w_out"] + (["log_tau"] if cfg.get("load_tau") else [])
        model.load_state_dict({k: sd[k] for k in keep}, strict=False)
    lr = cfg.get("lr", 2e-3); epochs = cfg.get("epochs", 30); bs = cfg.get("bs", 128)
    opt = torch.optim.Adam([
        {"params": [model.w_in, model.w_rec, model.w_out], "lr": lr},
        {"params": [model.log_tau], "lr": lr * cfg.get("tau_lr_mult", 5.0)}])
    n = len(D["ytr"]); steps = epochs * math.ceil(n / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, lr * cfg.get("tau_lr_mult", 5.0)],
                                                total_steps=steps, pct_start=0.15)
    lam = cfg.get("rate_penalty", 0.0); x_T = D["xtr"].shape[1]
    g = torch.Generator().manual_seed(cfg.get("order_seed", seed))  # minibatch order
    hist = []; best = None; t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = D["xtr"][idx].float()
            if cfg.get("shift", 0) > 0:                         # augmentation: random shift along the channel axis
                xb = channel_shift(xb, cfg["shift"], g)
            out, cnt = model(xb)
            loss = nn.functional.cross_entropy(out, D["ytr"][idx])
            if lam > 0:
                loss = loss + lam * cnt.mean() / x_T  # lam x mean firing probability per step
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            with torch.no_grad():
                model.log_tau.clamp_(math.log(TAU_MIN), math.log(TAU_MAX))
        va = evaluate(model, D["xva"], D["yva"])
        hist.append(round(va["acc"], 2))
        if best is None or va["acc"] > best["val_acc"]:
            te = evaluate(model, D["xte"], D["yte"])
            best = dict(epoch=ep, val_acc=va["acc"], val_loss=va["loss"], test_acc=te["acc"],
                        test_spikes=te["spikes"], test_synops=te["synops"],
                        spikes_per_neuron=te["spikes_per_neuron"],
                        tau=model.tau().detach().numpy().round(5).tolist())
            if cfg.get("save"):
                os.makedirs(os.path.dirname(cfg["save"]), exist_ok=True)
                torch.save(model.state_dict(), cfg["save"])
    te_last = evaluate(model, D["xte"], D["yte"])
    res = {k: v for k, v in cfg.items() if k not in ("tau0", "learn_mask", "thr")}
    res.update(best)
    res.update(val_hist=hist, last_test_acc=te_last["acc"], n_learnable_tau=int(model.learn_mask.sum().item()),
               n_params=sum(p.numel() for p in [model.w_in, model.w_rec, model.w_out]),
               energy_nj=best["test_synops"] * E_SYNOP_PJ / 1000.0, train_seconds=round(time.time() - t0, 1))
    return res


def run_jobs(jobs, out_path, procs=4):
    """Run configs in parallel (1 thread each); append results to a JSON list (resumable by name+seed)."""
    import torch.multiprocessing as mp
    done = json.load(open(out_path)) if os.path.exists(out_path) else []
    seen = {(r["name"], r["seed"]) for r in done}
    jobs = [j for j in jobs if (j["name"], j["seed"]) not in seen]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ctx = mp.get_context("fork")
    with ctx.Pool(procs) as pool:
        for r in pool.imap_unordered(train, jobs):
            done.append(r)
            json.dump(done, open(out_path, "w"))
            print(f"{r['name']:32s} seed {r['seed']} val {r['val_acc']:.2f} test {r['test_acc']:.2f} "
                  f"spikes {r['test_spikes']:.0f} synops {r['test_synops']:.0f} ({r['train_seconds']}s)", flush=True)
    return done
