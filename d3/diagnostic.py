"""Phase-1 D3 diagnostic (run BEFORE any allocator): on trained homogeneous nets, how is the
sensitivity of the loss to each neuron's membrane time constant distributed?

Scores per neuron i (theta_i = log tau_i):
  fisher_i  = mean over mini-batches (bs=16) of (dL/dtheta_i)^2      [Fisher-style]
  absgrad_i = |mean dL/dtheta_i|                                    [first-order]
  fd_i      = mean of loss change when tau_i is x2 and x0.5         [finite perturbation, ground truth]
Reported: Gini, top-10% share (reference: same for firing rate and for a flat vector), rank stability
(Spearman) across disjoint data halves, across a same-init/different-minibatch-order training run,
agreement fisher vs finite perturbation, and correlation with simple neuron features."""
import sys, json, math
import numpy as np, torch, torch.nn as nn
from scipy.stats import spearmanr
import snn


def gini(v):
    v = np.sort(np.abs(np.asarray(v, float))); n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum() + 1e-300))


def top_share(v, frac=0.1):
    v = np.sort(np.abs(np.asarray(v, float)))[::-1]
    return float(v[: max(1, int(round(frac * len(v))))].sum() / (v.sum() + 1e-300))


def load(name, s, H=128, tau_h=0.02):
    m = snn.RSNN(H=H, tau0=tau_h, learn_mask=True)
    m.load_state_dict(torch.load(f"{snn.ROOT}/data/ckpt/d3_{name}_s{s}.pt"), strict=False)
    m.learn_mask.fill_(1.0); m.eval()
    return m


def grad_scores(m, x, y, bs=16):
    sq = torch.zeros(m.H); mean = torch.zeros(m.H); nb = 0
    for i in range(0, len(y), bs):
        out, _ = m(x[i:i + bs].float())
        g, = torch.autograd.grad(nn.functional.cross_entropy(out, y[i:i + bs]), m.log_tau)
        sq += g ** 2; mean += g; nb += 1
    return (sq / nb).numpy(), (mean / nb).abs().numpy()


@torch.no_grad()
def fd_scores(m, x, y):
    def loss():
        return nn.functional.cross_entropy(m(x.float())[0], y).item()
    base = loss(); out = np.zeros(m.H)
    for i in range(m.H):
        old = m.log_tau[i].item(); d = []
        for f in (2.0, 0.5):
            m.log_tau[i] = old + math.log(f); d.append(loss() - base)
        m.log_tau[i] = old; out[i] = np.mean(d)
    return out


def one(args):
    tau_h, s = args
    torch.set_num_threads(1)
    D = snn.load_shd()
    perm = torch.randperm(len(D["ytr"]), generator=torch.Generator().manual_seed(7))
    A, B, C = perm[:2048], perm[2048:4096], perm[4096:5120]
    if True:
        m = load("homogeneous", s, tau_h=tau_h)
        fA, gA = grad_scores(m, D["xtr"][A], D["ytr"][A]); fB, gB = grad_scores(m, D["xtr"][B], D["ytr"][B])
        fV, gV = grad_scores(m, D["xva"], D["yva"])
        fd = fd_scores(m, D["xtr"][C], D["ytr"][C])
        ev = snn.evaluate(m, D["xva"], D["yva"]); rate = np.array(ev["spikes_per_neuron"])
        m2 = load("homogeneous_reorder", s, tau_h=tau_h)
        f2, _ = grad_scores(m2, D["xtr"][A], D["ytr"][A])
        sp = lambda a, b: float(spearmanr(a, b)[0])
        f = 0.5 * (fA + fB)
        r = dict(seed=s, gini_fisher=gini(f), top10_fisher=top_share(f), gini_absgrad=gini(gA), top10_absgrad=top_share(gA),
                 gini_fd=gini(fd), top10_fd=top_share(fd), frac_fd_negative=float((fd < 0).mean()),
                 gini_rate=gini(rate), top10_rate=top_share(rate),
                 spearman_halves_fisher=sp(fA, fB), spearman_halves_absgrad=sp(gA, gB), spearman_train_val_fisher=sp(f, fV),
                 spearman_reorder_run_fisher=sp(f, f2), spearman_fisher_fd=sp(f, np.abs(fd)), spearman_fisher_fd_signed=sp(f, fd),
                 spearman_fisher_rate=sp(f, rate),
                 spearman_fisher_outnorm=sp(f, m.w_out.detach().norm(dim=0).numpy()),
                 spearman_fisher_recoutnorm=sp(f, m.w_rec.detach().norm(dim=0).numpy()),
                 top10_overlap_halves=len(set(np.argsort(-fA)[:13]) & set(np.argsort(-fB)[:13])) / 13.0,
                 top10_overlap_reorder=len(set(np.argsort(-f)[:13]) & set(np.argsort(-f2)[:13])) / 13.0,
                 fisher=f.tolist(), fd=fd.tolist(), rate=rate.tolist())
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, list)}, flush=True)
        return r


def main(tau_h, seeds=range(5), procs=3):
    import torch.multiprocessing as mp
    with mp.get_context("fork").Pool(procs) as pool:
        rows = sorted(pool.map(one, [(tau_h, s) for s in seeds]), key=lambda r: r["seed"])
    json.dump(rows, open(f"{snn.ROOT}/results/phase1/d3_diagnostic.json", "w"))


if __name__ == "__main__":
    main(float(sys.argv[1]))
