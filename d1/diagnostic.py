"""Phase-1 D1 diagnostic: why do 2-bit channels collapse?
(a) single-channel loss vs bits: does dL_i(b) follow c*4^-b down to 2 bits?
(b) superadditivity: dL(group quantized jointly) - sum of individual dL, by bits, for groups of channels
    within one layer, across layers, and for whole layers / the whole network.
Loss = cross-entropy on a fixed set of training images, eval mode, NO BN re-estimation (pure perturbation).
Usage: python diagnostic.py SEED [threads] [n_blocks]"""
import sys, json
import numpy as np, torch
import quant as Q
from run_baselines import get_float


def main(seed, threads=1, n=1, n_img=1000, n_single=8, n_groups=12, gsize=8):
    torch.set_num_threads(threads)
    model, D = get_float(seed, n, 0)
    tables = Q.quant_tables(model); S = Q.sensitivities(model, tables, D, seed=seed)
    names = list(tables.keys()); rng = np.random.RandomState(seed)
    idx = torch.randperm(len(D["ytr"]), generator=torch.Generator().manual_seed(999))[:n_img]
    x, y = D["xtr"][idx], D["ytr"][idx]
    base = Q.evaluate(model, x, y)[1]

    def dloss(sel, b):
        """sel: list of (layer, channel); those channels at b bits, everything else float."""
        alloc = {k: np.zeros(tables[k]["n_ch"], int) for k in names}
        for k, c in sel:
            alloc[k][c] = b
        Q.set_bits(model, tables, alloc)
        return Q.evaluate(model, x, y)[1] - base

    out = dict(seed=seed, base_loss=base, single=[], groups=[], layers=[], network=[])
    # (a) single channels, n_single per layer
    for k in names:
        for c in rng.choice(tables[k]["n_ch"], min(n_single, tables[k]["n_ch"]), replace=False):
            bi = Q.BITS
            out["single"].append(dict(layer=k, ch=int(c), dl=[dloss([(k, c)], b) for b in bi],
                                      pred_law=[float(0.5 * S[k]["c"][c] * 4.0 ** -b) for b in bi],
                                      pred_table=[float(0.5 * S[k]["table"][c, j]) for j in range(len(bi))]))
    # (b) groups of gsize channels: within one layer vs spread across layers
    for kind in ["within", "across"]:
        for gi in range(n_groups):
            if kind == "within":
                k = names[rng.choice([i for i, kk in enumerate(names) if tables[kk]["n_ch"] >= gsize])]
                sel = [(k, int(c)) for c in rng.choice(tables[k]["n_ch"], gsize, replace=False)]
            else:
                ks = rng.choice(len(names), gsize, replace=len(names) < gsize)
                sel = [(names[i], int(rng.randint(tables[names[i]]["n_ch"]))) for i in ks]
                sel = list(dict.fromkeys(sel))
            for b in [2, 3, 4, 6]:
                joint = dloss(sel, b); indiv = [dloss([s], b) for s in sel]
                out["groups"].append(dict(kind=kind, bits=b, layer=sel[0][0] if kind == "within" else None,
                                          joint=joint, sum_indiv=float(np.sum(indiv))))
    # whole layers jointly, and the whole network vs sum of layers
    for b in [2, 3, 4, 6]:
        per_layer = []
        for k in names:
            dl = dloss([(k, c) for c in range(tables[k]["n_ch"])], b)
            per_layer.append(dl)
            out["layers"].append(dict(layer=k, bits=b, joint=dl))
        allsel = [(k, c) for k in names for c in range(tables[k]["n_ch"])]
        out["network"].append(dict(bits=b, joint=dloss(allsel, b), sum_layers=float(np.sum(per_layer))))
        print(seed, "bits", b, out["network"][-1], flush=True)
    json.dump(out, open(f"{Q.ROOT}/results/phase1/d1_diagnostic_seed{seed}.json", "w"))


if __name__ == "__main__":
    main(*[int(v) for v in sys.argv[1:]])
