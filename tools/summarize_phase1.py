"""Print the Phase-1 tables from results/phase1/*.json (markdown)."""
import json, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from stats import fmt, fmt_paired, mean_se
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/results/phase1"


def d3():
    rows = json.load(open(f"{R}/d3_baselines.json"))
    names = sorted({r["name"] for r in rows}, key=lambda n: (n.startswith("homogeneous_"), n))
    by = {n: {r["seed"]: r for r in rows if r["name"] == n} for n in names}
    print("\n### D3 baselines (SHD, H=128, test accuracy at best-validation epoch)\n")
    print("| model | seeds | val acc | test acc | spikes/sample | SynOps/sample | energy nJ | test at last epoch |")
    print("|---|---|---|---|---|---|---|---|")
    for n in names:
        g = lambda k: [r[k] for r in by[n].values()]
        print(f"| {n} | {len(by[n])} | {fmt(g('val_acc'))} | {fmt(g('test_acc'))} | {fmt(g('test_spikes'),0)} | "
              f"{fmt(g('test_synops'),0)} | {fmt(g('energy_nj'),1)} | {fmt(g('last_test_acc'))} |")
    acc = lambda n: {s: r["test_acc"] for s, r in by[n].items()}
    print("\nPaired test-accuracy differences (same seed = same weight init and minibatch order):\n")
    for a, b in [("learned_tau", "homogeneous"), ("random_heterogeneous", "homogeneous"), ("learned_tau", "random_heterogeneous")]:
        if a in by and b in by:
            print(f"- {a} - {b}: {fmt_paired(acc(a), acc(b))}")
    if "learned_tau" in by:
        t = np.array([r["tau"] for r in by["learned_tau"].values()]) * 1000
        print(f"\nLearned time constants (ms): mean {t.mean():.1f}, min {t.min():.1f}, max {t.max():.1f}, "
              f"10/50/90 percentiles {np.percentile(t,10):.1f}/{np.percentile(t,50):.1f}/{np.percentile(t,90):.1f}")


def d3_diag():
    rows = json.load(open(f"{R}/d3_diagnostic.json"))
    print("\n### D3 diagnostic: per-neuron sensitivity to the time constant (homogeneous nets)\n")
    keys = [k for k in rows[0] if not isinstance(rows[0][k], list) and k != "seed"]
    print("| quantity | mean +/- s.e. over seeds | per seed |"); print("|---|---|---|")
    for k in keys:
        v = [r[k] for r in rows]
        print(f"| {k} | {fmt(v,3)} | {', '.join(f'{x:.2f}' for x in v)} |")
    print("\nReference: a flat vector has Gini 0 and top-10% share 0.10; |N(0,1)| samples have Gini 0.41, "
          "chi-square(1) samples (squared Gaussian) have Gini 0.64 and top-10% share about 0.44.")


def d1():
    rows = [r for f in sorted(glob.glob(f"{R}/d1_baselines_seed*.json")) for r in json.load(open(f))]
    seeds = sorted({r["seed"] for r in rows})
    fl = [r for r in rows if r["alloc"] == "float"]
    print(f"\n### D1 baselines (ResNet-8, CIFAR-10, weight-only, seeds {seeds})\n")
    print(f"Float test accuracy: {fmt([r['rtn_test'] for r in fl])}; parameters quantized: {fl[0]['n_params']}\n")
    print("| budget | allocator | avg bits | RTN | RTN + BN re-est. | short fine-tune | paired vs per_layer (RTN+BN) | paired vs per_layer (fine-tune) |")
    print("|---|---|---|---|---|---|---|---|")
    for B in sorted({r["budget"] for r in rows if r["alloc"] != "float"}):
        sub = [r for r in rows if r["budget"] == B]
        ref = {r["seed"]: r for r in sub if r["alloc"] == "per_layer"}
        for a in ["uniform", "per_layer", "hawq_style", "wf", "baq", "wf_global"]:
            rs = {r["seed"]: r for r in sub if r["alloc"] == a}
            if not rs:
                continue
            g = lambda k: [r[k] for r in rs.values()]
            pc = lambda k: fmt_paired({s: r[k] for s, r in rs.items()}, {s: r[k] for s, r in ref.items()}) if a != "per_layer" else "-"
            print(f"| {B} | {a} | {np.mean(g('avg_bits')):.3f} | {fmt(g('rtn_test'))} | {fmt(g('rtn_bn_test'))} | {fmt(g('ft_test'))} | {pc('rtn_bn_test')} | {pc('ft_test')} |")
    # uniform 3-bit vs mixed allocations
    u3 = {r["seed"]: r for r in rows if r["alloc"] == "uniform" and r["budget"] == 3.0}
    if u3:
        print("\nUniform 3-bit against each mixed allocation at budget 3.0 (uniform - mixed):\n")
        for a in ["per_layer", "hawq_style", "wf", "baq", "wf_global"]:
            rs = {r["seed"]: r for r in rows if r["alloc"] == a and r["budget"] == 3.0}
            for k in ["rtn_bn_test", "ft_test"]:
                print(f"- vs {a} [{k}]: {fmt_paired({s: r[k] for s, r in u3.items()}, {s: r[k] for s, r in rs.items()})}")


def d1_diag():
    rows = [json.load(open(f)) for f in sorted(glob.glob(f"{R}/d1_diagnostic_seed*.json"))]
    bits = list(range(2, 9))
    print(f"\n### D1 diagnostic (seeds {[r['seed'] for r in rows]})\n")
    print("(a) Single-channel loss increase vs bits. Law predicts a ratio dL(b)/dL(b+1) = 4.\n")
    print("| bits b | median dL(b) | mean dL(b) | fraction dL<=0 | ratio of means dL(b)/dL(b+1) | mean measured / Fisher-law prediction | mean measured / Fisher-table prediction |")
    print("|---|---|---|---|---|---|---|")
    dl = np.array([s["dl"] for r in rows for s in r["single"]]); law = np.array([s["pred_law"] for r in rows for s in r["single"]])
    tab = np.array([s["pred_table"] for r in rows for s in r["single"]])
    for j, b in enumerate(bits):
        ratio = dl[:, j].mean() / dl[:, j + 1].mean() if j + 1 < len(bits) else float("nan")
        print(f"| {b} | {np.median(dl[:, j]):.2e} | {dl[:, j].mean():.2e} | {(dl[:, j] <= 0).mean():.2f} | {ratio:.2f} | "
              f"{dl[:, j].mean() / law[:, j].mean():.2f} | {dl[:, j].mean() / tab[:, j].mean():.2f} |")
    print("\n(b) Superadditivity: loss with a group quantized jointly vs sum of the members quantized alone.\n")
    print("| grouping | bits | mean joint dL | mean sum of individual dL | ratio joint/sum (of means) | per-seed ratios |")
    print("|---|---|---|---|---|---|")
    for kind in ["within", "across"]:
        for b in [2, 3, 4, 6]:
            per = []
            for r in rows:
                g = [x for x in r["groups"] if x["kind"] == kind and x["bits"] == b]
                per.append((np.mean([x["joint"] for x in g]), np.mean([x["sum_indiv"] for x in g])))
            per = np.array(per)
            print(f"| 8 channels, {kind} layer(s) | {b} | {per[:,0].mean():.2e} | {per[:,1].mean():.2e} | {per[:,0].mean()/per[:,1].mean():.2f} | "
                  f"{', '.join(f'{a/c:.2f}' for a, c in per)} |")
    for b in [2, 3, 4, 6]:
        per = np.array([[x["joint"], x["sum_layers"]] for r in rows for x in r["network"] if x["bits"] == b])
        print(f"| whole network vs sum of whole layers | {b} | {per[:,0].mean():.2e} | {per[:,1].mean():.2e} | {per[:,0].mean()/per[:,1].mean():.2f} | "
              f"{', '.join(f'{a/c:.2f}' for a, c in per)} |")
    print("\nWhole-layer loss increase by layer (mean over seeds):\n")
    names = list(dict.fromkeys(x["layer"] for x in rows[0]["layers"]))
    print("| layer | " + " | ".join(f"{b} bits" for b in [2, 3, 4, 6]) + " |"); print("|---|---|---|---|---|")
    for n in names:
        print(f"| {n} | " + " | ".join(f"{np.mean([x['joint'] for r in rows for x in r['layers'] if x['layer']==n and x['bits']==b]):.3f}" for b in [2, 3, 4, 6]) + " |")


if __name__ == "__main__":
    for f in (sys.argv[1:] or ["d3", "d3_diag", "d1", "d1_diag"]):
        try:
            globals()[f]()
        except FileNotFoundError as e:
            print(f"[{f}: results not available yet: {e}]")
