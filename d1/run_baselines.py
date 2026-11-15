"""Phase-1 D1 baselines. Usage: python run_baselines.py SEED [threads] [n_blocks] [epochs]
Trains (or reloads) the float net for SEED, then evaluates every allocator at every budget."""
import sys, os, json, time
import numpy as np, torch
import quant as Q

BUDGETS = [2.0, 2.25, 2.5, 3.0]


def get_float(seed, n, epochs, log=print):
    path = f"{Q.ROOT}/data/ckpt/d1_float_n{n}_s{seed}.pt"
    D = Q.load_cifar()
    if os.path.exists(path):
        model = Q.ResNet(n); model.load_state_dict(torch.load(path))
    else:
        model = Q.train_float(seed, n=n, epochs=epochs, log=log)
        torch.save(model.state_dict(), path)
    return model, D


def main(seed, threads=1, n=1, epochs=15, out_dir="results/phase1", ft_steps=100):
    torch.set_num_threads(threads)
    model, D = get_float(seed, n, epochs)
    fv, ft = Q.evaluate(model, D["xva"], D["yva"])[0], Q.evaluate(model, D["xte"], D["yte"])[0]
    print(f"seed {seed} float val {fv:.2f} test {ft:.2f}", flush=True)
    tables = Q.quant_tables(model); S = Q.sensitivities(model, tables, D, seed=seed)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    rows = [dict(seed=seed, alloc="float", budget=32, rtn_val=fv, rtn_test=ft, rtn_bn_val=fv, rtn_bn_test=ft,
                 ft_val=fv, ft_test=ft, avg_bits=32, n_params=sum(t["n_ch"] * t["fan_in"] for t in tables.values()))]
    for budget in BUDGETS:
        for name, fn in Q.ALLOCATORS.items():
            if name == "uniform" and not float(budget).is_integer():
                continue
            t0 = time.time()
            alloc = fn(tables, S, budget)
            r = Q.eval_alloc(state, tables, alloc, D, seed, n=n, ft_steps=ft_steps)
            allb = np.concatenate([np.asarray(v) for v in alloc.values()])
            r.update(seed=seed, alloc=name, budget=budget, hist={int(b): int((allb == b).sum()) for b in Q.BITS},
                     layer_mean_bits={k: round(float(np.mean(v)), 2) for k, v in alloc.items()})
            rows.append(r)
            print(f"seed {seed} B={budget} {name:10s} avg {r['avg_bits']:.3f} rtn {r['rtn_test']:.2f} "
                  f"rtn_bn {r['rtn_bn_test']:.2f} ft {r['ft_test']:.2f} ({time.time()-t0:.0f}s)", flush=True)
            json.dump(rows, open(f"{Q.ROOT}/{out_dir}/d1_baselines_seed{seed}.json", "w"))


if __name__ == "__main__":
    a = [int(v) for v in sys.argv[1:]]
    main(*a)
