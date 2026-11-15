"""Phase-1 D3 baselines: 5 paired seeds, H=128.
homogeneous (tuned tau), random_heterogeneous (Gamma-sampled tau, fixed), learned_tau (all taus trained),
rate-penalty sweep on the homogeneous net, and a same-init / different-minibatch-order homogeneous rerun
(used by the diagnostic for rank stability across training runs)."""
import sys, snn
TAU_H = float(sys.argv[1]); LR = float(sys.argv[2])
SEEDS = range(5); H = 128; EP = 30
ck = lambda name, s: f"{snn.ROOT}/data/ckpt/d3_{name}_s{s}.pt"
jobs = []
for s in SEEDS:
    base = dict(seed=s, H=H, epochs=EP, lr=LR, tau_h=TAU_H)
    jobs.append(dict(base, name="homogeneous", tau0=TAU_H, save=ck("homogeneous", s)))
    jobs.append(dict(base, name="random_heterogeneous", tau0=snn.make_tau("random_heterogeneous", H, s, TAU_H).tolist()))
    jobs.append(dict(base, name="learned_tau", tau0=TAU_H, learn_mask=True))
    jobs.append(dict(base, name="homogeneous_reorder", tau0=TAU_H, order_seed=1000 + s, save=ck("homogeneous_reorder", s)))
    for lam in [1.0, 3.0, 10.0, 30.0]:
        jobs.append(dict(base, name=f"homogeneous_pen{lam:g}", tau0=TAU_H, rate_penalty=lam))
snn.run_jobs(jobs, f"{snn.ROOT}/results/phase1/d3_baselines.json", procs=int(sys.argv[3]))
