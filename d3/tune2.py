"""Second tuning pass (seeds 100/101, validation only): does channel-shift augmentation or a larger
time-constant learning rate change the homogeneous vs learned_tau picture?"""
import snn
jobs = []
for seed in [100, 101]:
    for shift in [0, 4, 8]:
        for mode, mult in [("homogeneous", 5), ("learned_tau", 5), ("learned_tau", 30)]:
            if shift == 0 and mode == "homogeneous" and seed == 100:
                continue
            jobs.append(dict(name=f"tune2_{mode}_shift{shift}_taulr{mult}", seed=seed, tau0=0.02, lr=5e-3, epochs=30,
                             shift=shift, learn_mask=(mode == "learned_tau"), tau_lr_mult=mult))
snn.run_jobs(jobs, f"{snn.ROOT}/results/phase1/d3_tune2.json", procs=3)
