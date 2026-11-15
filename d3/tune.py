"""Tune the homogeneous baseline on the validation split (seed 100, not one of the paired seeds)."""
import sys, snn
jobs = []
for tau in [0.01, 0.02, 0.04, 0.08]:
    for lr in [2e-3, 5e-3]:
        jobs.append(dict(name=f"tune_hom_tau{int(tau*1000)}ms_lr{lr}", seed=100, tau0=tau, lr=lr, epochs=30))
snn.run_jobs(jobs, f"{snn.ROOT}/results/phase1/d3_tune.json")
