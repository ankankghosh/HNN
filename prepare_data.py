"""Convert raw downloads into compact .npy arrays under data/ (not committed).

SHD:     https://zenkelab.org/datasets/shd_{train,test}.h5.gz  -> data/shd/
CIFAR10: https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz (PNG mirror) -> data/cifar/
"""
import os, sys, glob
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def shd(dt=0.010, T=100, n_in=140):
    import h5py
    for split in ["train", "test"]:
        f = h5py.File(f"{ROOT}/data/shd/shd_{split}.h5", "r")
        times, units, labels = f["spikes"]["times"], f["spikes"]["units"], f["labels"][:]
        N = len(labels)
        X = np.zeros((N, T, n_in), dtype=np.uint8)
        g = 700 // n_in
        for i in range(N):
            t = np.floor(times[i] / dt).astype(int)
            u = units[i].astype(int) // g
            keep = t < T
            np.add.at(X[i], (t[keep], u[keep]), 1)
        np.save(f"{ROOT}/data/shd/{split}_x.npy", X)
        np.save(f"{ROOT}/data/shd/{split}_y.npy", labels.astype(np.int64))
        np.save(f"{ROOT}/data/shd/{split}_speaker.npy", f["extra"]["speaker"][:])
        print(split, X.shape, "mean spikes/sample", X.sum() / N, "max count", X.max())


def cifar():
    from PIL import Image
    classes = sorted(os.listdir(f"{ROOT}/data/cifar/cifar10/train"))
    for split in ["train", "test"]:
        xs, ys = [], []
        for ci, c in enumerate(classes):
            for p in sorted(glob.glob(f"{ROOT}/data/cifar/cifar10/{split}/{c}/*.png")):
                xs.append(np.asarray(Image.open(p).convert("RGB")))
                ys.append(ci)
        X = np.stack(xs).astype(np.uint8)
        np.save(f"{ROOT}/data/cifar/{split}_x.npy", X)
        np.save(f"{ROOT}/data/cifar/{split}_y.npy", np.array(ys, dtype=np.int64))
        print(split, X.shape)


if __name__ == "__main__":
    which = sys.argv[1:] or ["shd", "cifar"]
    if "shd" in which:
        shd()
    if "cifar" in which:
        cifar()
