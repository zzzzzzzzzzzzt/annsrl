"""Time an approximate k-NN graph (k=8) build on GIST1M with faiss.

Two build methods:
  nndescent (default) -- IndexNNDescentFlat; add() builds the whole graph in one
                         shot, so the timing is a single number.
  ivf                 -- IndexIVFFlat; the graph comes from self-querying the
                         index, so the timing splits into train / add / search.

The graph is not saved. Because this machine is shared and heavily contended
(observed 6x swings on an identical 50k build), the build is repeated and the
MEDIAN is the reported number; every raw timing and the load average around it
are printed so the reader can judge how much to trust it.
"""
import argparse
import os
import statistics
import time

import faiss
import numpy as np

BASE = "/mnt/HDD1/ANNS/datasets/gist-960-euclidean/gist1M/gist1M_base.fvecs"

ap = argparse.ArgumentParser()
ap.add_argument("--method", choices=["nndescent", "ivf"], default="nndescent")
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--eval_n", type=int, default=1000, help="nodes for the exact recall check")
# NN-Descent
ap.add_argument("--nnd_L", type=int, default=58, help="candidate pool size during build")
ap.add_argument("--nnd_iter", type=int, default=10)
ap.add_argument("--nnd_S", type=int, default=10)
ap.add_argument("--nnd_R", type=int, default=100)
# IVFFlat
ap.add_argument("--nlist", type=int, default=1024)
ap.add_argument("--nprobe", type=int, default=16)
ap.add_argument("--niter", type=int, default=12)
ap.add_argument("--pts_per_centroid", type=int, default=128)
ap.add_argument("--batch", type=int, default=16384)
args = ap.parse_args()
K = args.k


def read_fvecs(path):
    raw = np.memmap(path, dtype="int32", mode="r")
    dim = int(raw[0])
    rec = dim + 1
    n = raw.size // rec
    assert raw.size % rec == 0, "file size is not a multiple of the record size"
    return np.ascontiguousarray(raw.reshape(n, rec)[:, 1:].view("float32")), n, dim


def to_adjacency(cand, n, k):
    """[n, >=k+1] candidate ids -> [n, k] neighbours with self removed.

    The fast path is a plain slice. NN-Descent's final_graph is already
    self-free and distance-sorted (verified on a 50k subset: building at K=32
    and truncating to 8 still scores recall@8 = 0.9975, which is impossible if
    the rows were unordered), so the slice is exact there. IVF returns the self
    match at rank 0, so it always takes the masking path.
    """
    if cand.shape[1] >= k and not (cand[:, :k] == np.arange(n)[:, None]).any():
        return np.ascontiguousarray(cand[:, :k])
    assert cand.shape[1] > k, f"need > {k} candidates per row to drop the self match"
    keep = cand[:, : k + 1] != np.arange(n)[:, None]
    keep[np.cumsum(keep, axis=1) > k] = False
    keep[keep.sum(1) < k, k] = True          # self not returned -> drop last column
    return cand[:, : k + 1][keep].reshape(n, k)


t0 = time.perf_counter()
xb, n, dim = read_fvecs(BASE)
t_load = time.perf_counter() - t0

print("=" * 66)
print("dataset      :", BASE)
print(f"num points   : {n:,}")
print(f"dimension    : {dim}")
print(f"dtype        : {xb.dtype}  ({xb.nbytes / 2**30:.2f} GiB in RAM)")
print(f"metric       : L2")
if args.method == "nndescent":
    print(f"index        : NNDescent  K={K}  L={args.nnd_L}  iter={args.nnd_iter}"
          f"  S={args.nnd_S}  R={args.nnd_R}")
else:
    print(f"index        : IVFFlat  nlist={args.nlist}  nprobe={args.nprobe}")
print(f"k            : {K}")
print(f"threads      : {faiss.omp_get_max_threads()}  (cores: {os.cpu_count()})")
print(f"repeats      : {args.repeats}  (median is the reported number)")
print(f"load time    : {t_load:.1f} s  (I/O, not counted as build time)")
print("=" * 66, flush=True)


def build_nndescent():
    index = faiss.IndexNNDescentFlat(dim, K, faiss.METRIC_L2)
    nd = index.nndescent
    nd.L, nd.iter, nd.S, nd.R = args.nnd_L, args.nnd_iter, args.nnd_S, args.nnd_R
    t = time.perf_counter()
    index.add(xb)                       # one call builds the entire graph
    dt = time.perf_counter() - t
    cand = faiss.vector_to_array(nd.final_graph).reshape(n, -1)
    return dt, {"build": dt}, cand


def build_ivf():
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, args.nlist, faiss.METRIC_L2)
    index.cp.niter = args.niter
    index.cp.max_points_per_centroid = args.pts_per_centroid

    n_train = min(n, args.nlist * args.pts_per_centroid)
    xt = np.ascontiguousarray(xb[np.sort(np.random.default_rng(0).choice(n, n_train, replace=False))])
    t = time.perf_counter()
    index.train(xt)
    t_train = time.perf_counter() - t
    del xt

    t = time.perf_counter()
    index.add(xb)
    t_add = time.perf_counter() - t

    index.nprobe = args.nprobe
    cand = np.empty((n, K + 1), dtype="int32")
    t = time.perf_counter()
    for lo in range(0, n, args.batch):
        hi = min(lo + args.batch, n)
        _, cand[lo:hi] = index.search(xb[lo:hi], K + 1)
        el = time.perf_counter() - t
        print(f"      {hi:>9,}/{n:,}  elapsed {el:8.1f} s"
              f"  eta {el / hi * (n - hi):8.1f} s  ({hi / el:,.0f} pts/s)", flush=True)
    t_search = time.perf_counter() - t
    return t_train + t_add + t_search, \
        {"train": t_train, "add": t_add, "search": t_search}, cand


build = build_nndescent if args.method == "nndescent" else build_ivf
times, nbr = [], None
for r in range(args.repeats):
    load_before = os.getloadavg()[0]
    total, stages, cand = build()
    load_after = os.getloadavg()[0]
    nbr = to_adjacency(cand, n, K)
    del cand
    times.append(total)
    stage_str = "  ".join(f"{k}={v:.1f}s" for k, v in stages.items())
    print(f"\nrun {r + 1}/{args.repeats}: TOTAL {total:8.1f} s  ({total / 60:.2f} min)"
          f"   [{stage_str}]   load {load_before:.0f} -> {load_after:.0f}", flush=True)

median = statistics.median(times)
print("\n" + "-" * 66)
print(f"raw timings   : {', '.join(f'{t:.1f}s' for t in times)}")
print(f"MEDIAN BUILD  : {median:.1f} s  ({median / 60:.2f} min)   <- reported number")
print(f"spread        : min {min(times):.1f}s  max {max(times):.1f}s"
      f"  (max/min = {max(times) / min(times):.2f}x)")
print(f"throughput    : {n / median:,.0f} nodes/s")
print(f"graph         : {n:,} x {K} = {n * K:,} directed edges  (not saved)")
print("-" * 66, flush=True)

# ---- quality: exact brute force on a random sample, via numpy BLAS ----
print(f"\nexact recall@{K} check on {args.eval_n} random nodes ...", flush=True)
rng = np.random.default_rng(0)
qids = np.sort(rng.choice(n, args.eval_n, replace=False))
xq = np.ascontiguousarray(xb[qids])
sq_b = np.einsum("ij,ij->i", xb, xb)

t0 = time.perf_counter()
exact = np.empty((args.eval_n, K), dtype="int32")
CH = 250
for lo in range(0, args.eval_n, CH):
    hi = min(lo + CH, args.eval_n)
    D = xq[lo:hi] @ xb.T
    D *= -2.0
    D += sq_b
    D[np.arange(hi - lo), qids[lo:hi]] = np.inf          # exclude self
    exact[lo:hi] = np.argpartition(D, K, axis=1)[:, :K]
t_exact = time.perf_counter() - t0

hits = sum(len(set(exact[i].tolist()) & set(nbr[qids[i]].tolist())) for i in range(args.eval_n))
print(f"exact bruteforce for {args.eval_n} nodes took {t_exact:.1f} s")
print(f"graph recall@{K}  : {hits / (args.eval_n * K):.4f}")
print(f"shape         : {nbr.shape}")
print(f"self-loops    : {(nbr == np.arange(n)[:, None]).sum()}")
print(f"node 0 nbrs   : {nbr[0].tolist()}")
