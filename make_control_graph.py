"""Save learning-free control graphs as dynamic_edges snapshots.

Needed so c1_reshuffle.py can run its placebo: reshuffling the long-edge targets of a
graph whose long edges were placed by numpy MUST be a null. If it is not, C1's -0.014
on the policy graphs is an artifact of the reshuffle rather than learned information.
"""
import argparse, os, torch, lib_control
from probe_visits import NJ, EF, K, DCS_BUDGET, build
import lib

ap = argparse.ArgumentParser()
ap.add_argument('--dose', type=float, default=None)
ap.add_argument('--n_edges', type=int, default=None)
ap.add_argument('--seed', type=int, default=2)
ap.add_argument('--out', required=True)
a = ap.parse_args()

graph = build('knn')
h = lib.GraphEditHNSW(graph, ef=EF, k=K, n_jobs=NJ, max_trajectory=400, max_dcs=DCS_BUDGET)
edges, k = lib_control.rewire(h.adj, h.service_labels['pad'],
                             dose=a.dose, n_edges=a.n_edges, seed=a.seed)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
torch.save(edges, a.out)
print('wrote %s  (%d edges rewired, seed %d)' % (a.out, k, a.seed))
