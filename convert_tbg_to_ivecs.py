"""Convert .TBG graph files to plain ivecs format.

A .TBG file uses the same layout as .nsg: an 8-byte header
(width, enterpoint_node) followed by ivecs-style adjacency records
(degree, neighbor_0, ..., neighbor_{degree-1}) for each node.

Dropping the header yields a valid ivecs file readable by
lib.utils.read_ivecs / read_edges. The enterpoint_node is printed so it
can be recorded separately -- ivecs has nowhere to store it.
"""
import os
import sys
from struct import unpack

import numpy as np

DEFAULT_FILES = [
    "data/DEEP100K/DEEP100K_R24_C500_L300.TBG",
    "data/SIFT100K/SIFT100K_R24_C500_L300.TBG",
]


def convert(src, dst=None):
    dst = dst or os.path.splitext(src)[0] + ".ivecs"

    with open(src, "rb") as f:
        width, enterpoint = unpack("<II", f.read(8))
        body = f.read()

    with open(dst, "wb") as f:
        f.write(body)

    # Verify the result parses as ivecs and matches the source adjacency.
    edges = np.frombuffer(body, dtype=np.int32)
    n_nodes = 0
    i = 0
    while i < edges.size:
        deg = edges[i]
        i += 1 + deg
        n_nodes += 1
    assert i == edges.size, f"{dst}: trailing bytes, not a valid ivecs file"

    print(f"{src} -> {dst}")
    print(f"  width={width} enterpoint_node={enterpoint} nodes={n_nodes}")
    return dst


if __name__ == "__main__":
    for path in sys.argv[1:] or DEFAULT_FILES:
        convert(path)
