# /// script
# requires-python = ">=3.11"
# ///
"""Compute-DAG placement model (docs/architecture/16-compute-dag.md).

Prices placement moves for the training narrative before a pod session is
spent on them: nodes carry measured host/device costs, boundary edges carry
bytes, and a placement's makespan estimate is

    sum(node cost at its placement) + sum(boundary-edge bytes / bandwidth)

minus declared overlap. This is a pricing model, not a simulator — it exists
to kill sub-1s bets and to rank the survivors; same-pod measurement remains
the gate (fleet spread between identical L40S pods measured at ~25%).

Constants below are the 16M x 100 x 255 cell (cuda_depthwise) on the US-NC-1
L40S, 2026-07-15, post decisions 54/49/38/72 (fit 15.70s; the PR #35-era
baseline these constants replaced was 39.4s). Refresh them from the profile
lines of any same-pod run: grow-profile / cuda-upload-decomp /
cuda-round-decomp / ingest-profile. `None` means the node has no
implementation on that side today; pricing a move that needs one is exactly
the point.

    uv run scripts/dag_model.py            # evaluate the canned placements
    uv run scripts/dag_model.py --floor    # show the all-device floor math
"""
import argparse
from dataclasses import dataclass

GB = 1e9

# Effective transfer rates, measured on healthy PCIe4 hosts. D2H pageable
# joined once the fin_d2h counter landed (decision 54's instrumentation);
# PR #35's lesson stands: no edge gets priced from an undecomposed line.
BW = {
    "h2d": 14e9,        # pageable H2D, measured via gh_upload (0.88s / 12.8GB)
    "h2d_pinned": 24e9, # pinned, from ensure_dataset-style staging
    "d2h": 13.6e9,      # pageable D2H, measured via fin_d2h (0.94s / 12.8GB)
}


@dataclass
class Node:
    name: str
    host_s: float | None      # measured cost when placed on host
    dev_s: float | None       # measured cost when placed on device
    pinned: str | None = None # "host"/"device" when a constraint fixes placement
    note: str = ""


@dataclass
class Edge:
    src: str
    dst: str
    bytes_per_fit: float      # total bytes across the whole fit if it crosses
    bw: str = "h2d"
    overlapped: float = 0.0   # fraction hidden behind concurrent compute
    note: str = ""


# ---- the graph, post-decision-72 constants -----------------------------------
# The old "find 7.62" node was decision-62 misattribution: the find kernel is
# 0.13s and the bulk was the histogram build draining at find's sync. The
# cuda-round-decomp event pairs now price the build directly (hist_build), and
# the old 12.1s "unattributed" node is gone: fit-profile closes conservation
# (grow 14.05 + ingest 1.47 + objective/sample/score 0.51 ~= fit 15.70).

NODES = [
    Node("mapper_fit", 0.45, None, pinned="host", note="seeded RNG subsample; model-changing to move"),
    Node("bin", 4.65, 0.55, note="decision 54 landed: dbin kernel + streamed upload (host cost kept for the counterfactual)"),
    Node("gradients_scores", 0.51, None, pinned="host", note="objective+sample+score fit-profile lines"),
    Node("hist_build", None, 7.90, pinned="device", note="adv_hist+root_hist events; the round's dominant cost"),
    Node("root_sums", None, 1.02, pinned="device", note="decision-72 two-pass reduce, replaces the host loop"),
    Node("find", None, 0.13, pinned="device", note="find_kern events; decision 62's honest number"),
    Node("partition", None, 1.90, pinned="device"),
    Node("epilogue_map", None, 0.33, pinned="device", note="fin_map events"),
    Node("split_decisions", 0.03, None, pinned="host", note="doc 12 control plane; per-level sync floor"),
]

# Boundary edges and what they cost IF their endpoints straddle the cut.
EDGES = [
    Edge("raw_X", "bin", 6.4 * GB, bw="h2d_pinned", overlapped=0.3,
         note="decision 54: streamed raw upload, overlaps bin kernel"),
    Edge("bin", "hist_build", 1.6 * GB, bw="h2d_pinned",
         note="binned matrix upload — exists only when bin is on host"),
    Edge("gradients_scores", "hist_build", 12.8 * GB,
         note="gh per tree x100 — the decision-52 edge (measured 0.88s)"),
    Edge("epilogue_map", "gradients_scores", 12.8 * GB, bw="d2h",
         note="values+leaf_ids per tree x100 D2H — fin_d2h measured 0.94s"),
]

MEASURED_FIT_S = 15.70


def edge_cost(e: Edge) -> float:
    return e.bytes_per_fit / BW[e.bw] * (1.0 - e.overlapped)


def evaluate(placement: dict[str, str]) -> tuple[float, list[str]]:
    """Makespan estimate + line items for a {node: 'host'|'device'} map."""
    total, lines = 0.0, []
    by_name = {n.name: n for n in NODES}
    for n in NODES:
        side = placement.get(n.name, n.pinned or "host")
        cost = n.host_s if side == "host" else n.dev_s
        if cost is None:
            lines.append(f"  !! {n.name} has no measured {side} cost — decompose/prototype first")
            continue
        total += cost
        lines.append(f"  {n.name:18s} {side:6s} {cost:5.2f}s")
    for e in EDGES:
        src = placement.get(e.src, by_name[e.src].pinned if e.src in by_name else "host") \
            if e.src in by_name else "host"
        dst = placement.get(e.dst, by_name[e.dst].pinned or "host")
        if src != dst:
            c = edge_cost(e)
            total += c
            lines.append(f"  {e.src}->{e.dst:18s} xfer  {c:5.2f}s  ({e.bytes_per_fit/GB:.1f}GB {e.bw})")
    return total, lines


PLACEMENTS = {
    "status quo (post-53)": {"bin": "host"},
    "decision 54 (device bin)": {"bin": "device"},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", action="store_true")
    args = ap.parse_args()

    for name, pl in PLACEMENTS.items():
        total, lines = evaluate(pl)
        print(f"\n== {name}: est {total:.1f}s (measured fit {MEASURED_FIT_S}s)")
        print("\n".join(lines))

    if args.floor:
        dev = sum(n.dev_s for n in NODES if n.dev_s is not None)
        pinned_host = sum(n.host_s for n in NODES if n.pinned == "host" and n.host_s)
        ingest = 6.4 * GB / BW["h2d_pinned"]
        print(f"\n== all-device floor: ingest {ingest:.1f} + device {dev:.1f} "
              f"+ pinned-host {pinned_host:.1f} = {ingest + dev + pinned_host:.1f}s "
              f"(residual vs measured = per-level syncs + un-evented staging)")


if __name__ == "__main__":
    main()
