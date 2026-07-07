# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=1.26", "pandas>=2.2", "matplotlib>=3.8", "tabulate>=0.9"]
# ///
"""Analyze benchmarks/results/scaling.jsonl (scripts/bench_scaling.py output):
fits power-law exponents per (host, variant, axis), draws log-log throughput
curves, and writes benchmarks/results/scaling.md plus PNGs under
benchmarks/results/scaling/.

    uv run scripts/analyze_scaling.py [--in benchmarks/results/scaling.jsonl]

For the cols axis (where rows shrink past 4k cols to cap cell count) the
exponent comes from a joint 2D fit log t ~ a*log rows + b*log cols over the
rows+cols cells, alongside a throughput-vs-cols curve.
"""
import argparse
import json
import pathlib

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]


def load(path: pathlib.Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        flat = {**{f"cell_{k}": v for k, v in r.pop("cell").items()},
                **{f"host_{k}": v for k, v in r.pop("host").items()}, **r}
        rows.append(flat)
    df = pd.DataFrame(rows)
    # Latest result wins per logical run (reruns supersede).
    key = ["host_name", "variant", "threads", "repeat", "cell_axis", "cell_rows",
           "cell_cols", "cell_bins"]
    df = df.sort_values("ts").groupby(key, as_index=False).last()
    return df


def ok(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.status == "ok"].copy()


def mean_over_repeats(df: pd.DataFrame) -> pd.DataFrame:
    key = ["host_name", "variant", "threads", "cell_axis", "cell_rows",
           "cell_cols", "cell_bins"]
    agg = {c: "mean" for c in ("fit_s", "predict_s", "r2_train", "r2_test",
                               "fit_rows_per_s", "predict_rows_per_s")}
    return df.groupby(key, as_index=False).agg(agg | {"status": "first"})


def fit_exponent(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.any(y <= 0):
        return None
    a, _ = np.polyfit(np.log(x), np.log(y), 1)
    return round(float(a), 2)


def exponents_for(df: pd.DataFrame, host: str) -> pd.DataFrame:
    """Per variant: fit_s power-law exponent along each axis."""
    out = []
    d = df[df.host_name == host]
    base = d[d.cell_axis == "base"]
    for variant in sorted(d.variant.unique()):
        row: dict = {"variant": variant}
        v = d[d.variant == variant]
        vb = pd.concat([v, base[base.variant == variant]])
        rows_pts = vb[vb.cell_axis.isin(["rows", "base"])]
        row["rows_exp"] = fit_exponent(rows_pts.cell_rows.to_numpy(float),
                                       rows_pts.fit_s.to_numpy(float))
        bins_pts = vb[vb.cell_axis.isin(["bins", "base"])]
        row["bins_exp"] = fit_exponent(bins_pts.cell_bins.to_numpy(float),
                                       bins_pts.fit_s.to_numpy(float))
        # cols: joint 2D fit over rows+cols+base cells de-confounds the
        # shrinking-rows tail.
        joint = vb[vb.cell_axis.isin(["rows", "cols", "base"])]
        if len(joint) >= 4 and (joint.fit_s > 0).all():
            A = np.column_stack([np.log(joint.cell_rows.to_numpy(float)),
                                 np.log(joint.cell_cols.to_numpy(float)),
                                 np.ones(len(joint))])
            coef, *_ = np.linalg.lstsq(A, np.log(joint.fit_s.to_numpy(float)),
                                       rcond=None)
            row["rows_exp_joint"] = round(float(coef[0]), 2)
            row["cols_exp_joint"] = round(float(coef[1]), 2)
        else:
            row["rows_exp_joint"] = row["cols_exp_joint"] = None
        thr = vb[vb.cell_axis.isin(["threads", "base"])]
        row["threads_exp"] = fit_exponent(thr.threads.to_numpy(float),
                                          thr.fit_s.to_numpy(float))
        out.append(row)
    return pd.DataFrame(out)


AXIS_X = {"rows": "cell_rows", "cols": "cell_cols", "bins": "cell_bins",
          "threads": "threads"}


def plot_axis(df: pd.DataFrame, host: str, axis: str, outdir: pathlib.Path):
    d = df[(df.host_name == host) & df.cell_axis.isin([axis, "base"])]
    if d.empty:
        return None
    xcol = AXIS_X[axis]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for variant in sorted(d.variant.unique()):
        v = d[d.variant == variant].sort_values(xcol)
        if v.empty:
            continue
        cells_per_s = (v.cell_rows * v.cell_cols) / v.fit_s
        ax1.loglog(v[xcol], cells_per_s, "o-", label=variant, alpha=0.8)
        ax2.loglog(v[xcol], v.predict_rows_per_s, "o-", label=variant, alpha=0.8)
    ax1.set(xlabel=axis, ylabel="fit throughput (cells/s)",
            title=f"fit — {axis} axis")
    ax2.set(xlabel=axis, ylabel="predict rows/s", title=f"predict — {axis} axis")
    ax1.grid(True, which="both", alpha=0.3)
    ax2.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=7)
    fig.suptitle(host)
    fig.tight_layout()
    safe = host.replace("/", "-").replace(" ", "_")
    out = outdir / f"{safe}_{axis}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default=str(REPO / "benchmarks/results/scaling.jsonl"))
    ap.add_argument("--out", default=str(REPO / "benchmarks/results/scaling.md"))
    args = ap.parse_args()

    raw = load(pathlib.Path(args.inp))
    good = mean_over_repeats(ok(raw))
    outdir = REPO / "benchmarks" / "results" / "scaling"
    outdir.mkdir(parents=True, exist_ok=True)

    lines = ["# Scaling suite results", "",
             f"Generated by scripts/analyze_scaling.py from {args.inp}. "
             "Exponent a means fit_s ~ axis^a (log-log slope); cols uses a "
             "joint 2D fit over the rows+cols cells because rows shrink past "
             "4k cols.", ""]

    for host in sorted(raw.host_name.unique()):
        h = raw[raw.host_name == host].iloc[-1]
        lines += [f"## {host}", "",
                  f"{h.host_cpu_model} · {h.host_n_vcpu} vCPU · "
                  f"{h.host_ram_gb} GB RAM · GPU: {h.host_gpu or 'none'}"
                  + (f" ({h.host_gpu_vram_gb} GB)" if h.host_gpu else ""), ""]
        exp = exponents_for(good, host)
        if not exp.empty:
            lines += ["### Fitted exponents (fit_s ~ axis^a)", "",
                      md_table(exp), ""]
        base = good[(good.host_name == host) & (good.cell_axis == "base")]
        base_raw = ok(raw)[(ok(raw).host_name == host)
                           & (ok(raw).cell_axis == "base")]
        if not base_raw.empty:
            var = base_raw.groupby("variant").fit_s.agg(["mean", "std"])
            var["cv_pct"] = (100 * var["std"] / var["mean"]).round(1)
            lines += ["### Base cell (1M×100×255, depth 8, 100 iters, 16 threads)",
                      "", md_table(var.reset_index().round(3)), ""]
        for axis in ("rows", "cols", "bins", "threads"):
            png = plot_axis(good, host, axis, outdir)
            d = good[(good.host_name == host)
                     & good.cell_axis.isin([axis, "base"])]
            if d.empty:
                continue
            cols = list(dict.fromkeys(
                [AXIS_X[axis], "cell_rows", "cell_cols", "cell_bins", "threads",
                 "variant", "fit_s", "predict_s", "r2_test"]))
            tbl = d.sort_values([AXIS_X[axis], "variant"])[cols].round(3)
            lines += [f"### {axis} axis", ""]
            if png:
                lines += [f"![{axis}]({png.relative_to(REPO / 'benchmarks/results')})",
                          ""]
            lines += [md_table(tbl), ""]
        frontier = raw[(raw.host_name == host)
                       & raw.status.isin(["skipped", "oom", "timeout",
                                          "unsupported"])]
        if not frontier.empty:
            f = frontier[["variant", "cell_rows", "cell_cols", "cell_bins",
                          "status", "message"]].drop_duplicates()
            lines += ["### Feasibility frontier", "", md_table(f), ""]

    pathlib.Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} and {len(list(outdir.glob('*.png')))} plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
