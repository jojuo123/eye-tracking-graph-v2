"""Sensitivity analysis: for every synthetic damage type in `perturbations.py`, at a
range of severity levels, damage real REFLACX scanpaths and measure how much every
metric in `permutation_distances.py` / `scanpath_similarity.py` /
`trajectory_distances.py` moves -- producing one figure per damage type, each a grid
of small multiples (one subplot per metric family, since the families live on
different scales) plotting metric value vs. severity level.

This is purely a diagnostic/exploratory script (which metrics are sensitive to which
kind of scanpath-order error, and which are blind to it) -- it doesn't change any of
the three metric modules.

Requires `h5py` and `matplotlib`, unlike the rest of this directory (which is
numpy-only): both are optional extras for this one script, not full project
dependencies.

Usage: `python3 run_damage_evaluation.py` (uses this repo's local
`reflacx_data.h5` smoke-test file, 12 samples, by default -- see `--h5-path` to
point at a larger H5 file, e.g. once ERDA is mounted).

Long runs on a time-limited cluster: the sweep is checkpointed, so the full
REFLACX TRAIN split doesn't have to fit in one job's walltime. Results are
written to `results.csv` (long format, one row per damage type / metric /
parameter / severity level) alongside `results.json` and the figures, and
progress is saved to `checkpoint.json` every `--checkpoint-every` seconds and on
SIGTERM/SIGUSR1 (which is what SLURM sends when the walltime runs out).
Re-running the *same command* with the same `--output-dir` picks up where the
last job stopped; pass `--fresh` to start over instead. Every (damage type,
level, sample) work unit draws from its own seeded generator, so a resumed run
produces exactly the numbers an uninterrupted one would have, given the same
`--workers`/`--chunk-size` (see `--workers` below for why that qualifier is
there).

Multiprocessing (`--workers N`): the per-pair metrics are pure numpy/Python --
no I/O, no shared state -- so the sweep parallelizes over OS processes with no
special tricks *once the data is loaded*. The one thing that doesn't
parallelize is `h5py`: an open `h5py.File` can't be handed to worker processes
(it isn't picklable, and HDF5's C library isn't fork-safe once a file is open --
forking mid-read can corrupt the parent's handle). The workaround is the
structure this script already had before `--workers` existed: `main()` fully
reads every scanpath into plain numpy arrays via `load_local_reflacx_sequences`
*and closes the H5 file* before the sweep starts. Workers are only ever handed
those arrays (through a `Pool` initializer, once, not per task) -- `h5py` is
imported in worker processes (since they load this module) but never called.
"""

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import signal
import sys
import time

# Cap BLAS/OpenMP threading to 1 before numpy (or anything that pulls it in) loads.
# Every metric call here works on tiny arrays -- a few hundred points at most -- so
# BLAS's own thread pool is pure scheduling overhead even single-process; left at its
# default (often "all cores"), it multiplies with --workers's process-level
# parallelism into far more threads than the job actually has cores, which thrashes
# rather than helps. Only affects this process's env, not the calling shell's.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from perturbations import NEEDS_COORDS, PERTURBATIONS
from permutation_distances import (
    cayley_distance,
    hamming_distance,
    kendall_tau_distance,
    spearman_footrule_distance,
    ulam_distance,
)
from scanpath_similarity import cross_recurrence_analysis, multimatch, scanmatch, string_edit_distance
from trajectory_distances import (
    discrete_frechet_distance,
    dynamic_time_warping,
    edr_distance,
    erp_distance,
    hausdorff_distance,
    lcss_distance,
)

# ---------------------------------------------------------------------------
# Metric families -- grouped by scale, since a single axis can't honestly show a
# [0, 1]-bounded score next to a raw spatial distance (see dataviz "one axis" rule).
# ---------------------------------------------------------------------------

PERM_METRICS = ["hamming", "kendall_tau", "spearman_footrule", "cayley", "ulam"]
EDIT_METRICS = ["string_edit_distance", "scanmatch", "edr"]
MULTIMATCH_DIMS = ["vector", "length", "direction", "position", "multimatch"]
CRQA_METRICS = ["rec", "det", "lam", "corm"]
TRAJ_BOUNDED = ["dtw", "lcss"]
TRAJ_SPATIAL = ["discrete_frechet", "hausdorff", "erp"]

METRIC_GROUPS = [
    ("Permutation distances (fraction of max)", PERM_METRICS),
    ("Edit-based similarity (0-1)", EDIT_METRICS),
    ("MultiMatch dimensions (similarity, 0-1)", MULTIMATCH_DIMS),
    ("Cross-recurrence analysis (%)", CRQA_METRICS),
    ("Trajectory distances, bounded (0-1)", TRAJ_BOUNDED),
    ("Trajectory distances, spatial units", TRAJ_SPATIAL),
]
ALL_METRICS = [m for _, group in METRIC_GROUPS for m in group]
METRIC_GROUP_OF = {m: title for title, group in METRIC_GROUPS for m in group}


def evaluate_pair(
    coords,
    perm,
    eps_values=(0.05,),
    crqa_radius_values=(0.05,),
    grid_sizes=(5,),
    erp_gap_points=((0.0, 0.0),),
):
    """`coords`: `(n, 2)` original scanpath. `perm`: a permutation of `0..n-1` (see
    `perturbations.py`) -- the damaged scanpath is `coords[perm]`. Returns
    `{metric_name: {param_label: value}}` covering every metric in `ALL_METRICS`:
    metrics with no free parameter get a single `"default"` entry; metrics that take
    one (`lcss`/`edr`'s `eps`, the CRQA family's `radius`, `string_edit_distance`/
    `scanmatch`'s AOI `grid_shape`, `erp`'s `gap_point`) get one entry per value in
    the corresponding `*_values`/`grid_sizes`/`erp_gap_points` argument, so callers
    can see both severity-sensitivity and parameter-sensitivity at once. Count-based
    permutation distances are normalized by their maximum possible value (so they're
    comparable across scanpaths of different length); `dtw`/`string_edit_distance`/
    `edr` use those functions' own `normalize=True`; `erp` is divided by `n` (average
    per-point cost) to bring it onto roughly the same scale as `discrete_frechet`/
    `hausdorff` (otherwise it grows with `n` even for a fixed *fraction* of the path
    disturbed, since more points means more accumulated edit cost)."""
    n = coords.shape[0]
    identity = np.arange(n)
    damaged = coords[perm]

    max_kendall = n * (n - 1) / 2
    max_footrule = (n * n) // 2

    values = {
        "hamming": {"default": hamming_distance(identity, perm) / n},
        "kendall_tau": {
            "default": (kendall_tau_distance(identity, perm) / max_kendall) if max_kendall > 0 else 0.0
        },
        "spearman_footrule": {
            "default": (spearman_footrule_distance(identity, perm) / max_footrule) if max_footrule > 0 else 0.0
        },
        "cayley": {"default": (cayley_distance(identity, perm) / (n - 1)) if n > 1 else 0.0},
        "ulam": {"default": (ulam_distance(identity, perm) / (n - 1)) if n > 1 else 0.0},
        "dtw": {"default": dynamic_time_warping(coords, damaged, normalize=True)},
        "discrete_frechet": {"default": discrete_frechet_distance(coords, damaged)},
        "hausdorff": {"default": hausdorff_distance(coords, damaged)},
    }

    values["string_edit_distance"] = {
        f"grid={g}x{g}": string_edit_distance(coords, damaged, grid_shape=(g, g), normalize=True)
        for g in grid_sizes
    }
    values["scanmatch"] = {f"grid={g}x{g}": scanmatch(coords, damaged, grid_shape=(g, g)) for g in grid_sizes}
    values["edr"] = {f"eps={eps:g}": edr_distance(coords, damaged, eps=eps, normalize=True) for eps in eps_values}
    values["lcss"] = {f"eps={eps:g}": lcss_distance(coords, damaged, eps=eps) for eps in eps_values}
    values["erp"] = {
        f"gap=({gp[0]:g},{gp[1]:g})": (erp_distance(coords, damaged, gap_point=gp) / n if n > 0 else 0.0)
        for gp in erp_gap_points
    }

    mm = multimatch(coords, damaged)
    for dim in MULTIMATCH_DIMS:
        values[dim] = {"default": mm[dim] if mm[dim] is not None else float("nan")}

    for radius in crqa_radius_values:
        crqa = cross_recurrence_analysis(coords, damaged, radius=radius)
        label = f"radius={radius:g}"
        for key in CRQA_METRICS:
            values.setdefault(key, {})[label] = crqa[key] if crqa[key] is not None else float("nan")

    return values


def load_local_reflacx_sequences(
    h5_path, split="TRAIN", dataset_name="reflacx", x_col="x_position_norm", y_col="y_position_norm", max_fixations=100
):
    """Loads every sample's fixation `(x, y)` sequence from the REFLACX H5 file (same
    layout/columns `visualize_fixation_order_heatmap.py` reads), normalized `[0, 1]`.
    `max_fixations`: sequences longer than this are truncated to their first
    `max_fixations` fixations, to bound the runtime of this script's O(n^2)-ish
    metrics (several samples run into the hundreds of fixations)."""
    sequences = {}
    with h5py.File(h5_path, "r") as f:
        group = f[split][dataset_name]
        for sample_id in sorted(group.keys()):
            fx = group[sample_id]["fixations"]
            xy = np.stack([fx[x_col][()], fx[y_col][()]], axis=-1).astype(np.float64)
            xy = np.clip(xy, 0.0, 1.0)
            if max_fixations is not None and xy.shape[0] > max_fixations:
                xy = xy[:max_fixations]
            sequences[sample_id] = xy
    return sequences


# ---------------------------------------------------------------------------
# Resumable sweep: running totals + a work cursor, checkpointed to disk
# ---------------------------------------------------------------------------
#
# The sweep is one pass over `len(sequences) * len(PERTURBATIONS) * len(levels)`
# work units, in a fixed order, each unit contributing `n_repeats` damage draws for
# one (sample, damage type, severity level). The order is sample-major on purpose:
# every sample contributes to every (damage type, level) cell before the next sample
# starts, so a run that gets cut off early still has all of the curves, just averaged
# over fewer scanpaths -- rather than a few finished curves and nothing at all for
# the remaining damage types.
#
# Instead of keeping every draw in memory and averaging at the end, each unit folds
# its draws into running `[count, sum, sum_of_squares]` totals per (damage type,
# metric, param label, level) -- from which the pooled mean and std are recoverable
# exactly, in a few thousand numbers rather than the tens of millions of raw draws a
# full-split run produces. Totals plus the index of the next unfinished unit are the
# entire checkpoint, so a job killed at its walltime resumes from the last flush.
#
# Each unit seeds its own generator from (seed, unit indices) rather than drawing
# from one stream threaded through the whole sweep: that's what makes a resumed run
# reproduce the uninterrupted run's numbers exactly, instead of depending on where
# it was cut. (It also means this version's numbers differ from the pre-checkpoint
# version's -- same distribution, different draws.)

CHECKPOINT_VERSION = 1


def _unit_rng(seed, pert_index, level_index, sample_index):
    return np.random.default_rng([seed, pert_index, level_index, sample_index])


def new_accumulator():
    """`{pert_name: {metric: {param_label: {level_index: [count, sum, sumsq]}}}}`."""
    return {name: {m: {} for m in ALL_METRICS} for name in PERTURBATIONS}


def accumulate(accum, pert_name, level_index, values):
    for m in ALL_METRICS:
        for label, v in values[m].items():
            if isinstance(v, float) and np.isnan(v):
                continue
            by_level = accum[pert_name][m].setdefault(label, {})
            totals = by_level.setdefault(level_index, [0, 0.0, 0.0])
            totals[0] += 1
            totals[1] += float(v)
            totals[2] += float(v) * float(v)


def summarize(accum, levels):
    """Running totals -> the shape the plotting/JSON side expects:
    `{pert_name: {metric: {param_label: [[level, mean, std, n_draws], ...]}}}`,
    ordered by severity level and skipping levels with no draws yet (so a partial
    run still plots and exports)."""
    results = {}
    for pert_name, per_metric in accum.items():
        results[pert_name] = {}
        for m in ALL_METRICS:
            results[pert_name][m] = {}
            for label, by_level in per_metric.get(m, {}).items():
                series = []
                for level_index, level in enumerate(levels):
                    totals = by_level.get(level_index) or by_level.get(str(level_index))
                    if not totals or totals[0] == 0:
                        continue
                    count, total, total_sq = totals
                    mean = total / count
                    var = max(total_sq / count - mean * mean, 0.0)  # population std, as np.std
                    series.append([level, mean, var**0.5, count])
                if series:
                    results[pert_name][m][label] = series
    return results


def _config_fingerprint(args, erp_gap_points, sample_ids):
    """Identifies the sweep a checkpoint belongs to: resuming into a *different*
    sweep would silently pool incomparable draws, so the parameters, the sample list
    and the metric set all feed the hash."""
    payload = json.dumps(
        {
            "version": CHECKPOINT_VERSION,
            "levels": list(args.levels),
            "n_repeats": args.n_repeats,
            "seed": args.seed,
            "eps_values": list(args.eps_values),
            "crqa_radius_values": list(args.crqa_radius_values),
            "grid_sizes": list(args.grid_sizes),
            "erp_gap_points": [list(p) for p in erp_gap_points],
            "split": args.split,
            "dataset_name": args.dataset_name,
            "max_fixations": args.max_fixations,
            "sample_ids": list(sample_ids),
            "perturbations": list(PERTURBATIONS),
            "metrics": ALL_METRICS,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_atomic(path, write_fn):
    """Write via a temp file + `os.replace`, so a job killed mid-write leaves the
    previous checkpoint intact rather than a truncated one."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", newline="") as f:
        write_fn(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def save_checkpoint(path, fingerprint, cursor, units_total, accum, elapsed):
    state = {
        "version": CHECKPOINT_VERSION,
        "fingerprint": fingerprint,
        "cursor": cursor,
        "units_total": units_total,
        "elapsed_seconds": elapsed,
        "accum": accum,
    }
    _write_atomic(path, lambda f: json.dump(state, f))


def load_checkpoint(path, fingerprint):
    """Returns `(cursor, accum, elapsed)`, or `None` if there's nothing to resume.
    Raises on a checkpoint from a different sweep rather than silently discarding
    or mixing it."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        state = json.load(f)
    if state.get("fingerprint") != fingerprint:
        raise SystemExit(
            f"{path} was written by a run with different settings (different levels, seed, swept\n"
            "parameters, or input samples), so its partial totals can't be pooled with this run's.\n"
            "Re-run with the original settings to resume it, or pass --fresh to discard it, or point\n"
            "--output-dir somewhere else."
        )
    accum = new_accumulator()
    for pert_name, per_metric in state["accum"].items():
        if pert_name not in accum:
            continue
        for m, by_label in per_metric.items():
            for label, by_level in by_label.items():
                accum[pert_name][m][label] = {int(k): list(v) for k, v in by_level.items()}
    return state["cursor"], accum, state.get("elapsed_seconds", 0.0)


def write_results_csv(path, results):
    """Long format -- one row per (damage type, metric, swept parameter, severity
    level) -- so the curves load straight into pandas/R without unpacking the nested
    JSON. `n_draws` is the number of (sample, repeat) draws behind the mean, which
    varies for `swap_nearby_locations` (see `run_sweep`)."""
    def write(f):
        writer = csv.writer(f)
        writer.writerow(
            ["perturbation", "metric", "metric_group", "param_label", "level", "mean", "std", "n_draws"]
        )
        for pert_name, per_metric in results.items():
            for m in ALL_METRICS:
                for label, series in per_metric.get(m, {}).items():
                    for level, mean, std, count in series:
                        writer.writerow(
                            [pert_name, m, METRIC_GROUP_OF[m], label, level, f"{mean:.10g}", f"{std:.10g}", count]
                        )

    _write_atomic(path, write)


def merge_accumulators(dst, src):
    """Folds `src`'s running totals (see `new_accumulator`) into `dst`'s, in place --
    how a parallel worker's per-chunk totals get pooled into the main process's
    running accumulator."""
    for pert_name, per_metric in src.items():
        for m, by_label in per_metric.items():
            for label, by_level in by_label.items():
                dst_by_level = dst[pert_name][m].setdefault(label, {})
                for level_index, (count, total, total_sq) in by_level.items():
                    dst_totals = dst_by_level.setdefault(level_index, [0, 0.0, 0.0])
                    dst_totals[0] += count
                    dst_totals[1] += total
                    dst_totals[2] += total_sq


def _evaluate_unit(
    accum, unit, coords_list, pert_items, levels, seed, n_repeats,
    eps_values, crqa_radius_values, grid_sizes, erp_gap_points,
):
    """Runs one (sample, damage type, severity level) work unit -- `n_repeats` damage
    draws, each evaluated against every metric -- and folds the results into `accum`.
    Shared by the serial and parallel-worker paths in `run_sweep`, so both draw and
    seed identically; see `run_sweep` for `unit`'s indexing and the
    `swap_nearby_locations` skip rule."""
    n_levels = len(levels)
    sample_index, rest = divmod(unit, len(pert_items) * n_levels)
    pert_index, level_index = divmod(rest, n_levels)
    pert_name, pert_fn = pert_items[pert_index]
    level = levels[level_index]
    coords = coords_list[sample_index]
    n = coords.shape[0]
    needs_coords = pert_name in NEEDS_COORDS
    rng = _unit_rng(seed, pert_index, level_index, sample_index)

    for _ in range(n_repeats):
        perm = pert_fn(n, level, rng, coords=coords if needs_coords else None)
        if pert_name == "swap_nearby_locations" and level > 0 and np.array_equal(perm, np.arange(n)):
            continue
        vals = evaluate_pair(
            coords,
            perm,
            eps_values=eps_values,
            crqa_radius_values=crqa_radius_values,
            grid_sizes=grid_sizes,
            erp_gap_points=erp_gap_points,
        )
        accumulate(accum, pert_name, level_index, vals)


# ---------------------------------------------------------------------------
# Worker-process side of `--workers`: everything below runs inside a `Pool` worker,
# never in the main process. State is handed to each worker exactly once, via
# `Pool`'s `initializer`, and cached in `_worker_state` -- not per task -- since
# `coords_list` (the whole sweep's input data) is the one thing too big to want to
# re-send per chunk. `PERTURBATIONS`/`evaluate_pair`/etc. need no such handoff: each
# worker process loads this module itself (inherited via fork, or re-imported under
# spawn), so its own copy of those module-level globals is already there.
# ---------------------------------------------------------------------------

_worker_state = {}


def _init_worker(coords_list, levels, seed, n_repeats, eps_values, crqa_radius_values, grid_sizes, erp_gap_points):
    # SLURM (and an interactive Ctrl-C) delivers SIGTERM/SIGUSR1 to the whole process
    # group, i.e. to workers too, not just the main process. Workers ignore them: the
    # main process's own handler is what decides when to stop (see `main`), by simply
    # not submitting another batch -- a worker dying mid-chunk would instead surface
    # as a `pool.map` exception and lose that chunk's totals, checkpointed or not.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    _worker_state.update(
        coords_list=coords_list,
        pert_items=list(PERTURBATIONS.items()),
        levels=levels,
        seed=seed,
        n_repeats=n_repeats,
        eps_values=eps_values,
        crqa_radius_values=crqa_radius_values,
        grid_sizes=grid_sizes,
        erp_gap_points=erp_gap_points,
    )


def _process_chunk(unit_range):
    """Runs work units `[start, stop)` against this worker's cached `_worker_state`
    and returns `(start, stop, local_accum)` -- a fresh accumulator holding just this
    chunk's totals, for the main process to fold in via `merge_accumulators`."""
    start, stop = unit_range
    st = _worker_state
    local_accum = new_accumulator()
    for unit in range(start, stop):
        _evaluate_unit(
            local_accum, unit, st["coords_list"], st["pert_items"], st["levels"], st["seed"], st["n_repeats"],
            st["eps_values"], st["crqa_radius_values"], st["grid_sizes"], st["erp_gap_points"],
        )
    return start, stop, local_accum


def run_sweep(
    sequences,
    levels,
    n_repeats,
    seed=0,
    eps_values=(0.05,),
    crqa_radius_values=(0.05,),
    grid_sizes=(5,),
    erp_gap_points=((0.0, 0.0),),
    accum=None,
    start_unit=0,
    on_progress=None,
    should_stop=None,
    workers=1,
    chunk_size=8,
):
    """For every (perturbation type, severity level), draws `n_repeats` random
    damage realizations per sample (this is the "good amount of artificial
    orderings per sample" the averaging relies on -- a single realization at a
    given level is a noisy draw, e.g. which adjacent pairs `local_swaps` happens to
    pick; several repeats per sample, pooled with all other samples, is what turns
    that into a stable curve) and evaluates every metric -- at every swept parameter
    value, for metrics that take one (see `evaluate_pair`) -- comparing each damaged
    scanpath back to its own original.

    Folds each draw into `accum`'s running totals (see `new_accumulator`) and
    returns `(accum, next_unit)`: `next_unit == units_total` means the sweep
    finished, anything less means `should_stop` asked it to stop early and the
    caller should checkpoint and exit. `on_progress(next_unit, units_total)` is
    called after every work unit (`workers <= 1`) or every completed batch of chunks
    (`workers > 1`), for periodic checkpointing.

    `swap_nearby_locations` silently skips (sample, repeat) draws where the sample
    has no eligible near-revisit pair at that draw's search radius
    (`perturbations.swap_nearby_locations` returns the identity permutation in that
    case) -- counting those as "zero damage" would understate the metrics' true
    sensitivity by diluting the average with samples the perturbation couldn't even
    apply to.

    `workers`: `<= 1` runs in-process (the original, single-core behavior -- no
    `multiprocessing` overhead, simplest to debug). `> 1` spins up a `Pool` of that
    many worker processes (see the module docstring for why `h5py` doesn't need to,
    and can't, follow the data there) and hands out `chunk_size`-unit chunks, one
    batch of `workers` chunks at a time: each batch is a single blocking `pool.map`
    call, so `should_stop` is only checked between batches, never mid-chunk -- a
    SIGTERM's worst-case extra work is one in-flight chunk per worker (tune
    `chunk_size` down if that's more slack than the job's walltime margin allows).
    Chunks are contiguous unit ranges processed and returned in submission order, so
    the cursor this returns is still a single "everything before this is done"
    integer, exactly like the serial path's.
    """
    accum = new_accumulator() if accum is None else accum
    pert_items = list(PERTURBATIONS.items())
    coords_list = list(sequences.values())
    n_levels, n_samples = len(levels), len(coords_list)
    units_total = len(pert_items) * n_levels * n_samples

    if workers <= 1:
        for unit in range(start_unit, units_total):
            _evaluate_unit(
                accum, unit, coords_list, pert_items, levels, seed, n_repeats,
                eps_values, crqa_radius_values, grid_sizes, erp_gap_points,
            )
            if on_progress is not None:
                on_progress(unit + 1, units_total)
            if should_stop is not None and should_stop():
                return accum, unit + 1
        return accum, units_total

    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context("spawn")
    initargs = (coords_list, levels, seed, n_repeats, eps_values, crqa_radius_values, grid_sizes, erp_gap_points)
    cursor = start_unit
    with ctx.Pool(processes=workers, initializer=_init_worker, initargs=initargs) as pool:
        while cursor < units_total:
            batch, pos = [], cursor
            for _ in range(workers):
                if pos >= units_total:
                    break
                stop = min(pos + chunk_size, units_total)
                batch.append((pos, stop))
                pos = stop

            for chunk_start, chunk_stop, local_accum in pool.map(_process_chunk, batch):
                merge_accumulators(accum, local_accum)
                cursor = chunk_stop  # batch ranges are contiguous & returned in submission order
            if on_progress is not None:
                on_progress(cursor, units_total)
            if should_stop is not None and should_stop():
                return accum, cursor

    return accum, cursor


# ---------------------------------------------------------------------------
# Plotting -- validated categorical palette (dataviz skill, references/palette.md):
# fixed hue order, never cycled/reassigned per-chart; each metric group uses a fixed
# prefix of the 8-slot order (so within any subplot, only adjacent-in-the-validated-
# order pairs appear together).
# ---------------------------------------------------------------------------

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
GRID, AXIS_LINE = "#e1e0d9", "#c3c2b7"
TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED = "#0b0b0b", "#52514e", "#898781"

# One metric = one color (fixed across its param sweep, so the eye groups by metric
# first); one parameter value = one line style, in sweep order, so the eye can then
# read severity within a metric.
LINESTYLES = ["-", "--", ":", "-."]


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_LINE)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)


def plot_perturbation(pert_name, pert_results, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(19, 9))
    fig.patch.set_facecolor(PAGE)
    for ax, (group_title, metric_names) in zip(axes.ravel(), METRIC_GROUPS):
        _style_axes(ax)
        for color, metric in zip(PALETTE, metric_names):
            param_series = pert_results[metric]  # {param_label: [(level, mean, std, n), ...]}
            single_default = list(param_series.keys()) == ["default"]
            for i, (label, series) in enumerate(param_series.items()):
                xs = [row[0] for row in series]
                ys = [row[1] for row in series]
                stds = [row[2] for row in series]
                linestyle = LINESTYLES[i % len(LINESTYLES)]
                legend_label = metric if single_default else f"{metric} ({label})"
                ax.plot(
                    xs, ys, color=color, linewidth=1.6, marker="o", markersize=3.5,
                    linestyle=linestyle, label=legend_label,
                )
                lo = [y - s for y, s in zip(ys, stds)]
                hi = [y + s for y, s in zip(ys, stds)]
                ax.fill_between(xs, lo, hi, color=color, alpha=0.10, linewidth=0)
        ax.set_title(group_title, fontsize=9.5, color=TEXT_PRIMARY, loc="left")
        ax.set_xlabel("severity level", fontsize=8, color=TEXT_SECONDARY)
        ax.set_xlim(-0.02, 1.02)
        ax.legend(
            fontsize=6, frameon=False, labelcolor=TEXT_SECONDARY,
            loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0,
        )
    fig.suptitle(f"Damage type: {pert_name}", fontsize=13, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = os.path.join(out_dir, f"{pert_name}.png")
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return out_path


def _default_worker_count():
    """Cores actually available to this process, not the machine's total -- on a
    SLURM node, `os.cpu_count()` reports every core on the node regardless of
    `--cpus-per-task`, while the sched-affinity mask reflects the cgroup/cpuset SLURM
    actually confined this job to."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # sched_getaffinity is Linux-only
        return os.cpu_count() or 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--h5-path", default=os.path.join(os.path.dirname(__file__), "..", "reflacx_data.h5")
    )
    parser.add_argument("--split", default="TRAIN")
    parser.add_argument("--dataset-name", default="reflacx")
    parser.add_argument("--max-fixations", type=int, default=100)
    parser.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--n-repeats", type=int, default=5, help="random damage realizations per sample per level")
    parser.add_argument(
        "--crqa-radius-values", type=float, nargs="+", default=[0.03, 0.05, 0.08],
        help="cross_recurrence_analysis's `radius`, swept independently of severity",
    )
    parser.add_argument(
        "--eps-values", type=float, nargs="+", default=[0.025, 0.05, 0.1],
        help="lcss/edr's spatial match threshold `eps`, swept independently of severity",
    )
    parser.add_argument(
        "--grid-sizes", type=int, nargs="+", default=[3, 5, 8],
        help="string_edit_distance/scanmatch's AOI grid resolution (n -> an n x n grid)",
    )
    parser.add_argument(
        "--erp-gap-points", type=float, nargs="+", default=[0.0, 0.0, 0.5, 0.5],
        help="erp's reference gap point(s), as flat x,y pairs, e.g. `0 0 0.5 0.5` for two points",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "work_dir", "metric_damage_curves"),
    )
    parser.add_argument("--max-samples", type=int, default=None, help="use only the first N scanpaths")
    parser.add_argument(
        "--checkpoint-every", type=float, default=300.0,
        help="seconds between checkpoint/CSV flushes (a killed job loses at most this much work)",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="discard any existing checkpoint in --output-dir and start the sweep over",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="re-export CSV/JSON/figures from the existing checkpoint without evaluating anything",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="worker processes for the sweep (default 1 = single-process, original behavior). "
        "0 or negative = use every core this job has (see os.sched_getaffinity), "
        "i.e. what --cpus-per-task reserved on SLURM. See the module docstring for why this is "
        "safe despite h5py not being multiprocessing-friendly.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=8,
        help="work units per task handed to a worker process (--workers > 1 only); smaller = "
        "finer-grained load balancing and a shorter SIGTERM grace period, larger = less "
        "inter-process overhead",
    )
    args = parser.parse_args()
    if args.workers <= 0:
        args.workers = _default_worker_count()

    if len(args.erp_gap_points) % 2 != 0:
        parser.error("--erp-gap-points must be a flat list of x,y pairs (even count)")
    erp_gap_points = [
        (args.erp_gap_points[i], args.erp_gap_points[i + 1]) for i in range(0, len(args.erp_gap_points), 2)
    ]

    # SLURM sends SIGTERM at the walltime (and SIGUSR1 first, if the job asked for an
    # early warning via `--signal`); finish the work unit in flight, flush, and exit 0
    # so the next submission of the same command resumes instead of redoing the sweep.
    # Installed before the H5 load so a signal arriving during startup doesn't kill the
    # process outright.
    stop_requested = {"value": False}

    def request_stop(signum, _frame):
        stop_requested["value"] = True
        print(f"\nreceived signal {signum} -- finishing current work unit, then checkpointing", flush=True)

    for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        signal.signal(sig, request_stop)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "checkpoint.json")
    csv_path = os.path.join(args.output_dir, "results.csv")
    json_path = os.path.join(args.output_dir, "results.json")

    sequences = load_local_reflacx_sequences(
        args.h5_path, args.split, args.dataset_name, max_fixations=args.max_fixations
    )
    if args.max_samples is not None:
        sequences = dict(list(sequences.items())[: args.max_samples])
    lengths = np.array([xy.shape[0] for xy in sequences.values()])
    print(
        f"Loaded {len(sequences)} scanpaths from {args.h5_path}: "
        f"lengths min/mean/max = {lengths.min()}/{lengths.mean():.1f}/{lengths.max()}"
    )
    print(f"workers={args.workers}" + (f" chunk_size={args.chunk_size}" if args.workers > 1 else ""))

    fingerprint = _config_fingerprint(args, erp_gap_points, list(sequences))
    units_total = len(PERTURBATIONS) * len(args.levels) * len(sequences)

    if args.fresh and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    restored = None if args.fresh else load_checkpoint(checkpoint_path, fingerprint)
    if restored is None:
        cursor, accum, elapsed_before = 0, new_accumulator(), 0.0
    else:
        cursor, accum, elapsed_before = restored
        print(f"Resuming from {checkpoint_path}: {cursor}/{units_total} work units already done")

    def export(cursor_now, elapsed_now):
        results = summarize(accum, args.levels)
        save_checkpoint(checkpoint_path, fingerprint, cursor_now, units_total, accum, elapsed_now)
        write_results_csv(csv_path, results)
        _write_atomic(json_path, lambda f: json.dump(results, f, indent=2))
        return results

    if args.plot_only:
        if restored is None:
            raise SystemExit(f"--plot-only needs an existing checkpoint; none found at {checkpoint_path}")
        results = export(cursor, elapsed_before)
        for pert_name, pert_results in results.items():
            print(f"wrote {plot_perturbation(pert_name, pert_results, args.output_dir)}")
        return

    started = time.time()
    last_flush = started
    units_done_at_start = cursor

    def on_progress(units_done, total):
        nonlocal last_flush
        now = time.time()
        if not stop_requested["value"] and (now - last_flush) < args.checkpoint_every:
            return
        last_flush = now
        elapsed = elapsed_before + (now - started)
        export(units_done, elapsed)
        rate = (units_done - units_done_at_start) / max(now - started, 1e-9)
        eta_hours = ((total - units_done) / rate) / 3600 if rate > 0 else float("inf")
        print(
            f"[{units_done}/{total} units, {100 * units_done / total:.1f}%] "
            f"{rate * 3600:.0f} units/h, ETA {eta_hours:.1f} h -- checkpointed to {checkpoint_path}",
            flush=True,
        )

    accum, cursor = run_sweep(
        sequences,
        args.levels,
        args.n_repeats,
        seed=args.seed,
        eps_values=args.eps_values,
        crqa_radius_values=args.crqa_radius_values,
        grid_sizes=args.grid_sizes,
        erp_gap_points=erp_gap_points,
        accum=accum,
        start_unit=cursor,
        on_progress=on_progress,
        should_stop=lambda: stop_requested["value"],
        workers=args.workers,
        chunk_size=args.chunk_size,
    )

    elapsed = elapsed_before + (time.time() - started)
    results = export(cursor, elapsed)
    print(f"wrote {csv_path}\nwrote {json_path}")

    if cursor < units_total:
        print(
            f"\nINCOMPLETE: {cursor}/{units_total} work units done ({100 * cursor / units_total:.1f}%), "
            f"{elapsed / 3600:.2f} h of compute so far.\n"
            f"{csv_path} holds the partial curves. Re-run the same command to resume; "
            f"add --plot-only to draw the figures from what's done."
        )
        return

    for pert_name, pert_results in results.items():
        out_path = plot_perturbation(pert_name, pert_results, args.output_dir)
        print(f"wrote {out_path}")
    print(f"\nDone: {units_total} work units in {elapsed / 3600:.2f} h.")


if __name__ == "__main__":
    main()
