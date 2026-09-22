"""Scanpath comparison methods that work on raw `(x, y)` fixation positions (as
opposed to `permutation_distances.py`, which compares two *orderings* of the same
fixed set of items).

    string_edit_distance     -- Levenshtein distance between two AOI-label sequences
                                 (Privitera & Stark, 2000)
    scanmatch                -- Needleman-Wunsch alignment over AOI-label sequences,
                                 scored by spatial proximity rather than exact match
                                 (Cristino, Mathot, Theeuwes & Gilchrist, 2010)
    multimatch                -- vector-based scanpath similarity over 4 geometric
                                 dimensions (vector, length, direction, position);
                                 the 5th dimension of the original method, duration,
                                 is omitted (Dewhurst et al., 2012)
    cross_recurrence_analysis -- CRQA: recurrence/determinism/laminarity/temporal-order
                                 measures from the two scanpaths' cross-recurrence plot
                                 (Anderson, Bischof, Laidlaw, Risko & Kingstone, 2013)

`string_edit_distance` and `scanmatch` both first discretize fixations into a regular
grid of AOI labels via `_bin_positions` -- pass the same `grid_shape`/`bounds` to both
if comparing their outputs. `multimatch` and `cross_recurrence_analysis` work directly
on the continuous positions.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Shared: fixations -> AOI grid labels (string_edit_distance, scanmatch)
# ---------------------------------------------------------------------------


def _bin_positions(fixations, grid_shape=(5, 5), bounds=((0.0, 1.0), (0.0, 1.0))):
    """`fixations`: `(n, 2)` array of `(x, y)` positions. Bins each fixation into a
    `grid_shape[0] x grid_shape[1]` (cols x rows) regular grid spanning `bounds`
    (`((x_min, x_max), (y_min, y_max))`), clipping out-of-bounds positions to the
    nearest edge cell. Returns `(labels, centers)`:
      `labels`  -- `(n,)` int array, one grid-cell index per fixation (row-major:
                   `label = row * n_cols + col`).
      `centers` -- `(n_cols * n_rows, 2)` array, the `(x, y)` center of each grid
                   cell, indexed the same way as `labels` -- used by `scanmatch` to
                   score substitutions by spatial proximity.
    """
    fixations = np.asarray(fixations, dtype=np.float64)
    (x_min, x_max), (y_min, y_max) = bounds
    n_cols, n_rows = grid_shape
    cell_w = (x_max - x_min) / n_cols
    cell_h = (y_max - y_min) / n_rows

    col = np.clip(((fixations[:, 0] - x_min) / cell_w).astype(np.int64), 0, n_cols - 1)
    row = np.clip(((fixations[:, 1] - y_min) / cell_h).astype(np.int64), 0, n_rows - 1)
    labels = row * n_cols + col

    cols, rows = np.meshgrid(np.arange(n_cols), np.arange(n_rows))  # both (n_rows, n_cols)
    centers = np.stack(
        [x_min + (cols.ravel() + 0.5) * cell_w, y_min + (rows.ravel() + 0.5) * cell_h], axis=-1
    )  # (n_rows * n_cols, 2), same row-major order as `labels`

    return labels, centers


def _collapse_repeats(seq):
    """`[1, 1, 2, 2, 2, 3] -> [1, 2, 3]`: collapses consecutive duplicate labels, so
    dwelling on one AOI across several consecutive fixations counts as a single visit."""
    if len(seq) == 0:
        return seq
    out = [seq[0]]
    for x in seq[1:]:
        if x != out[-1]:
            out.append(x)
    return out


# ---------------------------------------------------------------------------
# String edit distance (Levenshtein over AOI labels)
# ---------------------------------------------------------------------------


def _levenshtein(seq_a, seq_b):
    """Classic Levenshtein distance: minimum number of substitutions/insertions/
    deletions (each cost 1) to turn `seq_a` into `seq_b`. O(len(a) * len(b))."""
    n, m = len(seq_a), len(seq_b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def string_edit_distance(
    fixations_a,
    fixations_b,
    grid_shape=(5, 5),
    bounds=((0.0, 1.0), (0.0, 1.0)),
    collapse_repeats=False,
    normalize=False,
):
    """String-edit-distance scanpath comparison (Levenshtein distance applied to AOI
    sequences, e.g. Privitera & Stark, 2000): bins each scanpath into a `grid_shape`
    grid of region labels (`_bin_positions`), then returns the Levenshtein distance
    between the two label sequences.

    `fixations_a`/`fixations_b`: `(n, 2)`/`(m, 2)` arrays of `(x, y)` positions (need
    not have equal length). `collapse_repeats`: if True, consecutive fixations in the
    same AOI count as one visit rather than one symbol each. `normalize`: if True,
    divide by `max(len(seq_a), len(seq_b))` to get a distance in `[0, 1]`.
    """
    labels_a, _ = _bin_positions(fixations_a, grid_shape, bounds)
    labels_b, _ = _bin_positions(fixations_b, grid_shape, bounds)
    seq_a, seq_b = labels_a.tolist(), labels_b.tolist()
    if collapse_repeats:
        seq_a, seq_b = _collapse_repeats(seq_a), _collapse_repeats(seq_b)

    distance = _levenshtein(seq_a, seq_b)
    if normalize:
        return distance / max(len(seq_a), len(seq_b), 1)
    return distance


# ---------------------------------------------------------------------------
# ScanMatch (Needleman-Wunsch alignment, spatially-weighted substitutions)
# ---------------------------------------------------------------------------


def _substitution_matrix(centers, max_score=1.0, sigma=None):
    """`centers`: `(k, 2)` grid-cell centers -> `(k, k)` substitution score matrix:
    `max_score` for identical AOIs, decaying as a Gaussian in inter-center distance
    otherwise (spatially closer AOIs score as more "similar" fixations, per Cristino
    et al., 2010). `sigma` defaults to the grid's mean nearest-neighbor center
    spacing, so the decay scales with cell size."""
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    if sigma is None:
        if centers.shape[0] <= 1:
            sigma = 1.0
        else:
            nn_dist = np.where(dists > 0, dists, np.inf).min(axis=-1)
            finite = nn_dist[np.isfinite(nn_dist)]
            sigma = float(np.mean(finite)) if finite.size > 0 and np.mean(finite) > 0 else 1.0
    return max_score * np.exp(-(dists**2) / (2 * sigma**2))


def _needleman_wunsch(seq_a, seq_b, sub_matrix, gap_penalty):
    """Global (Needleman-Wunsch) alignment score between label sequences `seq_a`,
    `seq_b`: per-pair substitution scores from `sub_matrix[label_a, label_b]`, and a
    fixed linear `gap_penalty` per unaligned (inserted/deleted) element. Returns the
    best-alignment total score (higher = more similar). O(len(a) * len(b))."""
    n, m = len(seq_a), len(seq_b)
    dp = np.zeros((n + 1, m + 1))
    dp[:, 0] = -gap_penalty * np.arange(n + 1)
    dp[0, :] = -gap_penalty * np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = dp[i - 1, j - 1] + sub_matrix[seq_a[i - 1], seq_b[j - 1]]
            delete = dp[i - 1, j] - gap_penalty
            insert = dp[i, j - 1] - gap_penalty
            dp[i, j] = max(match, delete, insert)
    return dp[n, m]


def scanmatch(
    fixations_a,
    fixations_b,
    grid_shape=(5, 5),
    bounds=((0.0, 1.0), (0.0, 1.0)),
    gap_penalty=0.0,
    sigma=None,
):
    """ScanMatch (Cristino, Mathot, Theeuwes & Gilchrist, 2010): bins each scanpath
    into a `grid_shape` grid of AOI labels (`_bin_positions`), then finds the best
    Needleman-Wunsch global alignment between the two label sequences -- scoring each
    aligned pair by spatial proximity of their AOIs (`_substitution_matrix`) rather
    than requiring an exact label match, the way plain `string_edit_distance` does.

    Returns a similarity in `[0, 1]` (1 = identical scanpaths): the raw alignment
    score normalized by the mean of the two sequences' self-alignment scores, per the
    original ScanMatch toolbox.

    `gap_penalty`: cost subtracted per unaligned fixation; the original toolbox
    defaults to 0 (skipping a fixation is free -- only mismatched *positions* are
    penalized, through the substitution matrix).
    """
    labels_a, centers = _bin_positions(fixations_a, grid_shape, bounds)
    labels_b, _ = _bin_positions(fixations_b, grid_shape, bounds)
    labels_a, labels_b = labels_a.tolist(), labels_b.tolist()

    sub_matrix = _substitution_matrix(centers, sigma=sigma)
    score = _needleman_wunsch(labels_a, labels_b, sub_matrix, gap_penalty)
    self_a = _needleman_wunsch(labels_a, labels_a, sub_matrix, gap_penalty)
    self_b = _needleman_wunsch(labels_b, labels_b, sub_matrix, gap_penalty)

    denom = (self_a + self_b) / 2
    if denom <= 0:
        return 0.0
    return float(np.clip(score / denom, 0.0, 1.0))


# ---------------------------------------------------------------------------
# MultiMatch, without the duration dimension
# ---------------------------------------------------------------------------


def _saccade_vectors(xy):
    """`(n, 2)` fixation positions -> `(n - 1, 2)` consecutive saccade vectors."""
    return xy[1:] - xy[:-1]


def _alignment_path(cost):
    """Lowest-cost monotonic alignment path through an `(m, n)` cost matrix, from
    `(0, 0)` to `(m - 1, n - 1)`, moving right/down/diagonally at each step -- the
    reference MultiMatch algorithm finds this via Dijkstra's shortest path over the
    same grid graph; this dynamic-programming formulation (equivalent, since the grid
    is a DAG) is simpler to implement correctly. Returns the path as a list of
    `(i, j)` index pairs, one per aligned (vector_a, vector_b) pair."""
    m, n = cost.shape
    dp = np.zeros((m, n))
    dp[0, 0] = cost[0, 0]
    for i in range(1, m):
        dp[i, 0] = dp[i - 1, 0] + cost[i, 0]
    for j in range(1, n):
        dp[0, j] = dp[0, j - 1] + cost[0, j]
    for i in range(1, m):
        for j in range(1, n):
            dp[i, j] = cost[i, j] + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    path = [(m - 1, n - 1)]
    i, j = m - 1, n - 1
    while (i, j) != (0, 0):
        candidates = []
        if i > 0:
            candidates.append((dp[i - 1, j], i - 1, j))
        if j > 0:
            candidates.append((dp[i, j - 1], i, j - 1))
        if i > 0 and j > 0:
            candidates.append((dp[i - 1, j - 1], i - 1, j - 1))
        _, i, j = min(candidates, key=lambda c: c[0])
        path.append((i, j))
    path.reverse()
    return path


def multimatch(fixations_a, fixations_b, screen_diagonal=2**0.5):
    """MultiMatch scanpath similarity (Dewhurst et al., 2012), restricted to its 4
    geometric dimensions -- vector, length, direction, position -- omitting the
    original 5th dimension, duration, for when per-fixation durations aren't
    available or aren't of interest. Represents each scanpath as its sequence of
    saccade vectors, finds the lowest-cost monotonic alignment between the two
    scanpaths' vectors, then reports normalized similarity (1 = identical) over that
    alignment.

    `fixations_a`/`fixations_b`: `(n, 2)`/`(m, 2)` arrays of `(x, y)` positions.
    `screen_diagonal`: normalizer for all 4 dimensions (all spatial) -- the maximum
    possible distance in the coordinate space the positions use. Defaults to
    `sqrt(2)`, correct if positions are normalized to `[0, 1]`.

    Returns a dict of the 4 sub-scores plus their mean under `"multimatch"`. Returns
    all-`None` if either scanpath is too short (< 2 fixations) to form a saccade
    vector.
    """
    dims = ("vector", "length", "direction", "position")
    fixations_a = np.asarray(fixations_a, dtype=np.float64)
    fixations_b = np.asarray(fixations_b, dtype=np.float64)
    if fixations_a.shape[0] < 2 or fixations_b.shape[0] < 2:
        return {**{dim: None for dim in dims}, "multimatch": None}

    vectors_a = _saccade_vectors(fixations_a)
    vectors_b = _saccade_vectors(fixations_b)
    # "Vector difference" cost drives the alignment itself, per the reference algorithm.
    cost = np.linalg.norm(vectors_a[:, None, :] - vectors_b[None, :, :], axis=-1)  # (m, n)
    path = _alignment_path(cost)

    lengths_a, lengths_b = np.linalg.norm(vectors_a, axis=-1), np.linalg.norm(vectors_b, axis=-1)
    angles_a = np.arctan2(vectors_a[:, 1], vectors_a[:, 0])
    angles_b = np.arctan2(vectors_b[:, 1], vectors_b[:, 0])

    vector_d, length_d, direction_d, position_d = [], [], [], []
    for i, j in path:
        vector_d.append(cost[i, j])
        length_d.append(abs(lengths_a[i] - lengths_b[j]))
        angle_diff = abs(angles_a[i] - angles_b[j])
        direction_d.append(min(angle_diff, 2 * np.pi - angle_diff))  # smaller of the two arcs
        position_d.append(np.linalg.norm(fixations_a[i] - fixations_b[j]))

    def similarity(distances, normalizer):
        return max(0.0, 1.0 - (sum(distances) / len(distances)) / normalizer)

    scores = {
        "vector": similarity(vector_d, screen_diagonal),
        "length": similarity(length_d, screen_diagonal),
        "direction": similarity(direction_d, np.pi),
        "position": similarity(position_d, screen_diagonal),
    }
    scores["multimatch"] = sum(scores.values()) / len(scores)
    return scores


# ---------------------------------------------------------------------------
# Cross-Recurrence Quantification Analysis (CRQA)
# ---------------------------------------------------------------------------


def _run_points(mask_1d, min_line):
    """Sum of run-lengths (each >= `min_line`) of contiguous `True` values in a 1D
    bool array -- e.g. `[1,1,1,0,1,1]` with `min_line=2` -> `3 + 2 = 5`."""
    total, run = 0, 0
    for v in mask_1d:
        if v:
            run += 1
        else:
            total += run if run >= min_line else 0
            run = 0
    total += run if run >= min_line else 0
    return total


def _recurrent_line_points(R, min_line, axis):
    """Number of `True` entries of the `(n, m)` recurrence matrix `R` that belong to a
    line of length >= `min_line` along `axis` ("diag", "vertical" or "horizontal").
    Each axis partitions `R`'s entries exactly once, so the result is always
    `<= R.sum()`."""
    n, m = R.shape
    if axis == "diag":
        return sum(_run_points(np.diagonal(R, offset=k), min_line) for k in range(-(n - 1), m))
    if axis == "vertical":
        return sum(_run_points(R[:, j], min_line) for j in range(m))
    if axis == "horizontal":
        return sum(_run_points(R[i, :], min_line) for i in range(n))
    raise ValueError(f"unknown axis {axis!r}")


def cross_recurrence_analysis(fixations_a, fixations_b, radius=0.05, min_line=2):
    """Cross-Recurrence Quantification Analysis (CRQA) between two scanpaths'
    fixation positions (Anderson, Bischof, Laidlaw, Risko & Kingstone, 2013,
    "Recurrence quantification analysis of eye movements"): builds the cross-
    recurrence matrix `R[i, j] = 1` if fixation `i` of `a` and fixation `j` of `b` are
    within `radius` of each other, then summarizes it with 4 standard measures.

    `fixations_a`/`fixations_b`: `(n, 2)`/`(m, 2)` arrays of `(x, y)` positions.
    `radius`: recurrence threshold, in the same units as the positions (default 0.05,
    i.e. 5% of the unit square for `[0, 1]`-normalized coordinates -- tune to the
    coordinate scale / typical AOI size). `min_line`: minimum run length, in
    consecutive recurrent points, counted as a deterministic/laminar structure rather
    than an isolated coincidental recurrence.

    Returns a dict:
      "rec"  -- %recurrence: fraction of the `(n, m)` grid that is recurrent.
      "det"  -- %determinism: fraction of recurrent points on a diagonal line of
                length >= `min_line` (i.e. part of a sub-sequence both scanpaths
                visit, in the same relative order) -- gaze-*pattern* similarity.
      "lam"  -- %laminarity: fraction of recurrent points on a vertical or horizontal
                line of length >= `min_line` (one scanpath dwells on/revisits a spot
                the other visits once) -- fixation-clustering asymmetry.
      "corm" -- center of recurrence mass, in `[-100, 100]`: ~0 if recurrent points
                are symmetric around the main diagonal (the two scanpaths pass
                through shared locations at the same relative pace); positive/negative
                if `a`'s recurrences systematically lag/lead `b`'s.
    Returns all-`None` if either scanpath is empty.
    """
    fixations_a = np.asarray(fixations_a, dtype=np.float64)
    fixations_b = np.asarray(fixations_b, dtype=np.float64)
    n, m = fixations_a.shape[0], fixations_b.shape[0]
    if n == 0 or m == 0:
        return {"rec": None, "det": None, "lam": None, "corm": None}

    dists = np.linalg.norm(fixations_a[:, None, :] - fixations_b[None, :, :], axis=-1)
    R = dists <= radius  # (n, m)

    n_rec = int(R.sum())
    rec = 100.0 * n_rec / (n * m)
    if n_rec == 0:
        return {"rec": rec, "det": 0.0, "lam": 0.0, "corm": 0.0}

    det = 100.0 * _recurrent_line_points(R, min_line, "diag") / n_rec
    vert = _recurrent_line_points(R, min_line, "vertical")
    horiz = _recurrent_line_points(R, min_line, "horizontal")
    lam = 100.0 * (vert + horiz) / (2 * n_rec)

    ii, jj = np.nonzero(R)
    denom = max(n, m) - 1
    corm = 100.0 * float(np.sum(jj - ii)) / (denom * n_rec) if denom > 0 else 0.0

    return {"rec": rec, "det": det, "lam": lam, "corm": corm}


if __name__ == "__main__":
    # Sanity checks against hand-verifiable / textbook cases, not exhaustive brute-force
    # cross-validation (unlike permutation_distances.py, these methods don't have a
    # meaningfully different independent reference implementation to check against).

    # Levenshtein: the classic "kitten" -> "sitting" example, distance 3.
    d = _levenshtein(list("kitten"), list("sitting"))
    assert d == 3, f"expected 3, got {d}"
    print(f'_levenshtein("kitten", "sitting") = {d} (expected 3)')

    rng = np.random.default_rng(0)
    scanpath = rng.uniform(0.0, 1.0, size=(12, 2))
    scanpath[0], scanpath[1] = [0.1, 0.1], [0.9, 0.9]  # force >1 AOI so binning isn't trivial

    print(f"\nstring_edit_distance(scanpath, scanpath) = {string_edit_distance(scanpath, scanpath)} (expected 0)")
    print(f"scanmatch(scanpath, scanpath) = {scanmatch(scanpath, scanpath):.6f} (expected 1.0)")

    mm_self = multimatch(scanpath, scanpath)
    print(f"multimatch(scanpath, scanpath) = {mm_self}")
    for dim, score in mm_self.items():
        assert abs(score - 1.0) < 1e-9, f"{dim}: expected 1.0, got {score}"

    cra_self = cross_recurrence_analysis(scanpath, scanpath, radius=0.05)
    print(f"cross_recurrence_analysis(scanpath, scanpath) = {cra_self}")
    n = scanpath.shape[0]
    assert abs(cra_self["det"] - 100.0) < 1e-9, cra_self  # identical paths: main diagonal fully recurrent
    assert abs(cra_self["corm"]) < 1e-9, cra_self  # symmetric about the main diagonal

    shuffled = scanpath[rng.permutation(n)]
    print(f"\nvs. a shuffled copy of itself (same points, different order):")
    print(f"  string_edit_distance = {string_edit_distance(scanpath, shuffled)}")
    print(f"  scanmatch            = {scanmatch(scanpath, shuffled):.6f}")
    print(f"  multimatch            = {multimatch(scanpath, shuffled)}")
    print(f"  cross_recurrence      = {cross_recurrence_analysis(scanpath, shuffled, radius=0.05)}")

    far = scanpath + 5.0  # shift entirely out of the unit square / any reasonable radius
    print(f"\nvs. a copy shifted far away:")
    print(f"  string_edit_distance = {string_edit_distance(scanpath, far, bounds=((0, 6), (0, 6)))}")
    print(f"  scanmatch            = {scanmatch(scanpath, far, bounds=((0, 6), (0, 6))):.6f}")
    print(f"  multimatch            = {multimatch(scanpath, far)}")
    print(f"  cross_recurrence      = {cross_recurrence_analysis(scanpath, far, radius=0.05)}")

    print("\nAll sanity checks passed.")
