"""Distance/dissimilarity metrics between two continuous trajectories -- sequences of
real-valued `(x, y)` points, of possibly different length, compared directly (unlike
`scanpath_similarity.py`'s `string_edit_distance`/`scanmatch`, which first discretize
positions into a grid of AOI labels). Naming and definitions follow the `traj-dist`
(Besse, Guillouet, Malinowski & Journet, 2016) and `similaritymeasures` (Jekel et al.,
2019) reference packages; reimplemented here in plain numpy (no scipy dependency, to
match the rest of this directory) rather than depending on either.

    dynamic_time_warping       -- DTW: cheapest elementwise-distance alignment,
                                   letting either trajectory stretch locally to match
                                   the other's pace (Sakoe & Chiba, 1978)
    lcss_distance              -- Longest Common Subsequence with a spatial
                                   tolerance: how much of one trajectory is, in order,
                                   spatially close to the other, ignoring outliers
                                   entirely rather than dragging the alignment toward
                                   them (Vlachos, Kollios & Gunopulos, 2002)
    discrete_frechet_distance  -- discrete Frechet distance: the shortest "leash"
                                   needed to walk both trajectories forward-only, in
                                   step, without backtracking (Eiter & Mannila, 1994)
    hausdorff_distance         -- worst-case nearest-neighbor distance between the two
                                   point sets; the only metric here that ignores point
                                   order entirely
    edr_distance                -- Edit Distance on Real sequence: Levenshtein with a
                                   spatial-tolerance match test in place of exact
                                   equality (Chen, Ozsu & Oria, 2005)
    erp_distance                -- Edit distance with Real Penalty: like EDR, but
                                   charges the actual distance to a fixed gap point for
                                   skipped points and for mismatches, instead of a flat
                                   0/1 cost -- making it a true metric, unlike EDR/LCSS
                                   (Chen & Ng, 2004)

All functions take `(n, 2)`/`(m, 2)` arrays of `(x, y)` positions (`n` need not equal
`m`); `eps`/`radius`-style parameters are in the same units as the positions and
default to 0.05, matching `cross_recurrence_analysis`'s convention for `[0, 1]`-
normalized coordinates elsewhere in this directory.
"""

import numpy as np


def _pairwise_dist(a, b):
    """`(n, 2)`, `(m, 2)` -> `(n, m)` Euclidean distance matrix."""
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def dynamic_time_warping(traj_a, traj_b, normalize=False):
    """Dynamic Time Warping (Sakoe & Chiba, 1978) cumulative distance: the cheapest
    monotonic, contiguous alignment between `traj_a` and `traj_b` under elementwise
    Euclidean cost, where each point may be matched to one or more consecutive points
    of the other trajectory -- unlike Frechet/Hausdorff, no single point has to "give
    up" its whole match; DTW instead stretches either trajectory locally to keep them
    aligned, which also makes it sensitive to noise/outliers (every point is charged
    for, there's no tolerance radius to ignore one).

    `normalize`: if True, divide by `n + m - 1` (the optimal path's maximum possible
    length) to make scores comparable across differently-sized trajectory pairs.
    """
    traj_a, traj_b = np.asarray(traj_a, dtype=np.float64), np.asarray(traj_b, dtype=np.float64)
    n, m = traj_a.shape[0], traj_b.shape[0]
    cost = _pairwise_dist(traj_a, traj_b)

    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dtw[i, j] = cost[i - 1, j - 1] + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    distance = dtw[n, m]
    if normalize and (n + m - 1) > 0:
        distance = distance / (n + m - 1)
    return float(distance)


def lcss_distance(traj_a, traj_b, eps=0.05, delta=None, return_length=False):
    """Longest Common Subsequence distance with a spatial tolerance (Vlachos, Kollios
    & Gunopulos, 2002): two points "match" if within `eps` of each other (rather than
    requiring exact equality, as string LCSS does), and the standard LCSS recursion
    finds the longest such matching subsequence, in order -- skipping any number of
    unmatched points from either trajectory for free, so (unlike DTW) an outlier
    segment just gets skipped rather than dragging the whole alignment toward it.

    `eps`: match radius (default 0.05). `delta`: optional maximum index offset
    `|i - j|` allowed between matched points (`None` = unconstrained, the simplest
    form of the algorithm).

    Returns `1 - lcss_length / min(n, m)`, a dissimilarity in `[0, 1]` (0 = one
    trajectory is a spatial subsequence of the other), following `traj-dist`'s `lcss`
    convention. Pass `return_length=True` to instead get the raw LCSS length.
    """
    traj_a, traj_b = np.asarray(traj_a, dtype=np.float64), np.asarray(traj_b, dtype=np.float64)
    n, m = traj_a.shape[0], traj_b.shape[0]
    cost = _pairwise_dist(traj_a, traj_b)

    L = np.zeros((n + 1, m + 1), dtype=np.int64)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if (delta is None or abs(i - j) <= delta) and cost[i - 1, j - 1] <= eps:
                L[i, j] = L[i - 1, j - 1] + 1
            else:
                L[i, j] = max(L[i - 1, j], L[i, j - 1])

    length = int(L[n, m])
    if return_length:
        return length
    denom = min(n, m)
    return 1.0 - length / denom if denom > 0 else 0.0


def discrete_frechet_distance(traj_a, traj_b):
    """Discrete Frechet distance (Eiter & Mannila, 1994): the smallest `eps` such
    that a person walking `traj_a` and a person walking `traj_b`, both starting at
    index 0, ending at their last index, and each only ever staying put or advancing
    to their next point (never backtracking), can stay within `eps` of each other
    throughout -- computed via the standard coupling-search DP, not simulation. More
    sensitive to a single outlier point than DTW (which can dilute it into an
    average) or Hausdorff (which ignores order, so an outlier only matters via its
    single nearest neighbor)."""
    traj_a, traj_b = np.asarray(traj_a, dtype=np.float64), np.asarray(traj_b, dtype=np.float64)
    n, m = traj_a.shape[0], traj_b.shape[0]
    cost = _pairwise_dist(traj_a, traj_b)

    ca = np.zeros((n, m))
    ca[0, 0] = cost[0, 0]
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], cost[i, 0])
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], cost[0, j])
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(cost[i, j], min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]))
    return float(ca[n - 1, m - 1])


def hausdorff_distance(traj_a, traj_b, directed=False):
    """Hausdorff distance: the worst-case nearest-neighbor distance between the two
    point sets -- `max` over one trajectory's points of their `min` distance to the
    other. Unlike every other metric in this module, point *order* is irrelevant:
    only which locations were visited matters, not when or in what sequence, so two
    scanpaths that visit the same regions in opposite order score as identical.

    `directed`: if True, returns only `h(a -> b) = max_i min_j dist(a_i, b_j)` (how
    far `a`'s worst point is from its nearest neighbor in `b`); the (default)
    symmetric Hausdorff distance is `max(h(a -> b), h(b -> a))`.
    """
    traj_a, traj_b = np.asarray(traj_a, dtype=np.float64), np.asarray(traj_b, dtype=np.float64)
    cost = _pairwise_dist(traj_a, traj_b)
    a_to_b = cost.min(axis=1).max()
    if directed:
        return float(a_to_b)
    b_to_a = cost.min(axis=0).max()
    return float(max(a_to_b, b_to_a))


def edr_distance(traj_a, traj_b, eps=0.05, normalize=False):
    """Edit Distance on Real sequence (Chen, Ozsu & Oria, 2005): Levenshtein distance
    where the substitution cost is 0 if two points are within `eps` of each other and
    1 otherwise (in place of exact label equality, as
    `scanpath_similarity.string_edit_distance` uses on binned AOI labels) -- so, like
    LCSS, it tolerates noise within `eps`, but (unlike LCSS) every *unmatched* point
    still costs a flat 1 via the usual insertion/deletion penalty, rather than nothing.

    `eps`: match radius (default 0.05, matching `lcss_distance`). `normalize`: if
    True, divide by `max(n, m)`.
    """
    traj_a, traj_b = np.asarray(traj_a, dtype=np.float64), np.asarray(traj_b, dtype=np.float64)
    n, m = traj_a.shape[0], traj_b.shape[0]
    cost = _pairwise_dist(traj_a, traj_b)

    D = np.zeros((n + 1, m + 1))
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = 0.0 if cost[i - 1, j - 1] <= eps else 1.0
            D[i, j] = min(D[i - 1, j - 1] + sub_cost, D[i - 1, j] + 1, D[i, j - 1] + 1)

    distance = D[n, m]
    if normalize and max(n, m) > 0:
        distance = distance / max(n, m)
    return float(distance)


def erp_distance(traj_a, traj_b, gap_point=None):
    """Edit distance with Real Penalty (Chen & Ng, 2004): like `edr_distance`, but
    charges the *actual* Euclidean distance to a fixed reference point `gap_point`
    (rather than a flat penalty of 1) for every inserted/deleted point, and the
    actual distance between matched points (rather than a 0/1 threshold test) for
    substitutions -- there is accordingly no `eps` to tune. This makes ERP a true
    metric (it satisfies the triangle inequality), unlike EDR and LCSS.

    `gap_point`: the fixed reference `(x, y)` a skipped point is charged against;
    defaults to the origin `(0, 0)`, the choice used in the original paper and in
    `traj-dist`.
    """
    traj_a, traj_b = np.asarray(traj_a, dtype=np.float64), np.asarray(traj_b, dtype=np.float64)
    n, m = traj_a.shape[0], traj_b.shape[0]
    g = np.zeros(2) if gap_point is None else np.asarray(gap_point, dtype=np.float64)

    gap_a = np.linalg.norm(traj_a - g, axis=-1)  # (n,): cost of deleting a_i
    gap_b = np.linalg.norm(traj_b - g, axis=-1)  # (m,): cost of deleting b_j
    cost = _pairwise_dist(traj_a, traj_b)

    D = np.zeros((n + 1, m + 1))
    D[1:, 0] = np.cumsum(gap_a)
    D[0, 1:] = np.cumsum(gap_b)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(
                D[i - 1, j - 1] + cost[i - 1, j - 1],
                D[i - 1, j] + gap_a[i - 1],
                D[i, j - 1] + gap_b[j - 1],
            )
    return float(D[n, m])


TRAJECTORY_DISTANCES = {
    "dtw": dynamic_time_warping,
    "lcss": lcss_distance,
    "discrete_frechet": discrete_frechet_distance,
    "hausdorff": hausdorff_distance,
    "edr": edr_distance,
    "erp": erp_distance,
}


if __name__ == "__main__":
    # Sanity checks against hand-computable cases: an evenly-sampled straight line and
    # a copy of it shifted by a constant offset `d`, where every metric's value is
    # derivable by hand (rather than exhaustive brute-force cross-validation, unlike
    # permutation_distances.py -- most of these algorithms don't have a meaningfully
    # different independent reference to check against).
    n = 6
    d = 0.2
    traj_a = np.stack([np.linspace(0.0, 1.0, n), np.zeros(n)], axis=-1)
    traj_b = traj_a + np.array([0.0, d])  # same x's, shifted up by d -> nearest neighbor is index-aligned

    print("Identical trajectory (traj_a vs. itself):")
    identical = {
        "dtw": dynamic_time_warping(traj_a, traj_a),
        "lcss_distance": lcss_distance(traj_a, traj_a, eps=1e-9),
        "discrete_frechet_distance": discrete_frechet_distance(traj_a, traj_a),
        "hausdorff_distance": hausdorff_distance(traj_a, traj_a),
        "edr_distance": edr_distance(traj_a, traj_a, eps=1e-9),
        "erp_distance": erp_distance(traj_a, traj_a),
    }
    for name, value in identical.items():
        print(f"  {name:>26s} = {value}")
        assert abs(value) < 1e-9, f"{name}: expected 0, got {value}"

    print(f"\ntraj_a vs. a copy shifted by (0, {d}) (index-aligned nearest neighbors):")
    dtw = dynamic_time_warping(traj_a, traj_b)
    frechet = discrete_frechet_distance(traj_a, traj_b)
    hausdorff = hausdorff_distance(traj_a, traj_b)
    print(f"  dynamic_time_warping      = {dtw:.6f} (expected {n * d:.6f} = n * d)")
    print(f"  discrete_frechet_distance = {frechet:.6f} (expected {d:.6f} = d)")
    print(f"  hausdorff_distance        = {hausdorff:.6f} (expected {d:.6f} = d)")
    assert abs(dtw - n * d) < 1e-9
    assert abs(frechet - d) < 1e-9
    assert abs(hausdorff - d) < 1e-9

    for eps, expect_full_match in [(d * 1.5, True), (d * 0.5, False)]:
        lcss_len = lcss_distance(traj_a, traj_b, eps=eps, return_length=True)
        edr = edr_distance(traj_a, traj_b, eps=eps)
        print(f"  eps={eps:.3f}: lcss_length = {lcss_len} (expected {n if expect_full_match else 0}),"
              f" edr_distance = {edr} (expected {0 if expect_full_match else n})")
        assert lcss_len == (n if expect_full_match else 0)
        assert edr == (0 if expect_full_match else n)

    # ERP with the default gap point (origin): matching every point directly costs n*d
    # (every pair is at Euclidean distance d); since that beats any skip/gap route
    # (which pays real distances to the origin instead, always >= the direct route
    # here), the optimal alignment is the direct one.
    erp = erp_distance(traj_a, traj_b)
    print(f"  erp_distance (gap_point=origin) = {erp:.6f} (expected {n * d:.6f} = n * d)")
    assert abs(erp - n * d) < 1e-9

    print("\nAll sanity checks passed.")
