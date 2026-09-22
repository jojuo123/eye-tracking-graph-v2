"""Synthetic scanpath damage: 8 parametrized ways to corrupt the *order* of a fixation
sequence, for stress-testing the metrics in `permutation_distances.py`,
`scanpath_similarity.py` and `trajectory_distances.py` (`run_damage_evaluation.py`
drives that experiment).

Every generator below takes a sequence length `n`, a severity `level` in `[0, 1]`
(0 = no damage / identity, 1 = maximal damage for that family), and a
`numpy.random.Generator`, and returns `perm`, a permutation of `0..n-1` such that
`damaged = original[perm]` -- i.e. position `i` of the damaged scanpath holds whatever
fixation was at position `perm[i]` in the original. This keeps every perturbation a
pure reordering of the *same* fixations (nothing added, removed, or moved in space)
except `swap_nearby_locations`, which needs the actual `(x, y)` positions to find
which fixations are spatially close but temporally far apart.

    displaced_fixation      -- move one fixation from an early position to the end (or
                                the end to the start), shifting everything between by
                                one; `level` sets how early/late that fixation was.
    local_swaps              -- swap each disjoint adjacent pair `(2k, 2k+1)`
                                independently with probability `level`.
    triplet_internal_shuffle -- cut into consecutive triplets; each triplet
                                independently, with probability `level`, has its 3
                                fixations internally reordered (triplet *position*
                                in the sequence is preserved).
    triplet_order_shuffle    -- cut into consecutive triplets (each kept internally
                                intact); `level` fraction of triplets get their
                                *order* permuted among themselves.
    swap_nearby_locations    -- find fixation pairs that revisit almost the same
                                location (within `radius`) far apart in time (index
                                gap >= `min_gap`), and swap `level` fraction of them.
    reverse_segment           -- reverse one contiguous stretch of length
                                `~level * n`, placed at a random offset.
    global_reverse            -- reverse a growing prefix of length `~level * n`;
                                at `level=1` this reverses the entire path.
    global_shuffle            -- randomly permute (with no fixed points) `~level * n`
                                positions, chosen at random; at `level=1` this is a
                                fully random permutation of the whole sequence.

`PERTURBATIONS` maps each name to its function for iteration/discovery.
"""

import numpy as np


def _random_derangement(k, rng):
    """A uniformly random derangement (permutation with no fixed point) of `0..k-1`,
    via rejection sampling (expected ~e ~= 2.7 tries, independent of `k`)."""
    if k < 2:
        return np.arange(k)
    identity = np.arange(k)
    while True:
        candidate = rng.permutation(k)
        if np.all(candidate != identity):
            return candidate


def displaced_fixation(n, level, rng, coords=None):
    """Move a single fixation from an early position to the end, or from the end to
    an early position (chosen with equal probability each call), shifting every
    fixation in between back/forward by one. `level` sets the moved fixation's
    original distance from its destination, as a fraction of `n - 1`: `level=0` moves
    a fixation already adjacent to its destination (minimal shift of the rest of the
    path); `level=1` moves the fixation from the opposite end (every other fixation
    shifts by one)."""
    perm = np.arange(n)
    if n < 2 or level <= 0:
        return perm
    k = int(np.clip(round(level * (n - 1)), 1, n - 1))
    if rng.random() < 0.5:
        i = n - 1 - k  # move fixation i to the end
        return np.concatenate([perm[:i], perm[i + 1 :], perm[i : i + 1]])
    i = k  # move fixation i to the start
    return np.concatenate([perm[i : i + 1], perm[:i], perm[i + 1 :]])


def local_swaps(n, level, rng, coords=None):
    """Independently swap each disjoint adjacent pair `(0,1), (2,3), ...` with
    probability `level` (0 = no swaps, 1 = every disjoint pair swapped)."""
    perm = np.arange(n)
    for i in range(0, n - 1, 2):
        if rng.random() < level:
            perm[i], perm[i + 1] = perm[i + 1], perm[i]
    return perm


def triplet_internal_shuffle(n, level, rng, coords=None):
    """Cut the path into consecutive, non-overlapping triplets (a trailing remainder
    of 1-2 fixations, if `n` isn't a multiple of 3, is left untouched). Each triplet
    independently, with probability `level`, is replaced by a uniformly random
    *non-identity* reordering of its own 3 fixations -- triplet boundaries (which
    positions belong to which triplet) never move."""
    perm = np.arange(n)
    n_triplets = n // 3
    non_identity_perms3 = [
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ]
    for t in range(n_triplets):
        if rng.random() < level:
            s = t * 3
            block = perm[s : s + 3].copy()
            choice = non_identity_perms3[rng.integers(len(non_identity_perms3))]
            perm[s : s + 3] = block[list(choice)]
    return perm


def triplet_order_shuffle(n, level, rng, coords=None):
    """Cut the path into consecutive triplets (each kept internally intact -- the
    opposite of `triplet_internal_shuffle`); a `level` fraction of the triplet
    *slots* are chosen and their triplets reassigned among themselves via a random
    derangement (so every chosen slot's triplet actually moves)."""
    perm = np.arange(n)
    n_triplets = n // 3
    if n_triplets < 2:
        return perm
    k = int(round(level * n_triplets))
    if k < 2:
        return perm
    slots = np.sort(rng.choice(n_triplets, size=k, replace=False))
    blocks = [perm[t * 3 : t * 3 + 3].copy() for t in slots]
    order = _random_derangement(k, rng)
    new_perm = perm.copy()
    for dest_pos, block in zip(slots, [blocks[i] for i in order]):
        new_perm[dest_pos * 3 : dest_pos * 3 + 3] = block
    return new_perm


def swap_nearby_locations(n, level, rng, coords, radius=0.05, min_gap=None):
    """The "revisit" perturbation: finds fixation pairs `(i, j)` that are spatially
    close (`dist(coords[i], coords[j]) <= radius`) but temporally far apart
    (`j - i >= min_gap`, default `max(3, n // 10)`), greedily picks a maximal set of
    such pairs that don't share an index, and swaps a `level` fraction of them.
    Requires `coords`, the `(n, 2)` positions the permutation will be applied to --
    unlike every other perturbation here, this one is not purely combinatorial.
    Returns the identity permutation if no eligible pair exists (e.g. a scanpath with
    no near-revisits at this `radius`)."""
    perm = np.arange(n)
    if coords is None or n < 2:
        return perm
    if min_gap is None:
        min_gap = max(3, n // 10)

    coords = np.asarray(coords, dtype=np.float64)
    dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    ii, jj = np.triu_indices(n, k=1)
    eligible = (dists[ii, jj] <= radius) & (jj - ii >= min_gap)
    pairs = list(zip(ii[eligible].tolist(), jj[eligible].tolist()))
    if not pairs:
        return perm

    order = rng.permutation(len(pairs))
    used = set()
    disjoint_pairs = []
    for idx in order:
        i, j = pairs[idx]
        if i not in used and j not in used:
            disjoint_pairs.append((i, j))
            used.add(i)
            used.add(j)

    k = int(round(level * len(disjoint_pairs)))
    for i, j in disjoint_pairs[:k]:
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def reverse_segment(n, level, rng, coords=None):
    """Reverse one contiguous stretch of the path, length `~level * n` (minimum 2),
    placed at a uniformly random offset. `level=0` leaves the path untouched;
    `level=1` reverses a stretch as long as the whole path (though not necessarily
    the *whole* path itself, unless the random offset happens to be 0 -- for a
    deterministic full-path reversal see `global_reverse`)."""
    perm = np.arange(n)
    if n < 2 or level <= 0:
        return perm
    length = int(np.clip(round(level * n), 2, n))
    start = int(rng.integers(0, n - length + 1))
    perm[start : start + length] = perm[start : start + length][::-1]
    return perm


def global_reverse(n, level, rng, coords=None):
    """Reverse a growing *prefix* of the path, length `~level * n` (minimum 2):
    `level=0` is the identity, `level=1` reverses the entire path (prefix length
    `n` == reversing everything). Unlike `reverse_segment`, this is anchored at the
    start rather than placed randomly, so severity grows toward one fixed endpoint:
    total path reversal."""
    perm = np.arange(n)
    if n < 2 or level <= 0:
        return perm
    length = int(np.clip(round(level * n), 2, n))
    perm[:length] = perm[:length][::-1]
    return perm


def global_shuffle(n, level, rng, coords=None):
    """Choose `~level * n` positions at random (minimum 2) and randomly permute
    (with no fixed points, via a random derangement) exactly those positions' values,
    leaving the rest of the path untouched. `level=0` is the identity; `level=1`
    shuffles every position -- a uniformly random derangement of the whole path."""
    perm = np.arange(n)
    if n < 2 or level <= 0:
        return perm
    k = int(np.clip(round(level * n), 2, n))
    positions = np.sort(rng.choice(n, size=k, replace=False))
    values = perm[positions].copy()
    order = _random_derangement(k, rng)
    perm[positions] = values[order]
    return perm


PERTURBATIONS = {
    "displaced_fixation": displaced_fixation,
    "local_swaps": local_swaps,
    "triplet_internal_shuffle": triplet_internal_shuffle,
    "triplet_order_shuffle": triplet_order_shuffle,
    "swap_nearby_locations": swap_nearby_locations,
    "reverse_segment": reverse_segment,
    "global_reverse": global_reverse,
    "global_shuffle": global_shuffle,
}

# Which perturbations need the actual (x, y) positions (not just n) to operate.
NEEDS_COORDS = {"swap_nearby_locations"}


if __name__ == "__main__":
    from permutation_distances import kendall_tau_distance

    rng = np.random.default_rng(0)
    n = 60
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]

    # A synthetic scanpath with a handful of deliberate near-revisits (same spot,
    # far apart in time), so swap_nearby_locations has eligible pairs to work with.
    coords = rng.uniform(0.0, 1.0, size=(n, 2))
    for i, j in [(2, 40), (5, 45), (10, 50)]:
        coords[j] = coords[i] + rng.normal(scale=0.005, size=2)

    for name, fn in PERTURBATIONS.items():
        needs_coords = name in NEEDS_COORDS
        mean_kendall_by_level = []
        for level in levels:
            trials = []
            for trial in range(30):
                perm = fn(n, level, rng, coords=coords if needs_coords else None)
                assert np.array_equal(np.sort(perm), np.arange(n)), f"{name}@{level}: not a valid permutation"
                trials.append(kendall_tau_distance(np.arange(n), perm))
            mean_kendall_by_level.append(np.mean(trials))
        print(f"{name:>26s}: mean kendall_tau by level {levels} = {[round(v, 2) for v in mean_kendall_by_level]}")
        assert mean_kendall_by_level[0] == 0.0, f"{name}: level 0 should be damage-free"
        if name != "swap_nearby_locations":  # eligible-pair count is data-dependent, not guaranteed monotonic
            assert mean_kendall_by_level == sorted(mean_kendall_by_level), f"{name}: damage should grow with level"

    print("\nAll perturbations produce valid permutations with damage increasing in level.")
