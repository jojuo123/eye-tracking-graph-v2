"""Distance metrics between two permutations of the same n items (e.g. a predicted vs.
ground-truth fixation ordering).

A permutation is a 1D array `p` of length `n` where `p[i]` is the label occupying
position `i`; `a` and `b` must contain the same set of `n` labels (they need not
already be `0..n-1` -- labels are canonicalized internally). Every function also
accepts a batch of permutation pairs (2D arrays, one permutation per row) and returns
one distance per row instead of a scalar.

    hamming_distance            -- positions where the two orderings disagree
    kendall_tau_distance        -- pairwise disagreements (== adjacent transpositions
                                    needed to sort one into the other)
    windowed_kendall_tau_distance
                                -- Kendall tau that ignores pairs closer than a
                                    DTW-style (Sakoe-Chiba) warping window in the
                                    reference ordering, so local reorderings are free
    spearman_footrule_distance  -- total absolute rank displacement, summed over items
    cayley_distance             -- minimum arbitrary transpositions to sort one into
                                    the other (== n minus the number of cycles in the
                                    relative permutation)
    ulam_distance                -- minimum insert/delete "move to front/back" ops to
                                    sort one into the other (== n minus the length of
                                    the longest increasing subsequence of the relative
                                    permutation, equivalently n minus the longest common
                                    subsequence of a and b)

Reference: P. Diaconis & R. Graham, "Spearman's Footrule as a Measure of Disarray"
(1977), for Hamming/Kendall tau/footrule/Cayley; S. Ulam's metric is the classic
edit-distance-by-LIS result for permutations.
"""

import bisect

import numpy as np


def _canon_labels(a, b):
    """1D arrays `a`, `b` holding the same `n` distinct (but arbitrary) labels ->
    `(a_idx, b_idx)`, both permutations of `0..n-1` obtained by remapping each label to
    its rank among `a`'s sorted unique values.
    """
    a, b = np.asarray(a), np.asarray(b)
    labels = np.sort(np.unique(a))
    canon = {label: i for i, label in enumerate(labels.tolist())}
    a_idx = np.array([canon[v] for v in a.tolist()], dtype=np.int64)
    b_idx = np.array([canon[v] for v in b.tolist()], dtype=np.int64)
    return a_idx, b_idx


def _relative_permutation(a, b):
    """`a`, `b` (same n labels) -> `pi`, the permutation of `0..n-1` with `pi[i]` = the
    position of `a[i]`'s label within `b`. `pi` expresses `a` "relative to" `b`: it is
    the identity iff `a == b` as orderings. Kendall tau, Cayley and Ulam distance are
    each a simple property of `pi`.
    """
    a_idx, b_idx = _canon_labels(a, b)
    inv_b = np.argsort(b_idx)  # inv_b[v] = position of value v within b_idx
    return inv_b[a_idx]


def _count_inversions(pi):
    """Number of pairs `i < j` with `pi[i] > pi[j]`, via a Fenwick tree -- O(n log n)."""
    n = len(pi)
    tree = [0] * (n + 1)

    def update(i):
        i += 1
        while i <= n:
            tree[i] += 1
            i += i & (-i)

    def query(i):
        i += 1
        total = 0
        while i > 0:
            total += tree[i]
            i -= i & (-i)
        return total

    inversions = 0
    for x in reversed(pi):
        inversions += query(x - 1)  # already-seen (i.e. to the right) values < x
        update(x)
    return inversions


def _num_cycles(pi):
    """Number of cycles of the permutation `pi` (mapping index -> value)."""
    n = len(pi)
    visited = [False] * n
    cycles = 0
    for i in range(n):
        if visited[i]:
            continue
        cycles += 1
        j = i
        while not visited[j]:
            visited[j] = True
            j = pi[j]
    return cycles


def _lis_length(pi):
    """Length of the longest strictly increasing subsequence of `pi`, via patience
    sorting -- O(n log n)."""
    tails = []
    for x in pi:
        i = bisect.bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


def _pairwise(distance_fn):
    """Wraps a single-pair (1D `a`, `b` -> scalar) distance so it also accepts a batch
    of pairs (2D `a`, `b`, one permutation per row -> 1D array of per-row distances).
    Any extra arguments (e.g. `window`) are passed through unchanged to every pair."""

    def wrapped(a, b, *args, **kwargs):
        a, b = np.asarray(a), np.asarray(b)
        if a.ndim == 1:
            return distance_fn(a, b, *args, **kwargs)
        return np.array([distance_fn(a[i], b[i], *args, **kwargs) for i in range(a.shape[0])])

    return wrapped


@_pairwise
def hamming_distance(a, b):
    """Number of positions at which `a` and `b` hold a different label. Range:
    `[0, n]` (0 excluded unless `n == 0`, since disagreeing in exactly one position is
    impossible for a permutation)."""
    a, b = np.asarray(a), np.asarray(b)
    return int(np.sum(a != b))


@_pairwise
def spearman_footrule_distance(a, b):
    """Sum, over every item, of the absolute difference between its position in `a`
    and its position in `b`. Range: `[0, floor(n^2 / 2)]`."""
    a_idx, b_idx = _canon_labels(a, b)
    pos_a, pos_b = np.argsort(a_idx), np.argsort(b_idx)  # pos[v] = position of value v
    return int(np.sum(np.abs(pos_a - pos_b)))


@_pairwise
def kendall_tau_distance(a, b):
    """Number of item pairs `(u, v)` ordered differently by `a` and `b` -- equivalently
    the minimum number of *adjacent* transpositions needed to turn `a` into `b`. Range:
    `[0, n * (n - 1) / 2]`."""
    return _count_inversions(_relative_permutation(a, b).tolist())


def windowed_kendall_max(n, window):
    """Number of position pairs `i < j` of an `n`-item ordering with `j - i > window` --
    the pairs `windowed_kendall_tau_distance` counts, and so its maximum (reached by
    the full reversal, which disorders every pair)."""
    window = max(int(window), 0)
    if window >= n - 1:
        return 0
    k = n - 1 - window  # number of distinct separations d = window+1 .. n-1
    return k * (k + 1) // 2  # sum_{d=window+1}^{n-1} (n - d)


@_pairwise
def windowed_kendall_tau_distance(a, b, window=0):
    """Kendall tau distance restricted to item pairs that sit more than `window`
    positions apart in `a` (the reference ordering, e.g. ground truth): a pair `(u, v)`
    with `|pos_a(u) - pos_a(v)| <= window` is never counted, however `b` orders it.
    This borrows DTW's Sakoe-Chiba band -- reorderings local enough to fit inside the
    warping window are tolerated, the way DTW tolerates local time misalignment -- while
    keeping Kendall's identity-based, pairwise view of order. `window=0` is plain
    `kendall_tau_distance`. Range: `[0, windowed_kendall_max(n, window)]`."""
    pi = _relative_permutation(a, b)  # pi[i] = position in b of the item at position i of a
    n = len(pi)
    i, j = np.triu_indices(n, k=max(int(window), 0) + 1)
    return int(np.sum(pi[i] > pi[j]))


@_pairwise
def cayley_distance(a, b):
    """Minimum number of (not necessarily adjacent) transpositions needed to turn `a`
    into `b`, i.e. `n` minus the number of cycles of the relative permutation. Range:
    `[0, n - 1]`."""
    pi = _relative_permutation(a, b).tolist()
    return len(pi) - _num_cycles(pi)


@_pairwise
def ulam_distance(a, b):
    """Minimum number of "remove an item and reinsert it elsewhere" moves needed to
    turn `a` into `b`, i.e. `n` minus the length of the longest increasing subsequence
    of the relative permutation (equivalently `n` minus the longest common subsequence
    of `a` and `b`). Range: `[0, n - 1]`."""
    pi = _relative_permutation(a, b).tolist()
    return len(pi) - _lis_length(pi)


DISTANCES = {
    "hamming": hamming_distance,
    "kendall_tau": kendall_tau_distance,
    "windowed_kendall_tau": windowed_kendall_tau_distance,
    "spearman_footrule": spearman_footrule_distance,
    "cayley": cayley_distance,
    "ulam": ulam_distance,
}


if __name__ == "__main__":
    # Brute-force cross-check against textbook definitions on random small permutations,
    # plus a couple of hand-checkable examples.
    rng = np.random.default_rng(0)

    def brute_hamming(a, b):
        return int(np.sum(np.asarray(a) != np.asarray(b)))

    def brute_kendall_tau(a, b):
        a, b = np.asarray(a), np.asarray(b)
        pos_a = {v: i for i, v in enumerate(a.tolist())}
        pos_b = {v: i for i, v in enumerate(b.tolist())}
        n = len(a)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                u, v = a[i], a[j]
                if (pos_a[u] - pos_a[v]) * (pos_b[u] - pos_b[v]) < 0:
                    count += 1
        return count

    def brute_footrule(a, b):
        a, b = np.asarray(a), np.asarray(b)
        pos_a = {v: i for i, v in enumerate(a.tolist())}
        pos_b = {v: i for i, v in enumerate(b.tolist())}
        return sum(abs(pos_a[v] - pos_b[v]) for v in a.tolist())

    def brute_cayley(a, b):
        # Minimum transpositions by brute-force cycle count on the composed permutation.
        a_idx, b_idx = _canon_labels(a, b)
        inv_b = np.argsort(b_idx)
        pi = inv_b[a_idx].tolist()
        return len(pi) - _num_cycles(pi)

    def brute_ulam(a, b):
        # n minus the length of the longest common subsequence of a and b (classic DP).
        a, b = list(np.asarray(a).tolist()), list(np.asarray(b).tolist())
        n, m = len(a), len(b)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return n - dp[n][m]

    checks = {
        "hamming": brute_hamming,
        "kendall_tau": brute_kendall_tau,
        "windowed_kendall_tau": brute_kendall_tau,  # window=0 default == plain Kendall tau
        "spearman_footrule": brute_footrule,
        "cayley": brute_cayley,
        "ulam": brute_ulam,
    }

    for trial in range(200):
        n = rng.integers(1, 9)
        a = rng.permutation(n)
        b = rng.permutation(n)
        for name, fn in DISTANCES.items():
            got = fn(a, b)
            want = checks[name](a, b)
            assert got == want, f"{name}(trial {trial}, n={n}): got {got}, want {want} (a={a}, b={b})"
    print("All 200 random single-pair trials match brute-force reference implementations.")

    def brute_windowed_kendall(a, b, window):
        a, b = np.asarray(a).tolist(), np.asarray(b).tolist()
        pos_b = {v: i for i, v in enumerate(b)}
        return sum(
            1
            for i in range(len(a))
            for j in range(i + window + 1, len(a))
            if pos_b[a[i]] > pos_b[a[j]]
        )

    for trial in range(200):
        n = int(rng.integers(1, 12))
        window = int(rng.integers(0, n + 1))
        a, b = rng.permutation(n), rng.permutation(n)
        got, want = windowed_kendall_tau_distance(a, b, window), brute_windowed_kendall(a, b, window)
        assert got == want, f"windowed_kendall(n={n}, w={window}): got {got}, want {want}"
        assert windowed_kendall_tau_distance(a, a[::-1], window) == windowed_kendall_max(n, window)
    print("All 200 windowed Kendall tau trials match brute force; reversal hits windowed_kendall_max.")

    a = np.array([0, 1, 2, 3, 4])
    b = np.array([4, 3, 2, 1, 0])
    print(f"\na = {a}\nb = {b} (fully reversed)")
    for name, fn in DISTANCES.items():
        print(f"  {name:>18s} = {fn(a, b)}")

    a_batch = np.stack([np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3])])
    b_batch = np.stack([np.array([0, 1, 2, 3]), np.array([3, 2, 1, 0])])
    print(f"\nbatched: a={a_batch.tolist()}, b={b_batch.tolist()}")
    for name, fn in DISTANCES.items():
        print(f"  {name:>18s} = {fn(a_batch, b_batch)}")
