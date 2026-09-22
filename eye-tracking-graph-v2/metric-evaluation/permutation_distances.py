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
    of pairs (2D `a`, `b`, one permutation per row -> 1D array of per-row distances)."""

    def wrapped(a, b):
        a, b = np.asarray(a), np.asarray(b)
        if a.ndim == 1:
            return distance_fn(a, b)
        return np.array([distance_fn(a[i], b[i]) for i in range(a.shape[0])])

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
