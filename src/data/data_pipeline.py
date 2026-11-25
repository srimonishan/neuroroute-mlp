"""data_pipeline.py - Member 1: data loading, tour generation, dataset building."""

import os
import numpy as np
import pandas as pd


# (this file lives in src/data/, so the project root is three levels up)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "tsp_data_100_distance_matrix.csv")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
OUT_PATH = os.path.join(OUT_DIR, "processed.npz")

# ---- Settings (change here if the group decides differently) ----
N_CITIES = 20        
N_INSTANCES = 500    
SEED = 42            


def load_distance_matrix(path=CSV_PATH):
    """Read the CSV file and return a 100x100 numpy distance matrix."""
    df = pd.read_csv(path, index_col=0)
    return df.values.astype(float)


def validate_matrix(D):
    """Check that the distance matrix is valid and print a summary."""
    n = D.shape[0]
    print("Shape            :", D.shape)
    print("Square matrix    :", D.shape[0] == D.shape[1])
    print("Missing values   :", int(np.isnan(D).sum()))
    print("Symmetric        :", np.allclose(D, D.T))
    print("Diagonal all 0   :", np.allclose(np.diag(D), 0))
    off_diag = D[~np.eye(n, dtype=bool)]
    print("Min distance     :", off_diag.min())
    print("Max distance     :", off_diag.max())
    print("Mean distance    :", round(off_diag.mean(), 2))
    # Duplicate cities would have identical distance rows
    n_dup = n - len({tuple(row) for row in D})
    print("Duplicate cities :", n_dup)


def tour_length(D, tour):
    """Total length of a closed tour (returns to the starting city)."""
    n = len(tour)
    return sum(D[tour[i], tour[(i + 1) % n]] for i in range(n))


def nearest_neighbour(D, start=0):
    """Build a tour by always moving to the closest unvisited city."""
    n = D.shape[0]
    visited = [False] * n
    tour = [start]
    visited[start] = True
    current = start
    for _ in range(n - 1):
        best_city, best_dist = None, float("inf")
        for city in range(n):
            if not visited[city] and D[current, city] < best_dist:
                best_city, best_dist = city, D[current, city]
        tour.append(best_city)
        visited[best_city] = True
        current = best_city
    return tour


def two_opt(D, tour):
    """Improve a tour by reversing segments while the length keeps decreasing."""
    tour = tour[:]
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]
                # Compare the two edges removed vs the two edges added
                if D[a, c] + D[b, d] < D[a, b] + D[c, d] - 1e-9:
                    tour[i:j + 1] = reversed(tour[i:j + 1])
                    improved = True
    return tour


def make_instances(D, n_instances, n_cities, rng):
    """Create sub-problems by sampling n_cities random cities from D.

    For each instance, a good tour (Nearest Neighbour + 2-opt) is computed
    and rotated so that it always starts at city 0 of the sub-problem.
    The tour direction is fixed (second city index < last city index), so
    the two equivalent directions of a tour never give conflicting labels.
    Returns the sub-matrices, their tours, and the original city indices.
    """
    matrices, tours, indices = [], [], []
    for _ in range(n_instances):
        idx = rng.choice(D.shape[0], size=n_cities, replace=False)
        sub = D[np.ix_(idx, idx)]
        tour = two_opt(sub, nearest_neighbour(sub, start=0))
        k = tour.index(0)
        tour = tour[k:] + tour[:k]          # rotate so tour starts at city 0
        if tour[1] > tour[-1]:              # canonical direction
            tour = [tour[0]] + tour[1:][::-1]
        matrices.append(sub)
        tours.append(tour)
        indices.append(idx)
    return np.array(matrices), np.array(tours), np.array(indices)


def tour_to_samples(sub, tour):
    """Turn one tour into n-1 training samples.

    Features (3n values): one-hot current city, visited mask,
    distances from the current city (divided by the instance max distance).
    Label: the next city in the tour.
    """
    n = len(tour)
    scale = sub.max()
    X, y = [], []
    visited = np.zeros(n)
    for t in range(n - 1):
        cur = tour[t]
        visited[cur] = 1
        onehot = np.zeros(n)
        onehot[cur] = 1
        dists = sub[cur] / scale
        X.append(np.concatenate([onehot, visited, dists]))
        y.append(tour[t + 1])
    return np.array(X), np.array(y)


def build_split(matrices, tours):
    """Build X, y arrays from a group of instances."""
    Xs, ys = [], []
    for sub, tour in zip(matrices, tours):
        X, y = tour_to_samples(sub, tour)
        Xs.append(X)
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def build_dataset(D, n_instances=N_INSTANCES, n_cities=N_CITIES, seed=SEED):
    """Full pipeline: instances -> split (70/15/15) -> samples -> .npz file."""
    rng = np.random.default_rng(seed)
    matrices, tours, indices = make_instances(D, n_instances, n_cities, rng)

    # Split on INSTANCES first, so no tour is shared between train and test
    n_train = int(0.70 * n_instances)
    n_val = int(0.15 * n_instances)
    tr = slice(0, n_train)
    va = slice(n_train, n_train + n_val)
    te = slice(n_train + n_val, n_instances)

    X_train, y_train = build_split(matrices[tr], tours[tr])
    X_val, y_val = build_split(matrices[va], tours[va])
    X_test, y_test = build_split(matrices[te], tours[te])

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(
        OUT_PATH,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        test_matrices=matrices[te], test_tours=tours[te],  # for tour-length evaluation
        # Split manifest: original city indices of every instance in each split
        train_city_idx=indices[tr], val_city_idx=indices[va], test_city_idx=indices[te],
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    D = load_distance_matrix()
    validate_matrix(D)

    # Baseline on the full 100-city instance
    nn_tour = nearest_neighbour(D, start=0)
    opt_tour = two_opt(D, nn_tour)
    print("\nFull 100-city baseline")
    print("Nearest Neighbour length:", round(tour_length(D, nn_tour), 2))
    print("After 2-opt length      :", round(tour_length(D, opt_tour), 2))
    print("Valid tour              :", sorted(opt_tour) == list(range(100)))

    # Build the MLP dataset
    print("\nBuilding dataset...")
    Xtr, ytr, Xva, yva, Xte, yte = build_dataset(D)
    print("X_train:", Xtr.shape, " y_train:", ytr.shape)
    print("X_val  :", Xva.shape, " y_val  :", yva.shape)
    print("X_test :", Xte.shape, " y_test :", yte.shape)
    print("Classes:", N_CITIES, " Features:", Xtr.shape[1])
    print("Saved to:", OUT_PATH)
