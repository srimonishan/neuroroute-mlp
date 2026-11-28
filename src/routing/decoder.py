"""decoder.py - TSP route decoding and tour validation."""

import numpy as np

from src.data.data_pipeline import nearest_neighbour


def tour_length(D, tour):
    """Total length of a closed tour, including the return edge."""
    D_arr = np.asarray(D, dtype=np.float64)
    n = len(tour)
    if n == 0:
        raise ValueError("Tour cannot be empty.")
    return float(sum(D_arr[tour[k], tour[(k + 1) % n]] for k in range(n)))


def validate_tour(D, tour, start=0):
    """Check Hamiltonian cycle, start/return city, and valid edge weights."""
    D_arr = np.asarray(D, dtype=np.float64)
    if D_arr.ndim != 2 or D_arr.shape[0] != D_arr.shape[1]:
        return False, f"Distance matrix must be square, got {D_arr.shape}."

    n = D_arr.shape[0]
    if len(tour) != n:
        return False, f"Expected tour of length {n}, got {len(tour)}."

    tour_list = [int(c) for c in tour]
    if tour_list[0] != int(start):
        return False, f"Tour must start at city {start}, got {tour_list[0]}."

    if set(tour_list) != set(range(n)):
        return False, "Tour does not visit each city in 0..n-1 exactly once."

    for k in range(n):
        u, v = tour_list[k], tour_list[(k + 1) % n]
        cost = D_arr[u, v]
        if not np.isfinite(cost) or cost < 0.0:
            return False, f"Invalid edge ({u} -> {v}) with cost {cost}."

    return True, "valid"


def build_step_features(D, current_city, visited_mask, scale=None):
    """Build the 3n feature vector matching data_pipeline.tour_to_samples."""
    D_arr = np.asarray(D, dtype=np.float64)
    n = D_arr.shape[0]

    if scale is None:
        finite_vals = D_arr[np.isfinite(D_arr)]
        max_val = float(np.max(finite_vals)) if finite_vals.size > 0 else 1.0
        scale = max_val if max_val > 0.0 else 1.0

    onehot = np.zeros(n, dtype=np.float64)
    onehot[current_city] = 1.0

    visited = np.asarray(visited_mask, dtype=np.float64)
    dists = np.where(np.isfinite(D_arr[current_city]), D_arr[current_city] / scale, 1.0)
    return np.concatenate([onehot, visited, dists])


def decode_tour(model, D, start=0):
    """Decode a valid closed TSP tour step by step, masking visited cities."""
    D_arr = np.asarray(D, dtype=np.float64)
    if D_arr.ndim != 2 or D_arr.shape[0] != D_arr.shape[1]:
        raise ValueError(f"Distance matrix must be square, got {D_arr.shape}.")

    n = D_arr.shape[0]
    if n < 2:
        raise ValueError("TSP decoding requires at least 2 cities.")
    if hasattr(model, "input_dim") and model.input_dim != 3 * n:
        raise ValueError(
            f"Model expects input_dim={model.input_dim}, got {n} cities (3n={3 * n})."
        )
    if hasattr(model, "output_dim") and model.output_dim != n:
        raise ValueError(f"Model expects output_dim={model.output_dim}, got {n} cities.")
    if not (0 <= start < n):
        raise ValueError(f"start city must be in [0, {n - 1}], got {start}.")

    finite_vals = D_arr[np.isfinite(D_arr)]
    max_val = float(np.max(finite_vals)) if finite_vals.size > 0 else 1.0
    scale = max_val if max_val > 0.0 else 1.0

    prev_mode = getattr(model, "training", None)
    if hasattr(model, "eval"):
        model.eval()

    try:
        visited = np.zeros(n, dtype=np.float64)
        tour = [int(start)]
        current = int(start)

        for _ in range(n - 1):
            visited[current] = 1.0
            features = build_step_features(D_arr, current, visited, scale=scale)
            logits = model.forward(features[None, :])[0]

            invalid = (visited == 1.0) | (~np.isfinite(D_arr[current])) | (D_arr[current] < 0.0)
            if np.all(invalid):
                raise ValueError(f"No valid unvisited edges available from city {current}.")

            masked_logits = np.where(invalid, -np.inf, logits)
            next_city = int(np.argmax(masked_logits))
            tour.append(next_city)
            current = next_city

        return tour
    finally:
        if prev_mode is True and hasattr(model, "train"):
            model.train()


def evaluate_routing(model, matrices, ref_tours=None, start=0):
    """Evaluate decoded tour validity and costs across a batch of distance matrices."""
    mats = np.asarray(matrices, dtype=np.float64)
    if mats.ndim != 3:
        raise ValueError(f"Expected 3D matrices (K, n, n), got {mats.shape}.")

    num_instances = mats.shape[0]
    tours = []
    model_costs = []
    nn_costs = []
    ref_costs = []
    valid_count = 0

    for idx in range(num_instances):
        D = mats[idx]
        tour = decode_tour(model, D, start=start)
        is_valid, _ = validate_tour(D, tour, start=start)
        if is_valid:
            valid_count += 1
        tours.append(tour)
        model_costs.append(tour_length(D, tour))
        nn_costs.append(tour_length(D, nearest_neighbour(D, start=start)))

        if ref_tours is not None:
            ref_costs.append(tour_length(D, ref_tours[idx]))

    result = {
        "num_instances": num_instances,
        "valid_tours": valid_count,
        "valid_tour_rate": 100.0 * valid_count / num_instances if num_instances > 0 else 0.0,
        "mean_model_cost": float(np.mean(model_costs)) if model_costs else 0.0,
        "mean_nn_cost": float(np.mean(nn_costs)) if nn_costs else 0.0,
        "model_costs": model_costs,
        "nn_costs": nn_costs,
        "tours": tours,
    }

    if ref_tours is not None and ref_costs:
        mean_ref = float(np.mean(ref_costs))
        gaps = [100.0 * (m - r) / r for m, r in zip(model_costs, ref_costs) if r > 0.0]
        result["mean_ref_cost"] = mean_ref
        result["mean_gap_vs_ref_pct"] = float(np.mean(gaps)) if gaps else 0.0
        result["ref_costs"] = ref_costs

    return result
