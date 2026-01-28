import numpy as np

try:
    from sklearn.neighbors import NearestNeighbors
except Exception:  # pragma: no cover
    NearestNeighbors = None


def _as_intrinsics_matrix(K, fx=None, fy=None, cx=None, cy=None):
    if K is not None:
        K = np.asarray(K, dtype=np.float32)
        if K.shape == (3, 3):
            return K
        # Some pipelines store intrinsics as homogeneous 4x4, or as a 3x4 projection-like matrix.
        if K.shape == (4, 4):
            return K[:3, :3]
        if K.shape == (3, 4):
            return K[:3, :3]
        raise ValueError(f"camera_intrinsics must be 3x3 (or 4x4/3x4 with K in top-left), got {K.shape}")
    if fx is None or fy is None or cx is None or cy is None:
        raise ValueError("Missing intrinsics: provide 3x3 K or fx/fy/cx/cy")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def depth_to_pointcloud(depth, K, mask=None, max_points=None, rng=None):
    """Back-project a depth map to a point cloud in camera coordinates.

    Args:
        depth: [H, W] float array, depth in meters.
        K: [3, 3] intrinsics.
        mask: optional [H, W] bool array; True keeps point.
        max_points: optional int; random subsample.
        rng: optional np.random.Generator.

    Returns:
        points: [N, 3] float32 array.
    """
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth must be [H,W], got {depth.shape}")

    K = np.asarray(K, dtype=np.float32)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)

    ys, xs = np.nonzero(valid)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    z = depth[ys, xs]
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy

    pts = np.stack([x, y, z], axis=1).astype(np.float32)

    if max_points is not None and pts.shape[0] > int(max_points):
        if rng is None:
            rng = np.random.default_rng(0)
        idx = rng.choice(pts.shape[0], size=int(max_points), replace=False)
        pts = pts[idx]

    return pts


def depth_to_corresponding_pointclouds(depth_gt, depth_pred, K, mask=None):
    """Back-project GT+Pred depth using the same pixel mask for 1:1 point correspondence.

    Returns:
        pts_gt, pts_pr
    """
    depth_gt = np.asarray(depth_gt, dtype=np.float32)
    depth_pred = np.asarray(depth_pred, dtype=np.float32)
    if depth_gt.shape != depth_pred.shape:
        raise ValueError(f"gt/pred depth shape mismatch: {depth_gt.shape} vs {depth_pred.shape}")

    K = np.asarray(K, dtype=np.float32)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    valid = (
        np.isfinite(depth_gt)
        & np.isfinite(depth_pred)
        & (depth_gt > 0)
        & (depth_pred > 0)
    )
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)

    ys, xs = np.nonzero(valid)
    if ys.size == 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, empty

    z_gt = depth_gt[ys, xs]
    z_pr = depth_pred[ys, xs]

    x_gt = (xs.astype(np.float32) - cx) * z_gt / fx
    y_gt = (ys.astype(np.float32) - cy) * z_gt / fy
    x_pr = (xs.astype(np.float32) - cx) * z_pr / fx
    y_pr = (ys.astype(np.float32) - cy) * z_pr / fy

    pts_gt = np.stack([x_gt, y_gt, z_gt], axis=1).astype(np.float32)
    pts_pr = np.stack([x_pr, y_pr, z_pr], axis=1).astype(np.float32)
    return pts_gt, pts_pr


def depth_to_corresponding_pointclouds_with_pixels(depth_gt, depth_pred, K, mask=None):
    """Like depth_to_corresponding_pointclouds, but also returns pixel coordinates."""
    depth_gt = np.asarray(depth_gt, dtype=np.float32)
    depth_pred = np.asarray(depth_pred, dtype=np.float32)
    if depth_gt.shape != depth_pred.shape:
        raise ValueError(f"gt/pred depth shape mismatch: {depth_gt.shape} vs {depth_pred.shape}")

    K = np.asarray(K, dtype=np.float32)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    valid = (
        np.isfinite(depth_gt)
        & np.isfinite(depth_pred)
        & (depth_gt > 0)
        & (depth_pred > 0)
    )
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)

    ys, xs = np.nonzero(valid)
    if ys.size == 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, empty, ys.astype(np.int64), xs.astype(np.int64)

    z_gt = depth_gt[ys, xs]
    z_pr = depth_pred[ys, xs]

    x_gt = (xs.astype(np.float32) - cx) * z_gt / fx
    y_gt = (ys.astype(np.float32) - cy) * z_gt / fy
    x_pr = (xs.astype(np.float32) - cx) * z_pr / fx
    y_pr = (ys.astype(np.float32) - cy) * z_pr / fy

    pts_gt = np.stack([x_gt, y_gt, z_gt], axis=1).astype(np.float32)
    pts_pr = np.stack([x_pr, y_pr, z_pr], axis=1).astype(np.float32)
    return pts_gt, pts_pr, ys.astype(np.int64), xs.astype(np.int64)


def transform_points(points, T):
    """Apply 4x4 transform to Nx3 points."""
    points = np.asarray(points, dtype=np.float32)
    T = np.asarray(T, dtype=np.float32)
    if points.size == 0:
        return points
    if T.shape != (4, 4):
        raise ValueError(f"T must be 4x4, got {T.shape}")
    R = T[:3, :3]
    t = T[:3, 3]
    return (points @ R.T) + t


def _nn_distances(src, dst):
    """Compute nearest-neighbor distances from src->dst."""
    if NearestNeighbors is None:
        raise ImportError("scikit-learn is required for pointcloud NN metrics")
    if src.shape[0] == 0 or dst.shape[0] == 0:
        return np.full((src.shape[0],), np.inf, dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
    nn.fit(dst)
    dists, _ = nn.kneighbors(src, return_distance=True)
    return dists[:, 0].astype(np.float32)


def nn_query_1nn(src, dst):
    """1-NN query from src->dst.

    Returns:
        dists: [N] float32
        idx:   [N] int64 indices into dst
    """
    if NearestNeighbors is None:
        raise ImportError("scikit-learn is required for pointcloud NN metrics")
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    if src.shape[0] == 0 or dst.shape[0] == 0:
        return (
            np.full((src.shape[0],), np.inf, dtype=np.float32),
            np.full((src.shape[0],), -1, dtype=np.int64),
        )
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
    nn.fit(dst)
    dists, idx = nn.kneighbors(src, return_distance=True)
    return dists[:, 0].astype(np.float32), idx[:, 0].astype(np.int64)


def depth_to_normal_map(depth, K, mask=None, eps: float = 1e-6):
    """Estimate per-pixel normals from a depth map (camera coordinates).

    Uses cross products of forward differences in image space.
    Normal direction is ambiguous; downstream metrics should use |dot|.

    Returns:
        normals: [H, W, 3] float32 (zeros for invalid/border pixels)
    """
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth must be [H,W], got {depth.shape}")
    H, W = depth.shape
    K = np.asarray(K, dtype=np.float32)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)

    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    z = depth
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    P = np.stack([x, y, z], axis=2)  # [H,W,3]

    # Forward differences
    dx = P[:, 1:, :] - P[:, :-1, :]
    dy = P[1:, :, :] - P[:-1, :, :]

    # Cross product at [0:H-1,0:W-1]
    n = np.cross(dx[:-1, :, :], dy[:, :-1, :])

    # Normalize
    norm = np.linalg.norm(n, axis=2, keepdims=True)
    n = n / (norm + eps)

    normals = np.zeros((H, W, 3), dtype=np.float32)
    normals[:-1, :-1, :] = n.astype(np.float32)

    # Invalidate where depth invalid (center pixel) or neighbors invalid
    valid_center = valid.copy()
    valid_dx = valid[:, 1:] & valid[:, :-1]
    valid_dy = valid[1:, :] & valid[:-1, :]
    valid_n = valid_dx[:-1, :] & valid_dy[:, :-1] & valid_center[:-1, :-1]
    normals[:-1, :-1, :][~valid_n] = 0.0
    return normals


def normal_consistency_from_normals(src_normals, dst_normals, nn_index):
    """Compute mean/median |dot(n_src, n_dst[nn])| for NN correspondences."""
    src_n = np.asarray(src_normals, dtype=np.float32)
    dst_n = np.asarray(dst_normals, dtype=np.float32)
    nn_index = np.asarray(nn_index, dtype=np.int64)
    if src_n.shape[0] == 0 or dst_n.shape[0] == 0:
        return {"mean": float('nan'), "median": float('nan'), "n": 0}
    ok = (nn_index >= 0) & (nn_index < dst_n.shape[0])
    src_n = src_n[ok]
    nn_index = nn_index[ok]
    if src_n.shape[0] == 0:
        return {"mean": float('nan'), "median": float('nan'), "n": 0}
    dst_sel = dst_n[nn_index]

    # Filter out zero normals
    src_len = np.linalg.norm(src_n, axis=1)
    dst_len = np.linalg.norm(dst_sel, axis=1)
    keep = (src_len > 1e-6) & (dst_len > 1e-6)
    if not np.any(keep):
        return {"mean": float('nan'), "median": float('nan'), "n": 0}
    src_n = src_n[keep]
    dst_sel = dst_sel[keep]

    dots = np.abs(np.sum(src_n * dst_sel, axis=1)).astype(np.float32)
    return {"mean": float(np.mean(dots)), "median": float(np.median(dots)), "n": int(dots.shape[0])}


def align_scale_to_reference(points, reference_points):
    """Align `points` to `reference_points` with a robust global scale + translation.

    This mirrors the scale alignment used in your mv_recon pipeline (median center
    and median radius), but without ICP. Suitable for single-frame evaluation.

    Returns:
        aligned_points, scale
    """
    points = np.asarray(points, dtype=np.float32)
    reference_points = np.asarray(reference_points, dtype=np.float32)
    if points.shape[0] == 0 or reference_points.shape[0] == 0:
        return points, float('nan')

    c_ref = np.median(reference_points, axis=0)
    c_src = np.median(points, axis=0)
    r_ref = np.median(np.linalg.norm(reference_points - c_ref, axis=1)) + 1e-8
    r_src = np.median(np.linalg.norm(points - c_src, axis=1)) + 1e-8
    scale = float(r_ref / r_src)
    aligned = (points - c_src) * scale + c_ref
    return aligned.astype(np.float32), scale


def nn_distance_stats(src, dst):
    """Return mean/median NN distance from src -> dst."""
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    if src.shape[0] == 0 or dst.shape[0] == 0:
        return {"mean": float('inf'), "median": float('inf'), "n": int(src.shape[0])}
    d = _nn_distances(src, dst)
    return {"mean": float(np.mean(d)), "median": float(np.median(d)), "n": int(d.shape[0])}


def accuracy_completion(pred_points, gt_points):
    """Compute mv_recon-style accuracy/completion based on NN distances.

    - accuracy: mean NN distance pred -> gt
    - completion: mean NN distance gt -> pred
    """
    acc = nn_distance_stats(pred_points, gt_points)
    comp = nn_distance_stats(gt_points, pred_points)
    return {
        "acc": acc["mean"],
        "acc_median": acc["median"],
        "comp": comp["mean"],
        "comp_median": comp["median"],
        "n_pred": int(np.asarray(pred_points).shape[0]),
        "n_gt": int(np.asarray(gt_points).shape[0]),
    }


def chamfer_distance(src, dst):
    """Symmetric Chamfer distance (mean NN distance both directions)."""
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    if src.shape[0] == 0 or dst.shape[0] == 0:
        return float('inf')
    d1 = _nn_distances(src, dst).mean()
    d2 = _nn_distances(dst, src).mean()
    return float(0.5 * (d1 + d2))


def fscore(src, dst, tau):
    """F-score at distance threshold tau (meters) using NN distances."""
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    if src.shape[0] == 0 or dst.shape[0] == 0:
        return 0.0
    tau = float(tau)
    if tau <= 0:
        raise ValueError("tau must be > 0")

    d_src = _nn_distances(src, dst)
    d_dst = _nn_distances(dst, src)
    precision = float(np.mean(d_src <= tau))
    recall = float(np.mean(d_dst <= tau))
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def pointwise_l2_stats(pts_gt, pts_pred):
    """Mean/median L2 error between corresponding point clouds."""
    pts_gt = np.asarray(pts_gt, dtype=np.float32)
    pts_pred = np.asarray(pts_pred, dtype=np.float32)
    if pts_gt.shape != pts_pred.shape:
        raise ValueError(f"pts shape mismatch: {pts_gt.shape} vs {pts_pred.shape}")
    if pts_gt.shape[0] == 0:
        return {"mean": float('nan'), "median": float('nan'), "n": 0}
    err = np.linalg.norm(pts_gt - pts_pred, axis=1)
    return {"mean": float(err.mean()), "median": float(np.median(err)), "n": int(err.shape[0])}
