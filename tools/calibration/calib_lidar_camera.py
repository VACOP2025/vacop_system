#!/usr/bin/env python3
# calib_lidar_camera.py
#
# Extrinsic calibration LiDAR -> Camera from:
#  - dataset_test/calibration.yml
#  - dataset_test/images/0001.jpg..0010.jpg
#  - dataset_test/pointclouds/0001.asc..0010.asc
#
# Target board:
#  - checkerboard 6x8 squares -> 5x7 inner corners
#  - square size = 10 cm
#  - white border 10 cm on all 4 sides (total panel = 0.80m x 1.00m)
#
# Outputs:
#  - extrinsics_lidar_to_camera.json / .yaml
#  - dataset_test/reprojection/XXXX_overlay_{pair,mean}.jpg

import os
import json
import yaml
import numpy as np
import cv2

# =========================
# USER SETTINGS
# =========================
DATASET_DIR = os.path.expanduser("~/PGE/dataset_test")

# 6x8 squares => 5x7 inner corners
INNER_COLS = 5
INNER_ROWS = 7

# Real geometry
SQUARE_SIZE_M = 0.10  # 10 cm
BORDER_M = 0.10       # 10 cm on each side

# Available IDs -> now 0001..0010
IDS = list(range(1, 11))  # 0001..0010

# RANSAC plane
PLANE_DIST_THRESH = 0.01
PLANE_ITERS = 4000

# Reprojection settings
REPROJ_MAX_POINTS = 40000
REPROJ_MIN_Z = 0.10
REPROJ_DOT_RADIUS = 1


# =========================
# Robust YAML loader
# =========================
def load_camera_yaml(path: str):
    with open(path, "r") as f:
        txt = f.read()

    lines = []
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith("%YAML:"):
            continue
        if s == "---":
            continue
        lines.append(line.replace("!!opencv-matrix", ""))
    cleaned = "\n".join(lines)

    y = yaml.safe_load(cleaned)
    if not isinstance(y, dict):
        raise RuntimeError("calibration.yml: contenu inattendu (pas un dict).")

    def read_matrix(node):
        if isinstance(node, dict) and "rows" in node and "cols" in node and "data" in node:
            rows = int(node["rows"])
            cols = int(node["cols"])
            return np.array(node["data"], dtype=np.float64).reshape(rows, cols)

        if isinstance(node, dict) and "data" in node:
            data = np.array(node["data"], dtype=np.float64)
            if data.size == 9:
                return data.reshape(3, 3)
            return data.reshape(-1, 1)

        raise RuntimeError(f"Impossible de lire une matrice depuis: {node}")

    if "camera_matrix" in y and "distortion_coefficients" in y:
        K = read_matrix(y["camera_matrix"])
        D = read_matrix(y["distortion_coefficients"]).reshape(-1, 1)
        return K, D

    K_candidates = ["K", "camera_matrix", "CameraMatrix", "intrinsics"]
    D_candidates = ["D", "dist", "dist_coeffs", "distortion_coefficients", "DistCoeffs"]

    K = None
    D = None

    for kk in K_candidates:
        if kk in y:
            K = read_matrix(y[kk])
            break

    for dk in D_candidates:
        if dk in y:
            D = read_matrix(y[dk])
            break

    if K is None or D is None:
        raise RuntimeError(f"Impossible de trouver K et D. Clés trouvées: {list(y.keys())}")

    return K, np.array(D, dtype=np.float64).reshape(-1, 1)


# =========================
# Chessboard detection + PnP
# =========================
def board_object_points_inner():
    obj = []
    for r in range(INNER_ROWS):
        for c in range(INNER_COLS):
            x = BORDER_M + (c + 1) * SQUARE_SIZE_M
            y = BORDER_M + (r + 1) * SQUARE_SIZE_M
            obj.append([x, y, 0.0])
    return np.array(obj, dtype=np.float64)

def detect_corners(image_path: str):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    pattern = (INNER_COLS, INNER_ROWS)
    ok, corners = cv2.findChessboardCornersSB(
        gray, pattern,
        flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    )
    if not ok or corners is None:
        return None
    return corners.reshape(-1, 2).astype(np.float64)

def solve_pnp_board_to_cam(corners_2d: np.ndarray, K: np.ndarray, D: np.ndarray):
    obj = board_object_points_inner()
    ok, rvec, tvec = cv2.solvePnP(obj, corners_2d, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R_cb, _ = cv2.Rodrigues(rvec)
    t_cb = tvec.reshape(3)

    T_cb = np.eye(4)
    T_cb[:3, :3] = R_cb
    T_cb[:3, 3] = t_cb
    return T_cb


# =========================
# LiDAR pose of board from plane rectangle
# =========================
def load_asc_xyz(path: str) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] >= 4:
        return data[:, 1:4]
    if data.shape[1] == 3:
        return data[:, 0:3]
    raise RuntimeError(f"ASC columns unexpected: {data.shape}")

def fit_plane_ransac(points: np.ndarray, dist_thresh=0.01, n_iter=3000):
    if points.shape[0] < 30:
        return None, None, None

    rng = np.random.default_rng(0)
    best_inliers = None
    best_n = None
    best_d = None
    N = points.shape[0]

    for _ in range(n_iter):
        idx = rng.choice(N, size=3, replace=False)
        p1, p2, p3 = points[idx]
        n = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -np.dot(n, p1)

        dist = np.abs(points @ n + d)
        inliers = dist < dist_thresh

        if best_inliers is None or np.count_nonzero(inliers) > np.count_nonzero(best_inliers):
            best_inliers = inliers
            best_n = n
            best_d = d

    return best_n, best_d, best_inliers

def make_plane_basis(n: np.ndarray):
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, n)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a); u /= np.linalg.norm(u)
    v = np.cross(n, u); v /= np.linalg.norm(v)
    return u, v

def project_to_plane_2d(points: np.ndarray, origin: np.ndarray, u: np.ndarray, v: np.ndarray):
    rel = points - origin[None, :]
    return np.stack([rel @ u, rel @ v], axis=1)

def umeyama_alignment(src: np.ndarray, dst: np.ndarray):
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    X = src - mu_s
    Y = dst - mu_d
    S = (X.T @ Y) / src.shape[0]
    U, _, Vt = np.linalg.svd(S)
    Rm = Vt.T @ U.T
    if np.linalg.det(Rm) < 0:
        Vt[-1, :] *= -1
        Rm = Vt.T @ U.T
    t = mu_d - Rm @ mu_s
    return Rm, t

def estimate_board_pose_from_lidar(points: np.ndarray):
    n, d, inliers = fit_plane_ransac(points, dist_thresh=PLANE_DIST_THRESH, n_iter=PLANE_ITERS)
    if inliers is None or np.count_nonzero(inliers) < 30:
        return None

    pts_in = points[inliers]
    origin = pts_in.mean(axis=0)
    u, v = make_plane_basis(n)

    pts2d = project_to_plane_2d(pts_in, origin, u, v).astype(np.float32)
    rect = cv2.minAreaRect(pts2d)
    box2d = cv2.boxPoints(rect).astype(np.float64)

    corners_l = origin[None, :] + box2d[:, 0:1] * u[None, :] + box2d[:, 1:2] * v[None, :]
    corners_l = corners_l.reshape(4, 3)

    W = 6 * SQUARE_SIZE_M + 2 * BORDER_M  # 0.80
    H = 8 * SQUARE_SIZE_M + 2 * BORDER_M  # 1.00
    corners_b = np.array([
        [0.0, 0.0, 0.0],
        [W,   0.0, 0.0],
        [W,   H,   0.0],
        [0.0, H,   0.0],
    ], dtype=np.float64)

    base = [0, 1, 2, 3]
    orders = []
    for shift in range(4):
        cyc = base[shift:] + base[:shift]
        orders.append(cyc)
        orders.append(list(reversed(cyc)))

    best = None
    best_err = 1e18
    for order in orders:
        dst = corners_l[order]
        Rm, t = umeyama_alignment(corners_b, dst)
        pred = (Rm @ corners_b.T).T + t[None, :]
        err = np.mean(np.linalg.norm(pred - dst, axis=1))
        if err < best_err:
            best_err = err
            best = (Rm, t)

    if best is None:
        return None

    R_lb, t_lb = best
    T_lb = np.eye(4)
    T_lb[:3, :3] = R_lb
    T_lb[:3, 3] = t_lb
    return T_lb


# =========================
# Transform utils + averaging
# =========================
def invert_T(T):
    Rm = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = Rm.T
    Ti[:3, 3] = -Rm.T @ t
    return Ti

def rotmat_to_quat(Rm):
    tr = np.trace(Rm)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (Rm[2, 1] - Rm[1, 2]) / S
        y = (Rm[0, 2] - Rm[2, 0]) / S
        z = (Rm[1, 0] - Rm[0, 1]) / S
    else:
        if (Rm[0, 0] > Rm[1, 1]) and (Rm[0, 0] > Rm[2, 2]):
            S = np.sqrt(1.0 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2]) * 2
            w = (Rm[2, 1] - Rm[1, 2]) / S
            x = 0.25 * S
            y = (Rm[0, 1] + Rm[1, 0]) / S
            z = (Rm[0, 2] + Rm[2, 0]) / S
        elif Rm[1, 1] > Rm[2, 2]:
            S = np.sqrt(1.0 + Rm[1, 1] - Rm[0, 0] - Rm[2, 2]) * 2
            w = (Rm[0, 2] - Rm[2, 0]) / S
            x = (Rm[0, 1] + Rm[1, 0]) / S
            y = 0.25 * S
            z = (Rm[1, 2] + Rm[2, 1]) / S
        else:
            S = np.sqrt(1.0 + Rm[2, 2] - Rm[0, 0] - Rm[1, 1]) * 2
            w = (Rm[1, 0] - Rm[0, 1]) / S
            x = (Rm[0, 2] + Rm[2, 0]) / S
            y = (Rm[1, 2] + Rm[2, 1]) / S
            z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)

def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ], dtype=np.float64)

def average_transforms(T_list):
    ts = np.stack([T[:3, 3] for T in T_list], axis=0)
    t_mean = ts.mean(axis=0)

    qs = np.stack([rotmat_to_quat(T[:3, :3]) for T in T_list], axis=0)
    q0 = qs[0]
    for i in range(len(qs)):
        if np.dot(qs[i], q0) < 0:
            qs[i] = -qs[i]
    q_mean = qs.mean(axis=0)
    q_mean /= np.linalg.norm(q_mean)

    R_mean = quat_to_rotmat(q_mean)

    Tm = np.eye(4)
    Tm[:3, :3] = R_mean
    Tm[:3, 3] = t_mean
    return Tm


# =========================
# Reprojection
# =========================
def save_reprojection_overlay(img_path: str, pts_lidar: np.ndarray, T_lidar_to_cam: np.ndarray,
                              K: np.ndarray, D: np.ndarray, out_path: str):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {img_path}")
    h, w = img.shape[:2]

    # Subsampling
    if pts_lidar.shape[0] > REPROJ_MAX_POINTS:
        rng = np.random.default_rng(0)
        idx = rng.choice(pts_lidar.shape[0], size=REPROJ_MAX_POINTS, replace=False)
        pts = pts_lidar[idx]
    else:
        pts = pts_lidar

    R = T_lidar_to_cam[:3, :3]
    t = T_lidar_to_cam[:3, 3]

    # Points in camera frame (to filter by z)
    pts_cam = (R @ pts.T).T + t[None, :]
    mask = pts_cam[:, 2] > REPROJ_MIN_Z
    pts = pts[mask]
    pts_cam = pts_cam[mask]
    if pts.shape[0] == 0:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, img)
        return

    # projectPoints expects rvec/tvec
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)
    proj, _ = cv2.projectPoints(pts.astype(np.float64), rvec, tvec, K, D)
    proj = proj.reshape(-1, 2)

    for (u, v) in proj:
        ui = int(round(u))
        vi = int(round(v))
        if 0 <= ui < w and 0 <= vi < h:
            cv2.circle(img, (ui, vi), REPROJ_DOT_RADIUS, (0, 255, 0), -1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)


# =========================
# Main
# =========================
def main():
    cam_yaml = os.path.join(DATASET_DIR, "calibration.yml")
    images_dir = os.path.join(DATASET_DIR, "images")
    clouds_dir = os.path.join(DATASET_DIR, "pointclouds")
    reproj_dir = os.path.join(DATASET_DIR, "reprojection")

    K, D = load_camera_yaml(cam_yaml)
    print("Loaded intrinsics:", cam_yaml)
    print("K=\n", K)
    print("D=", D.reshape(-1))

    T_list = []
    per_pair_T = {}
    used = 0

    for idx in IDS:
        img_path = os.path.join(images_dir, f"{idx:04d}.jpg")
        cloud_path = os.path.join(clouds_dir, f"{idx:04d}.asc")

        print(f"\n[{idx:04d}]")
        if not os.path.exists(img_path):
            print("  - Missing image:", img_path)
            continue
        if not os.path.exists(cloud_path):
            print("  - Missing cloud:", cloud_path)
            continue

        corners2d = detect_corners(img_path)
        if corners2d is None:
            print("  - Chessboard NOT detected.")
            continue

        T_cb = solve_pnp_board_to_cam(corners2d, K, D)
        if T_cb is None:
            print("  - solvePnP failed.")
            continue

        pts = load_asc_xyz(cloud_path)
        T_lb = estimate_board_pose_from_lidar(pts)
        if T_lb is None:
            print("  - LiDAR board pose failed.")
            continue

        T_lidar_to_cam = T_cb @ invert_T(T_lb)
        T_list.append(T_lidar_to_cam)
        per_pair_T[idx] = T_lidar_to_cam
        used += 1
        print("  - OK (pair used).")

    if used < 0:
        raise RuntimeError("Not enough valid pairs (need >=2).")

    T_mean = average_transforms(T_list)

    out = {
        "dataset_dir": DATASET_DIR,
        "board": {
            "chessboard_squares": [6, 8],
            "inner_corners": [INNER_COLS, INNER_ROWS],
            "square_size_m": float(SQUARE_SIZE_M),
            "border_m": float(BORDER_M),
            "panel_size_m": [
                float(6 * SQUARE_SIZE_M + 2 * BORDER_M),
                float(8 * SQUARE_SIZE_M + 2 * BORDER_M),
            ],
        },
        "used_pairs": int(used),
        "T_lidar_to_camera": T_mean.tolist(),
        "R_lidar_to_camera": T_mean[:3, :3].tolist(),
        "t_lidar_to_camera": T_mean[:3, 3].tolist(),
    }

    out_json = os.path.join(DATASET_DIR, "extrinsics_lidar_to_camera.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    out_yaml = os.path.join(DATASET_DIR, "extrinsics_lidar_to_camera.yaml")
    with open(out_yaml, "w") as f:
        yaml.safe_dump({"T_lidar_to_camera": T_mean.tolist()}, f, sort_keys=False)

    print("\nDONE ✅")
    print("Used pairs:", used)
    print("Saved:", out_json)
    print("Saved:", out_yaml)
    print("T_lidar_to_camera (mean)=\n", T_mean)
    t = T_mean[:3, 3]
    print("t (m) =", t)
    print("||t|| (m) =", float(np.linalg.norm(t)))

    # Reprojection overlays
    os.makedirs(reproj_dir, exist_ok=True)
    print("\nGenerating reprojection overlays in:", reproj_dir)

    for idx in sorted(per_pair_T.keys()):
        img_path = os.path.join(images_dir, f"{idx:04d}.jpg")
        cloud_path = os.path.join(clouds_dir, f"{idx:04d}.asc")
        pts = load_asc_xyz(cloud_path)

        out_pair = os.path.join(reproj_dir, f"{idx:04d}_overlay_pair.jpg")
        save_reprojection_overlay(img_path, pts, per_pair_T[idx], K, D, out_pair)

        out_mean = os.path.join(reproj_dir, f"{idx:04d}_overlay_mean.jpg")
        save_reprojection_overlay(img_path, pts, T_mean, K, D, out_mean)

    print("Reprojection done ✅")
    print("Exemple :", os.path.join(reproj_dir, "0001_overlay_mean.jpg"))

    # --- EXTRACT A SPECIFIC PAIR ---
    BEST_ID = 4  

    if BEST_ID in per_pair_T:
        T_best = per_pair_T[BEST_ID]
        print("\n=== BEST PAIR EXTRINSICS ===")
        print(f"Pair {BEST_ID:04d}")
        print("T_lidar_to_camera=\n", T_best)
        print("t (m) =", T_best[:3, 3])
        print("||t|| (m) =", float(np.linalg.norm(T_best[:3, 3])))

        out_best_json = os.path.join(DATASET_DIR, f"extrinsics_lidar_to_camera_pair_{BEST_ID:04d}.json")
        with open(out_best_json, "w") as f:
            json.dump({
                "best_pair": int(BEST_ID),
                "T_lidar_to_camera": T_best.tolist(),
                "R_lidar_to_camera": T_best[:3, :3].tolist(),
                "t_lidar_to_camera": T_best[:3, 3].tolist(),
            }, f, indent=2)

        out_best_yaml = os.path.join(DATASET_DIR, f"extrinsics_lidar_to_camera_pair_{BEST_ID:04d}.yaml")
        with open(out_best_yaml, "w") as f:
            yaml.safe_dump({"T_lidar_to_camera": T_best.tolist()}, f, sort_keys=False)

        print("Saved:", out_best_json)
        print("Saved:", out_best_yaml)
    else:
        print(f"\nPair {BEST_ID:04d} not available (maybe chessboard/plane failed).")


if __name__ == "__main__":
    main()