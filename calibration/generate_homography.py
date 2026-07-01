import argparse
import os
import numpy as np
import cv2


# ZED FHD1200 intrinsics
fx, fy = 737.085, 736.99
cx, cy = 971.613, 582.358

K = np.array([
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

# ZED lens distortion (factory)
# Only used if we click points on RAW images
dist = np.array([
    -0.0134761,
    -0.0298904,
     0.000704037,
     7.03782e-05,
     0.00780624
], dtype=np.float64)

# User settings
USE_RECTIFIED = False                 # True if ZED SDK rectified image

# RANSAC threshold is in *world units* because we solve img->world.
# If  world is meters, 0.10 means 10 cm tolerance.
RANSAC_THRESH_METERS = 0.25

# --- Optional verification: run YOLO and annotate distances on the same image ---
VERIFY_WITH_YOLO = True
YOLO_WEIGHTS_PATH = "yolov11n_tuned_1024.pt"   # default Ultralytics weights in repo root
YOLO_CONF = 0.25
YOLO_IMGSZ = 1280

VERIFY_OUT_PATH = "homography_verify.png"

# Click points helper
_clicked = []

def _mouse_cb(event, x, y, flags, userdata):
    if event == cv2.EVENT_LBUTTONDOWN:
        _clicked.append((float(x), float(y)))
        print(f"Clicked pixel: ({x}, {y})")


def collect_image_points(img_bgr):
    global _clicked
    _clicked = []

    disp = img_bgr.copy()
    cv2.namedWindow("Click cone FOOTPOINTS (bottom center)", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Click cone FOOTPOINTS (bottom center)", _mouse_cb)

    print("\nINSTRUCTIONS:")
    print("  - Left-click each cone FOOTPOINT (bottom center).")
    print("  - Press 'u' to undo last point.")
    print("  - Press 'q' when done.\n")

    while True:
        disp = img_bgr.copy()
        for i, (px, py) in enumerate(_clicked):
            cv2.circle(disp, (int(px), int(py)), 6, (0, 255, 255), -1)
            cv2.putText(disp, str(i), (int(px) + 8, int(py) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Click cone FOOTPOINTS (bottom center)", disp)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('u') and len(_clicked) > 0:
            removed = _clicked.pop()
            print(f"Undo: removed {removed}")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    return np.array(_clicked, dtype=np.float64)


# 1- Provide world points in the SAME ORDER as clicks
def get_world_points_for_clicks(n_points):
    """Enter world points (X forward, Y right) in meters.

    You can hardcode your cone coordinates here for repeatability.
    The only hard requirement is: length == n_points and ordering
    matches the click order.
    """

    world = None  # <-- set to np.array([...], dtype=np.float64)

    if world is None:
        print("\nWORLD POINT ENTRY")
        print("Enter X (forward) and Y (right) in meters for each clicked point.")
        vals = []
        for i in range(n_points):
            while True:
                s = input(f"  Point {i} world (X Y): ").strip()
                try:
                    Xs, Ys = s.split()
                    X, Y = float(Xs), float(Ys)
                    vals.append([X, Y])
                    break
                except Exception:
                    print("    Invalid. Format: X Y   (example: 10.0 -3.0)")
        world = np.array(vals, dtype=np.float64)

    if world.shape[0] != n_points:
        raise ValueError(f"Need {n_points} world points, got {world.shape[0]}")

    return world

def apply_homography(H, u, v):
    """Project an image pixel (u,v) to ground (X forward, Y right)."""
    p = np.array([float(u), float(v), 1.0], dtype=np.float64)
    q = H @ p
    if abs(q[2]) < 1e-9:
        return np.nan, np.nan
    return float(q[0] / q[2]), float(q[1] / q[2])


def print_per_point_errors(H_i2w, img_pts_u, world_pts, inliers):
    errs = []
    print("\nPer-point reprojection error (meters):")
    for i, (uv, XY) in enumerate(zip(img_pts_u, world_pts)):
        Xh, Yh = apply_homography(H_i2w, uv[0], uv[1])
        e = float(np.hypot(Xh - XY[0], Yh - XY[1]))
        errs.append(e)
        inl = int(inliers[i, 0]) if inliers is not None else -1
        print(f"  {i:02d}  inlier={inl}  err={e:.3f} m   img=({uv[0]:.1f},{uv[1]:.1f})  world=({XY[0]:.2f},{XY[1]:.2f})")

    print(f"Mean error: {np.mean(errs):.3f} m")
    print(f"Max  error: {np.max(errs):.3f} m")


def draw_box_distances(img_bgr, H_i2w, boxes_xyxy, labels=None):
    """Annotate boxes with (X,Y) and distance from origin using bottom-center pixel."""
    out = img_bgr.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
        u = 0.5 * (x1 + x2)
        v = y2  # bottom pixel (footpoint proxy)
        X, Y = apply_homography(H_i2w, u, v)
        d = float(np.hypot(X, Y))

        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

        tag = labels[i] if labels is not None and i < len(labels) else f"cone{i}"
        txt = f"{tag}  X={X:.2f}m Y={Y:.2f}m d={d:.2f}m"
        cv2.putText(out, txt, (int(x1), max(0, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return out


def verify_with_yolo(img_bgr, H_i2w):
    """Run YOLO (ultralytics) if available and weights exist; save annotated image."""
    if not VERIFY_WITH_YOLO:
        return

    try:
        from ultralytics import YOLO
    except Exception as e:
        print("\n[VERIFY] ultralytics not available; skipping YOLO verification.")
        print(f"         ({e})")
        return

    try:
        model = YOLO(YOLO_WEIGHTS_PATH)
    except Exception as e:
        print("\n[VERIFY] Could not load YOLO weights; skipping YOLO verification.")
        print(f"         weights='{YOLO_WEIGHTS_PATH}'  error={e}")
        return

    print("\n[VERIFY] Running YOLO on IMAGE_PATH...")
    res = model.predict(source=img_bgr, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, verbose=False)
    if not res:
        print("[VERIFY] No results returned.")
        return

    r0 = res[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        print("[VERIFY] No detections.")
        return

    boxes = r0.boxes.xyxy.cpu().numpy()

    labels = None
    try:
        cls = r0.boxes.cls.cpu().numpy().astype(int)
        if hasattr(model, "names") and isinstance(model.names, dict):
            labels = [model.names.get(int(c), str(int(c))) for c in cls]
        elif hasattr(model, "names") and isinstance(model.names, list):
            labels = [model.names[int(c)] if int(c) < len(model.names) else str(int(c)) for c in cls]
    except Exception:
        pass

    annotated = draw_box_distances(img_bgr, H_i2w, boxes, labels=labels)
    cv2.imwrite(VERIFY_OUT_PATH, annotated)
    print(f"[VERIFY] Saved annotated image: {VERIFY_OUT_PATH}")

# Main


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Homography matrix from cone footpoint correspondences."
    )
    parser.add_argument(
        "image",
        help="Path to the ZED camera image to use for calibration."
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save the .npy matrix (default: homography_matrix.npy in same dir as image)."
    )
    args = parser.parse_args()

    image_path = args.image
    output_dir = os.path.dirname(os.path.abspath(image_path))
    save_npy_path = args.output if args.output else os.path.join(output_dir, "homography_matrix.npy")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(
            f"Could not read image at '{image_path}'."
        )

    # Click pixels on the image we will actually use in our pipeline.
    # If USE_RECTIFIED=False, this is typically a RAW image.
    img_pts = collect_image_points(img)

    if img_pts.shape[0] < 4:
        raise ValueError("Need at least 4 point correspondences.")

    world_pts = get_world_points_for_clicks(img_pts.shape[0])

    # If points came from RAW image, undistort them into the pinhole/rectified pixel space.
    if not USE_RECTIFIED:
        img_pts_u = cv2.undistortPoints(img_pts.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
    else:
        img_pts_u = img_pts

    # Estimate homography (UNDISTORTED/RECTIFIED image pixels -> ground meters)
    H_i2w, inliers = cv2.findHomography(
        img_pts_u,
        world_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESH_METERS
    )

    if H_i2w is None:
        raise RuntimeError("cv2.findHomography failed. Check point ordering and diversity.")

    np.save(save_npy_path, H_i2w)

    np.set_printoptions(precision=8, suppress=True)
    print("\nH_i2w (UNDISTORTED/RECTIFIED image -> ground [X forward, Y right] meters):")
    print(H_i2w)

    if inliers is not None:
        print(f"Inliers: {int(inliers.sum())} / {len(inliers)}")

    print_per_point_errors(H_i2w, img_pts_u, world_pts, inliers)
    verify_with_yolo(img, H_i2w)
    # Debug overlay
    dbg = img.copy()
    for i, (p, w) in enumerate(zip(img_pts, world_pts)):
        cv2.circle(dbg, (int(p[0]), int(p[1])), 6, (0, 255, 255), -1)
        cv2.putText(dbg, f"{i}:{w[0]:.2f},{w[1]:.2f}", (int(p[0]) + 8, int(p[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imwrite(os.path.join(output_dir, "homography_debug.png"), dbg)
    print("Saved:")
    print(f"  - {save_npy_path}")
    print(f"  - {os.path.join(output_dir, 'homography_debug.png')}")

    # Automatically update the YAML config file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, "..", "config", "homography.yaml")
    yaml_path = os.path.normpath(yaml_path)

    values = H_i2w.flatten().tolist()
    values_str = ", ".join(repr(v) for v in values)

    yaml_content = (
        "zed_center:\n"
        "  ros__parameters:\n"
        f"    homography_matrix: [{values_str}]\n"
    )

    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\n[AUTO-UPDATE] Updated YAML config: {yaml_path}")
    print(f"  homography_matrix: [{values_str}]")


if __name__ == "__main__":
    main()