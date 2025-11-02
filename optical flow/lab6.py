# optical_flow_demo.py
# Based on: LearnOpenCV — "Optical Flow in OpenCV (C++/Python)"
# https://learnopencv.com/optical-flow-in-opencv/
# Requires: pip install numpy opencv-python opencv-contrib-python

import argparse
import os
import cv2
import numpy as np

# ------------------------------
# Sparse optical flow (Lucas–Kanade) demo
# ------------------------------
def lucas_kanade_method(video_path, save=False, save_dir="optical_flow_frames"):
    cap = cv2.VideoCapture(video_path)

    # Shi–Tomasi corner detection params
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7,
    )

    # Lucas–Kanade optical flow params
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    # Random colors for tracks
    color = np.random.randint(0, 255, (100, 3))

    # First frame + features
    ret, old_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame from video.")
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Drawing mask
    mask = np.zeros_like(old_frame)

    if save:
        os.makedirs(save_dir, exist_ok=True)
        counter = 0

    while True:
        # Read new frame
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

        # Display
        img = cv2.add(frame, mask)
        cv2.imshow("frame", img)

        if save:
            cv2.imwrite(os.path.join(save_dir, f"frame_{counter:06d}.jpg"), img)
            counter += 1

        k = cv2.waitKey(25) & 0xFF
        if k == 27:  # ESC
            break

        # Update state
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cap.release()
    cv2.destroyAllWindows()


# ------------------------------
# Dense optical flow wrapper (shared by multiple methods)
# ------------------------------
def dense_optical_flow(method, video_path, params=None, to_gray=False, save=False, save_dir="optical_flow_frames"):
    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame from video.")

    # HSV canvas (S=255 constant)
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    # Method-specific preprocessing
    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    if save:
        os.makedirs(save_dir, exist_ok=True)
        counter = 0

    params = [] if params is None else params

    while True:
        # Next frame
        ret, new_frame = cap.read()
        frame_copy = None if not ret else new_frame.copy()
        if not ret:
            break

        if to_gray:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        # Compute flow. Signature differs per method, but this works for:
        # - cv2.optflow.calcOpticalFlowSparseToDense(prev, next, None, *params)
        # - cv2.calcOpticalFlowFarneback(prev, next, None, *params)
        # - cv2.optflow.calcOpticalFlowDenseRLOF(prev, next, None, *params)
        flow = method(old_frame, new_frame, None, *params)

        # Visualize in HSV: Hue = angle, Value = normalized magnitude
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Show
        if frame_copy is not None:
            cv2.imshow("frame", frame_copy)
        cv2.imshow("optical flow", bgr)

        if save:
            cv2.imwrite(os.path.join(save_dir, f"flow_{counter:06d}.jpg"), bgr)
            counter += 1

        k = cv2.waitKey(25) & 0xFF
        if k == 27:  # ESC
            break

        # Update previous
        old_frame = new_frame

    cap.release()
    cv2.destroyAllWindows()


# ------------------------------
# CLI (matches the LearnOpenCV examples)
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Optical Flow demos (LearnOpenCV)")
    parser.add_argument("--algorithm", type=str, required=True,
                        choices=["lucaskanade", "lucaskanade_dense", "farneback", "rlof"])
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to an input video (e.g., videos/people.mp4)")
    parser.add_argument("--save", action="store_true", help="Save output frames to disk")
    parser.add_argument("--save_dir", type=str, default="optical_flow_frames", help="Directory to save frames")
    args = parser.parse_args()

    alg = args.algorithm.lower()
    vp = args.video_path

    if alg == "lucaskanade":
        lucas_kanade_method(vp, save=args.save, save_dir=args.save_dir)

    elif alg == "lucaskanade_dense":
        # Dense Lucas–Kanade via SparseToDense (needs grayscale)
        method = cv2.optflow.calcOpticalFlowSparseToDense
        dense_optical_flow(method, vp, params=[], to_gray=True, save=args.save, save_dir=args.save_dir)

    elif alg == "farneback":
        # Default Farneback params from the article:
        # pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        method = cv2.calcOpticalFlowFarneback
        params = [0.5, 3, 15, 3, 5, 1.2, 0]
        dense_optical_flow(method, vp, params=params, to_gray=True, save=args.save, save_dir=args.save_dir)

    elif alg == "rlof":
        # Dense RLOF expects 3-channel input; no grayscale conversion
        method = cv2.optflow.calcOpticalFlowDenseRLOF
        dense_optical_flow(method, vp, params=[], to_gray=False, save=args.save, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
