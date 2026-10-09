"""Write a SLAM-compatible camera trajectory without running DROID-SLAM.

DROID-SLAM's ``droid_backends`` extension has only ``.cu`` sources, so the world-space stage cannot
run on CPU. But HaWoR only consumes SLAM through one small contract, in
``lib/eval_utils/custom_utils.load_slam_cam``:

    pred_cam['traj']   (N, 7) float   [tx, ty, tz, qx, qy, qz, qw]   camera-to-world
    pred_cam['scale']  scalar         multiplied into the translation

So anything that can produce a camera-to-world trajectory can stand in. Two modes:

``identity``
    Every camera at the origin. HaWoR's "world" frame then equals its camera frame, so the output
    is camera-space hand pose. Use when you only want per-frame hands and will place them yourself.

``headpose``
    Read an existing 6DoF trajectory, e.g. ``head_pose_6dof.npz`` with ``T`` of shape (N, 4, 4)
    holding T_world_cam. This is the useful mode when head pose was already solved by another
    system: HaWoR then produces genuine world-space hands on CPU, because the GPU-only part was
    never HaWoR's to do.

Usage:
    python scripts/scripts_test_video/make_camera_npz.py \\
        --out <seq_folder>/SLAM/hawor_slam_w_scale_0_1823.npz --frames 1823 --mode identity

    python scripts/scripts_test_video/make_camera_npz.py \\
        --out <seq_folder>/SLAM/hawor_slam_w_scale_0_1823.npz \\
        --mode headpose --headpose head_pose_6dof.npz --frames 1823
"""

import argparse
import os

import numpy as np


def matrix_to_quat_xyzw(R):
    """(N,3,3) rotation matrices -> (N,4) quaternions as [x, y, z, w].

    Shepperd's method: pick the branch with the largest denominator so the square root is never
    taken of a near-zero number, which is where the naive trace formula loses precision.
    """
    R = np.asarray(R, dtype=np.float64)
    n = R.shape[0]
    q = np.zeros((n, 4), dtype=np.float64)          # x, y, z, w
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    tr = m00 + m11 + m22

    a = tr > 0
    if a.any():
        s = np.sqrt(tr[a] + 1.0) * 2
        q[a, 3] = 0.25 * s
        q[a, 0] = (R[a, 2, 1] - R[a, 1, 2]) / s
        q[a, 1] = (R[a, 0, 2] - R[a, 2, 0]) / s
        q[a, 2] = (R[a, 1, 0] - R[a, 0, 1]) / s

    b = (~a) & (m00 >= m11) & (m00 >= m22)
    if b.any():
        s = np.sqrt(1.0 + m00[b] - m11[b] - m22[b]) * 2
        q[b, 3] = (R[b, 2, 1] - R[b, 1, 2]) / s
        q[b, 0] = 0.25 * s
        q[b, 1] = (R[b, 0, 1] + R[b, 1, 0]) / s
        q[b, 2] = (R[b, 0, 2] + R[b, 2, 0]) / s

    c = (~a) & (~b) & (m11 >= m22)
    if c.any():
        s = np.sqrt(1.0 + m11[c] - m00[c] - m22[c]) * 2
        q[c, 3] = (R[c, 0, 2] - R[c, 2, 0]) / s
        q[c, 0] = (R[c, 0, 1] + R[c, 1, 0]) / s
        q[c, 1] = 0.25 * s
        q[c, 2] = (R[c, 1, 2] + R[c, 2, 1]) / s

    d = (~a) & (~b) & (~c)
    if d.any():
        s = np.sqrt(1.0 + m22[d] - m00[d] - m11[d]) * 2
        q[d, 3] = (R[d, 1, 0] - R[d, 0, 1]) / s
        q[d, 0] = (R[d, 0, 2] + R[d, 2, 0]) / s
        q[d, 1] = (R[d, 1, 2] + R[d, 2, 1]) / s
        q[d, 2] = 0.25 * s

    return q / np.linalg.norm(q, axis=1, keepdims=True)


def load_headpose(path, frames):
    """Return (N,3) t_c2w and (N,3,3) R_c2w from a 6DoF head-pose npz."""
    d = dict(np.load(path, allow_pickle=True))
    if 'T' not in d:
        raise SystemExit(f"{path} has no 'T'. Keys present: {sorted(d)}")
    T = np.asarray(d['T'], dtype=np.float64)
    if T.ndim != 3 or T.shape[1:] != (4, 4):
        raise SystemExit(f"'T' must be (N,4,4) camera-to-world, got {T.shape}")

    if len(T) < frames:
        raise SystemExit(f"head pose has {len(T)} frames, the clip needs {frames}. "
                         f"Check that both were run on the same video at the same fps.")
    if len(T) > frames:
        print(f"  note: head pose has {len(T)} frames, truncating to the clip's {frames}")
        T = T[:frames]
    return T[:, :3, 3], T[:, :3, :3]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, help='path of the npz to write')
    ap.add_argument('--frames', type=int, required=True, help='number of frames in the clip')
    ap.add_argument('--mode', default='identity', choices=['identity', 'headpose'])
    ap.add_argument('--headpose', default=None, help="6DoF npz with 'T' (N,4,4) camera-to-world")
    ap.add_argument('--scale', type=float, default=1.0,
                    help='multiplied into the translation by load_slam_cam')
    args = ap.parse_args()

    n = args.frames
    if args.mode == 'identity':
        t = np.zeros((n, 3), dtype=np.float64)
        q = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))     # x,y,z,w identity
        print(f"identity trajectory, {n} frames -- output will be CAMERA-space, not world")
    else:
        if not args.headpose:
            raise SystemExit('--mode headpose requires --headpose <npz>')
        t, R = load_headpose(args.headpose, n)
        q = matrix_to_quat_xyzw(R)
        print(f"trajectory from {args.headpose}: {n} frames, "
              f"translation span {np.ptp(t, axis=0).round(3).tolist()} m")

    traj = np.concatenate([t, q], axis=1).astype(np.float32)    # (N,7) tx ty tz qx qy qz qw
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, traj=traj, scale=np.float32(args.scale))
    print(f"wrote {args.out}  traj{traj.shape}  scale={args.scale}")


if __name__ == '__main__':
    main()
