#!/usr/bin/env python
"""Run HaWoR's own inference on one clip and emit the viz_delivery hand21 npz.

Nothing in HaWoR's model path is modified. This driver calls the exact four stages that
``demo.py`` calls, in the same order and with the same arguments:

    detect_track_video  ->  hawor_motion_estimation  ->  hawor_slam  ->  hawor_infiller

and then replaces ONLY demo.py's visualisation tail (aitviewer) with the delivery-format
writer. No smoothing, gating, depth repair or outlier rejection is applied anywhere.

WORLD -> CAMERA
---------------
HaWoR's ``hawor_infiller`` returns MANO parameters in DROID-SLAM WORLD space. The delivery
format needs CAMERA space, so the joints are mapped back through the same SLAM extrinsics
demo.py uses for ``--vis_mode cam``:

    X_cam = R_w2c @ X_world + t_w2c

demo.py additionally left-multiplies both the cameras and the vertices by R_x = diag(1,-1,-1)
before calling ``run_vis2_on_video_cam``. That cancels exactly:

    (R_x R_c2w)^T (R_x v) - (R_x R_c2w)^T (R_x t_c2w) = R_c2w^T v - R_c2w^T t_c2w

so the raw (un-flipped) R_w2c/t_w2c from ``load_slam_cam`` give the identical camera frame.
--check-roundtrip verifies this numerically rather than trusting the algebra.
"""
import argparse, json, os, sys, time

import numpy as np
import torch

HAWOR = os.path.expanduser('~/projects/HaWoR')
sys.path.insert(0, HAWOR)
os.chdir(HAWOR)                      # HaWoR resolves _DATA/ and weights/ relative to cwd

# EgoForce's own intrinsics reader, so both models read the calibration identically.
sys.path.insert(0, os.path.expanduser('~/projects/EgoForce'))
from fusion.calibration import read_K                                   # noqa: E402

from scripts.scripts_test_video.detect_track_video import detect_track_video   # noqa: E402
from scripts.scripts_test_video.hawor_video import (hawor_motion_estimation,   # noqa: E402
                                                    hawor_infiller)
from scripts.scripts_test_video.hawor_slam import hawor_slam                   # noqa: E402
from lib.eval_utils.custom_utils import load_slam_cam                          # noqa: E402
from hawor.utils.process import run_mano, run_mano_left                        # noqa: E402
from hawor.utils.rotation import rotation_matrix_to_angle_axis                 # noqa: E402

HAND = {0: 'left', 1: 'right'}


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def joints_world(pred_trans, pred_rot, pred_hand_pose, pred_betas, idx):
    """MANO joints for one hand over the whole sequence -> (T, 21, 3) in HaWoR world space."""
    fn = run_mano if idx == 1 else run_mano_left
    out = fn(pred_trans[idx:idx + 1], pred_rot[idx:idx + 1],
             pred_hand_pose[idx:idx + 1], betas=pred_betas[idx:idx + 1])
    return out['joints'][0].detach().cpu().numpy()          # (T, 21, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--clip', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--check-roundtrip', action='store_true',
                    help='verify world->cam against the pre-SLAM cam_space params')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---------------------------------------------------------------- intrinsics (never invented)
    (fx, fy, cx, cy), block = read_K(a.calib, prefer='rectified', eye='left')
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    log(f'intrinsics [{block}/left] fx={fx:.4f} fy={fy:.4f} cx={cx:.4f} cy={cy:.4f}')

    # demo.py takes a single --img_focal and assumes the principal point is the image centre.
    # Record the deviation rather than leaving it implicit.
    assert abs(fx - fy) < 1e-6, f'fx != fy ({fx} vs {fy}); --img_focal cannot represent this'

    class A:                       # the arg object HaWoR's stage functions expect
        video_path = a.video
        img_focal = float(fx)
        input_type = 'file'
        checkpoint = './weights/hawor/checkpoints/hawor.ckpt'
        infiller_weight = './weights/hawor/checkpoints/infiller.pt'
        vis_mode = 'cam'
    args = A()

    # ---------------------------------------------------------------- HaWoR, unmodified
    log('stage 1/4  detect_track_video')
    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)
    T_frames = len(imgfiles)
    log(f'  {T_frames} frames extracted, seq_folder={seq_folder}')

    log('stage 2/4  hawor_motion_estimation')
    frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)
    log(f'  img_focal used by HaWoR = {img_focal}')

    slam_path = os.path.join(seq_folder, f'SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz')
    if not os.path.exists(slam_path):
        log('stage 3/4  hawor_slam (DROID-SLAM + Metric3D)')
        hawor_slam(args, start_idx, end_idx)
    else:
        log('stage 3/4  hawor_slam  (cached)')
    R_w2c, t_w2c, R_c2w, t_c2w = load_slam_cam(slam_path)

    log('stage 4/4  hawor_infiller')
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(
        args, start_idx, end_idx, frame_chunks_all)

    # Which frames the model actually SAW (pre-infill). hawor_infiller sets pred_valid=1 across
    # every window it fills, so the post-infill mask cannot distinguish observed from invented.
    observed = np.zeros((2, T_frames), dtype=bool)
    for idx in (0, 1):
        for ck in frame_chunks_all.get(idx, []):
            observed[idx, np.asarray(ck, dtype=int)] = True

    # Three tiers of provenance, not two. hawor_motion_estimation interpolates the bbox across
    # detector gaps (hawor_video.py:136-137) and runs the ViT on every frame of the span, so
    # `observed` means "the ViT saw this frame", which is wider than "the detector fired".
    # detector_fired is a diagnostic only; it does not change what is delivered.
    detector_fired = np.zeros((2, T_frames), dtype=bool)
    try:
        tracks = np.load(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_tracks.npy',
                         allow_pickle=True).item()
        for tid in tracks:                      # same left/right grouping as hawor_video.py:77-92
            trk = tracks[tid]
            v = np.array([t['det'] for t in trk])
            if v.sum() == 0:
                continue
            hr = np.concatenate([t['det_handedness'] for t in trk])[v]
            side = 0 if hr.sum() / len(hr) < 0.5 else 1
            for t in trk:
                if t['det']:
                    detector_fired[side, int(t['frame'])] = True
    except Exception as e:                      # diagnostic only - never fail the run for it
        log(f'  (detector_fired diagnostic unavailable: {type(e).__name__}: {e})')

    final_valid = np.asarray(pred_valid, dtype=bool)
    log(f'  observed: L={observed[0].sum()} R={observed[1].sum()}  |  '
        f'after infill: L={final_valid[0].sum()} R={final_valid[1].sum()}')

    # ---------------------------------------------------------------- world -> camera
    rows = []
    n_slam = len(R_w2c)
    Rw2c = R_w2c.numpy().astype(np.float64)
    tw2c = t_w2c.numpy().astype(np.float64)

    for idx in (0, 1):
        Jw = joints_world(pred_trans, pred_rot, pred_hand_pose, pred_betas, idx)  # (T,21,3) world
        for f in range(T_frames):
            if not final_valid[idx, f] or f >= n_slam:
                continue
            Xc = Jw[f] @ Rw2c[f].T + tw2c[f]                 # (21,3) camera frame
            rows.append(dict(frame=f, hand=idx, kp3d=Xc, measured=bool(observed[idx, f])))

    rows.sort(key=lambda r: (r['frame'], r['hand']))
    log(f'  {len(rows)} rows ({sum(r["measured"] for r in rows)} observed, '
        f'{sum(not r["measured"] for r in rows)} infilled)')

    kp3d_cam = np.stack([r['kp3d'] for r in rows]).astype(np.float64)
    frame_idx = np.array([r['frame'] for r in rows], dtype=np.int64)
    is_right = np.array([r['hand'] for r in rows], dtype=np.int8)
    measured = np.array([r['measured'] for r in rows], dtype=bool)

    Z = np.clip(kp3d_cam[..., 2], 1e-6, None)
    kp2d = np.stack([fx * kp3d_cam[..., 0] / Z + cx,
                     fy * kp3d_cam[..., 1] / Z + cy], axis=-1)

    # ---------------------------------------------------------------- round-trip verification
    rt = None
    if a.check_roundtrip:
        rt = verify_roundtrip(seq_folder, frame_chunks_all, Rw2c, tw2c, rows)

    # ---------------------------------------------------------------- write
    import cv2
    cap = cv2.VideoCapture(a.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)); Nvid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    npz_path = os.path.join(a.out, f'{a.clip}_hand21_keypoints.npz')
    np.savez(npz_path,
             kp3d_cam=kp3d_cam, kp2d=kp2d, frame_idx=frame_idx,
             is_right_wilor=is_right, is_right_mp=is_right,
             source=np.array(['wilor'] * len(rows), dtype='<U16'),
             depth_measured=measured,
             K=K, width=W, height=H, fps=fps, step=1)

    # sibling carrying the coverage denominator (every frame was processed, step=1)
    sib = os.path.join(a.out, f'{a.clip}_3d_keypoints.npz')
    np.savez(sib, processed_frames=np.arange(T_frames, dtype=np.int64),
             frame_idx=frame_idx, K=K, width=W, height=H, fps=fps, step=1)

    meta = dict(
        clip=a.clip, model='HaWoR', repo_commit=os.popen('git -C %s rev-parse HEAD' % HAWOR).read().strip(),
        frames=T_frames, video_frames=Nvid, width=W, height=H, fps=fps,
        intrinsics=dict(block=block, fx=fx, fy=fy, cx=cx, cy=cy,
                        img_focal_passed_to_hawor=float(fx),
                        hawor_assumed_principal_point=[W / 2, H / 2],
                        principal_point_offset_px=[cx - W / 2, cy - H / 2],
                        note='demo.py takes a single --img_focal and assumes the principal point '
                             'is the image centre. K written here is the clip\'s true rectified '
                             'left K; the offset above is how far HaWoR\'s internal camera sits '
                             'from it.'),
        rows=len(rows), observed_rows=int(measured.sum()), infilled_rows=int((~measured).sum()),
        observed_frames=dict(left=int(observed[0].sum()), right=int(observed[1].sum())),
        detector_fired_frames=dict(left=int(detector_fired[0].sum()),
                                   right=int(detector_fired[1].sum())),
        provenance_note='depth_measured=True means HaWoR\'s ViT ran on that image frame. '
                        'HaWoR interpolates the bbox across detector gaps, so that is wider '
                        'than detector_fired_frames. depth_measured=False rows are the '
                        'transformer infiller\'s output, with no image evidence.',
        final_valid_frames=dict(left=int(final_valid[0].sum()), right=int(final_valid[1].sum())),
        slam_frames=n_slam, roundtrip=rt,
        postprocessing='none - raw HaWoR output',
    )
    with open(os.path.join(a.out, f'{a.clip}_hawor_metadata.json'), 'w') as fh:
        json.dump(meta, fh, indent=2)

    log(f'wrote {npz_path}')
    log(f'      {sib}')
    print(json.dumps(meta, indent=2))


def verify_roundtrip(seq_folder, frame_chunks_all, Rw2c, tw2c, rows):
    """Compare world->cam joints against the pre-SLAM cam_space params for OBSERVED frames.

    hawor_motion_estimation writes its camera-space MANO params to cam_space/<idx>/<a>_<b>.json
    BEFORE any SLAM runs. Recomputing joints from those and comparing against the world joints
    mapped back through R_w2c/t_w2c proves the frame conversion is right - if the two disagree,
    the delivered 3D is in the wrong frame. Infilled frames are excluded: they have no cam_space
    entry by construction.
    """
    by_key = {(r['frame'], r['hand']): r['kp3d'] for r in rows if r['measured']}
    # A (hand, frame) can appear in MORE THAN ONE chunk: two tracker ids whose per-frame handedness
    # disagreed with their majority handedness both land in the same hand's merged track, so
    # parse_chunks emits overlapping chunks (seen on episode_002, right hand, frames 1073 & 1082).
    # hawor_infiller resolves that by last-write-wins, and this driver mirrors it, so the delivered
    # row corresponds to exactly one chunk. Compare against the chunk that WROTE it - i.e. take the
    # min per key - and report the overlaps separately instead of failing on a discarded value.
    per_key = {}
    for idx in (0, 1):
        for ck in frame_chunks_all.get(idx, []):
            p = os.path.join(seq_folder, 'cam_space', str(idx), f'{ck[0]}_{ck[-1]}.json')
            if not os.path.exists(p):
                continue
            with open(p) as fh:
                d = {k: torch.tensor(v) for k, v in json.load(fh).items()}
            rot = rotation_matrix_to_angle_axis(d['init_root_orient'])
            hp = rotation_matrix_to_angle_axis(d['init_hand_pose'])
            fn = run_mano if idx == 1 else run_mano_left
            Jc = fn(d['init_trans'], rot, hp, betas=d['init_betas'])['joints'][0]
            Jc = Jc.detach().cpu().numpy()                      # (t,21,3) camera frame, no SLAM
            for i, f in enumerate(np.asarray(ck, dtype=int)):
                got = by_key.get((int(f), idx))
                if got is not None:
                    per_key.setdefault((int(f), idx), []).append(
                        float(np.linalg.norm(got - Jc[i], axis=-1).max()))
    if not per_key:
        return dict(n=0, note='no observed frames to check')
    dup = {k: v for k, v in per_key.items() if len(v) > 1}
    errs = np.asarray([min(v) for v in per_key.values()])
    out = dict(n=int(errs.size), max_m=float(errs.max()), median_m=float(np.median(errs)),
               p99_m=float(np.percentile(errs, 99)),
               frames_in_multiple_chunks=int(len(dup)),
               multi_chunk_keys=[[int(f), int(h)] for (f, h) in sorted(dup)][:20])
    log(f'  ROUND-TRIP world->cam vs pre-SLAM cam_space: n={out["n"]} '
        f'median={out["median_m"]:.3e} m  max={out["max_m"]:.3e} m')
    return out


if __name__ == '__main__':
    main()
