#!/usr/bin/env python
"""Principal-point correction for HaWoR camera-frame keypoints (in place on a delivery npz).

HaWoR places every hand with a camera whose principal point is the IMAGE CENTRE
(scripts/scripts_test_video/hawor_video.py: img_center = [w/2, h/2]); only the focal length is
taken from the calibration. run_hawor_clip.py then projects kp3d_cam with the clip's TRUE K, so
on a camera whose principal point is off-centre every delivered 2D point is shifted by
(cx - W/2, cy - H/2) px from where HaWoR saw the hand (akai: up to ~(+47, -25) px; ZED and
bitrobot: < 2 px).

Correction: X' = X - Z (cx - W/2)/fx,  Y' = Y - Z (cy - H/2)/fy,  Z' = Z, applied to kp3d_cam and
kp3d_cam_raw; kp2d / kp2d_raw are re-projected with the true K, which makes them equal to HaWoR's
own (image-centre) projection. Stored: pp_offset_px = (cx - W/2, cy - H/2) and pp_corrected=True;
undo with X = X' + Z * pp_offset_px[0] / fx (likewise Y).
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True, help='delivery npz, rewritten in place')
    a = ap.parse_args()
    z = dict(np.load(a.npz, allow_pickle=False))
    if bool(z.get('pp_corrected', False)):
        print(f'pp_correct: already corrected (offset {z["pp_offset_px"]} px) - unchanged'); return
    K = z['K'].astype(np.float64); W, H = float(z['width']), float(z['height'])
    off = np.array([K[0, 2] - W / 2, K[1, 2] - H / 2])
    for k3, k2 in (('kp3d_cam', 'kp2d'), ('kp3d_cam_raw', 'kp2d_raw')):
        if k3 not in z:
            continue
        X = z[k3].astype(np.float64)
        X[..., 0] -= X[..., 2] * off[0] / K[0, 0]
        X[..., 1] -= X[..., 2] * off[1] / K[1, 1]
        Zc = np.clip(X[..., 2], 1e-6, None)
        z[k3] = X.astype(z[k3].dtype)
        if k2 in z:
            z[k2] = np.stack([K[0, 0] * X[..., 0] / Zc + K[0, 2],
                              K[1, 1] * X[..., 1] / Zc + K[1, 2]], -1).astype(z[k2].dtype)
    z['pp_offset_px'] = off.astype(np.float32); z['pp_corrected'] = np.bool_(True)
    np.savez(a.npz, **z)
    print(f'pp_correct: principal point offset ({off[0]:+.1f}, {off[1]:+.1f}) px removed from {len(z["frame_idx"])} rows')


if __name__ == '__main__':
    main()
