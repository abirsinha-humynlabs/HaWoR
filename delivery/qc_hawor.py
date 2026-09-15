#!/usr/bin/env python
"""QC one delivery npz: numbers AND drawn frames.

The reprojection gate is reported but is NOT an independent check here: this producer DEFINES
kp2d as project(kp3d_cam, K), so the error is ~0 by construction. It proves the two arrays agree,
not that either sits on a hand. The drawn frames are what actually tests that, which is why this
script always writes them.
"""
import argparse, json, os, sys

import numpy as np
import cv2

sys.path.insert(0, os.path.expanduser('~/projects/EgoForce'))
from fusion.topology import HAND_EDGES, PALM_EDGES, FINGER_CHAINS, bone_lengths, project_pinhole


def draw(frame, kp2d, is_right):
    for a, b in PALM_EDGES:
        cv2.line(frame, tuple(np.int32(kp2d[a])), tuple(np.int32(kp2d[b])), (190, 190, 190), 1, cv2.LINE_AA)
    for _, chain, col in FINGER_CHAINS:
        for a, b in zip(chain[:-1], chain[1:]):
            cv2.line(frame, tuple(np.int32(kp2d[a])), tuple(np.int32(kp2d[b])), col, 2, cv2.LINE_AA)
        for j in chain[1:]:
            cv2.circle(frame, tuple(np.int32(kp2d[j])), 3, col, -1, cv2.LINE_AA)
    w = tuple(np.int32(kp2d[0]))
    cv2.circle(frame, w, 6, (140, 210, 255) if is_right else (255, 214, 120), -1, cv2.LINE_AA)
    cv2.putText(frame, 'R' if is_right else 'L', (w[0] + 10, w[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(frame, 'R' if is_right else 'L', (w[0] + 10, w[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True); ap.add_argument('--video', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--clip', required=True)
    ap.add_argument('--frames', default='', help='comma list; default = auto-chosen')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = np.load(a.npz, allow_pickle=False)
    kp3d, kp2d, fi = d['kp3d_cam'], d['kp2d'], d['frame_idx']
    K = d['K']; meas = d['depth_measured'].astype(bool); isr = d['is_right_wilor'].astype(int)
    R = {}

    # -- reprojection (self-consistency only; see module docstring)
    err = np.linalg.norm(project_pinhole(kp3d, K) - kp2d, axis=-1)
    R['reproj'] = dict(median_px=float(np.median(err)), p95_px=float(np.percentile(err, 95)),
                       max_px=float(err.max()),
                       note='kp2d is DEFINED as project(kp3d_cam,K); ~0 by construction, not an '
                            'independent validation')

    # -- depth
    Z = kp3d[..., 2]
    R['depth'] = dict(
        rows=int(len(fi)),
        rows_any_joint_behind_lens=int((Z <= 0).any(1).sum()),
        frac_any_joint_behind_lens=round(float((Z <= 0).any(1).mean()), 5),
        rows_any_joint_under_5cm=int((Z < 0.05).any(1).sum()),
        frac_any_joint_under_5cm=round(float((Z < 0.05).any(1).mean()), 5),
        rows_any_joint_over_3m=int((Z > 3.0).any(1).sum()),
        wrist_Z_m=dict(p1=float(np.percentile(Z[:, 0], 1)), median=float(np.median(Z[:, 0])),
                       p99=float(np.percentile(Z[:, 0], 99)),
                       min=float(Z[:, 0].min()), max=float(Z[:, 0].max())))

    # -- bone lengths (all rows, and observed-only)
    def bstats(x):
        if len(x) < 2: return {}
        L = bone_lengths(x); m = L.mean(0); cv = L.std(0) / np.clip(m, 1e-9, None)
        return dict(n=int(len(x)), bone_len_mean_cm=round(float(m.mean() * 100), 3),
                    bone_cv_median=round(float(np.median(cv)), 5),
                    bone_cv_max=round(float(cv.max()), 5))
    # per-finger chain length: an independent check that the 21 joints really are in
    # OpenPose order. If the order were wrong these would not read as fingers.
    obs = kp3d[meas] if meas.any() else kp3d
    R['finger_chain_cm'] = {}
    for nm, chain, _ in FINGER_CHAINS:
        seg = sum(float(np.median(np.linalg.norm(obs[:, b] - obs[:, a], axis=-1)))
                  for a, b in zip(chain[:-1], chain[1:]))
        R['finger_chain_cm'][nm] = round(seg * 100, 2)
    R['palm_width_cm'] = round(float(np.median(
        np.linalg.norm(obs[:, 17] - obs[:, 5], axis=-1))) * 100, 2)

    R['bones_all_rows'] = bstats(kp3d)
    R['bones_observed_only'] = bstats(kp3d[meas])

    # -- 2D jitter: median frame-to-frame wrist displacement, per hand, consecutive frames only
    R['jitter_px'] = {}
    for h, name in ((1, 'right'), (0, 'left')):
        sel = isr == h
        f, w = fi[sel], kp2d[sel][:, 0, :]
        o = np.argsort(f); f, w = f[o], w[o]
        step = np.diff(f) == 1
        if step.sum():
            dz = np.linalg.norm(np.diff(w, axis=0)[step], axis=-1)
            R['jitter_px'][name] = dict(n=int(step.sum()), median=round(float(np.median(dz)), 3),
                                        p95=round(float(np.percentile(dz, 95)), 3))
    R['rows'] = dict(total=int(len(fi)), observed=int(meas.sum()), infilled=int((~meas).sum()),
                     left=int((isr == 0).sum()), right=int((isr == 1).sum()))

    # -- pick frames to LOOK at
    if a.frames:
        picks = [int(x) for x in a.frames.split(',')]
    else:
        both = [int(f) for f in np.unique(fi) if (fi == f).sum() >= 2]
        obs = fi[meas]
        picks = []
        if len(obs):
            picks += [int(obs.min()), int(np.percentile(obs, 25)), int(np.median(obs)),
                      int(np.percentile(obs, 75)), int(obs.max())]
        if both:
            picks += [both[len(both) // 2], both[len(both) // 4]]
        picks = sorted(set(picks))

    cap = cv2.VideoCapture(a.video)
    made = []
    for f in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, frame = cap.read()
        if not ok: continue
        rows = np.where(fi == f)[0]
        for i in rows:
            draw(frame, kp2d[i], bool(isr[i]))
        tag = '+'.join(('R' if isr[i] else 'L') + ('' if meas[i] else '~') for i in rows)
        cv2.putText(frame, f'{a.clip}  frame {f}  [{tag}]  (~ = infilled)', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, f'{a.clip}  frame {f}  [{tag}]  (~ = infilled)', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        p = os.path.join(a.out, f'{a.clip}_f{f:05d}.png')
        cv2.imwrite(p, frame); made.append(p)
    cap.release()
    R['frames_drawn'] = made
    print(json.dumps(R, indent=2))
    with open(os.path.join(a.out, f'{a.clip}_qc.json'), 'w') as fh:
        json.dump(R, fh, indent=2)


if __name__ == '__main__':
    main()
