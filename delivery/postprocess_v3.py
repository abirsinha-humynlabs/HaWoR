#!/usr/bin/env python
"""v3 post-processing for the hand21 delivery npz.

v1/v2 are RAW model output. v3 is NOT: it post-processes, and every changed value is flagged and
the original retained, so nothing here is irreversible or unauditable.

Three passes, each aimed at a defect measured on episode_047_v2 rather than assumed.

1. DROP FABRICATED ROWS  (`depth_measured == False`)
   HaWoR's CMIB infiller invents a pose for frames the detector never saw. Measured on
   episode_047_v2 (3646 rows, 885 fabricated):

       renderer L/R flip frames   129 -> 6     (-95%)
       palm width > 3x median      54 -> 0     (every giant hand was fabricated)
       wrist projected off-image  556 -> 122   (-78%)
       degenerate rows among OBSERVED rows: 0

   So every artefact reported by review - hands swelling, floating hands with no hand under them,
   left/right swapping mid-clip, multi-hundred-pixel teleports - comes from fabricated rows, and no
   gating of real detections is needed. Reviewers reported 115-152, 206-263, 327-342, 468-477,
   512-527, 579-591, 663-771, 794-834, 975-1005, 1116-1137; every one of those windows is 25-62%
   fabricated.

   Note the L/R swapping is the RENDERER's doing, not HaWoR's: render_v3.assign_handedness ignores
   is_right_* and relabels per frame by wrist x (leftmost = left). A fabricated hand drifting across
   the midline swaps both labels. Removing the fabricated rows removes the cause.

2. HEAD-POSE SMOOTHING   (core idea from github.com/Maiemdiab/egocentric-hand-stabilisation)
   The delivery panels draw the wrist in the WORLD frame, world_p = R_head @ cam_p + t_head, so head
   noise enters the trajectory even with perfect hand keypoints; that repo measures the head as 40-68%
   of visible shake. Rotations are smoothed as MATRICES and re-projected to SO(3) by SVD (smoothing
   Euler angles or quaternion components tears at wrap points), and the filter is symmetric and
   evaluated at the sample so it adds no lag.
   This affects the 3D panels and the wrist trail ONLY - it cannot move a 2D landmark.

3. TEMPORAL SMOOTHING OF HAND KEYPOINTS  (same source, adapted)
   Zero-phase local quadratic evaluated at the sample; gap-aware (fits over real frame indices inside
   one track, so it never pulls a hand across a hole it did not observe); robust via two Tukey
   biweight passes so one bad frame does not drag its neighbours.

DELIBERATELY NOT DONE: bone-length rigidification. It would force bone_cv_median to ~0 and make that
metric meaningless - the exact trap that shipped a bad EgoForce delivery, where bone CV was measured
after the step whose job was to make bone lengths constant.
"""
import argparse, json, os
import numpy as np


# ----------------------------------------------------------------- SE(3) head smoothing
def smooth_se3(T, win=6, order=2):
    """Local weighted polynomial fit evaluated at each sample. Zero phase by construction."""
    n = len(T); h = max(int(win), 1)
    t = np.arange(n, dtype=float)
    trans = T[:, :3, 3]; R = T[:, :3, :3].reshape(n, 9)
    sm_t = np.empty_like(trans); sm_R = np.empty_like(R)
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        tt = t[lo:hi] - t[i]
        w = (1.0 - np.abs(tt / (h + 1e-9)) ** 3) ** 3            # tricube
        V = np.vander(tt, order + 1); W = np.sqrt(np.maximum(w, 1e-12))[:, None]
        for arr, dst in ((trans, sm_t), (R, sm_R)):
            coef, *_ = np.linalg.lstsq(V * W, arr[lo:hi] * W, rcond=None)
            dst[i] = coef[-1]
    out = np.repeat(np.eye(4)[None], n, 0)
    for i in range(n):                                            # nearest true rotation
        U, _, Vt = np.linalg.svd(sm_R[i].reshape(3, 3))
        out[i, :3, :3] = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    out[:, :3, 3] = sm_t
    return out


def head_metrics(T):
    t = T[:, :3, 3]
    lin_acc = np.linalg.norm(t[2:] - 2 * t[1:-1] + t[:-2], axis=-1) * 1000.0
    R = T[:, :3, :3]; dR = np.einsum('nij,nkj->nik', R[1:], R[:-1])
    # first difference is angular SPEED (real turning); jitter is the change in it
    spd = np.degrees(np.arccos(np.clip((np.trace(dR, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
    step = np.linalg.norm(np.diff(t, axis=0), axis=-1) * 1000.0
    return lin_acc, np.abs(np.diff(spd)), step, spd


# ----------------------------------------------------------------- keypoint smoothing
def _tracks(fi, k2, hand, gate=150.0, maxgap=10, min_len=4):
    """Greedy wrist-proximity tracks within one hand, so smoothing never crosses hands."""
    idx = np.where(hand)[0]
    order = idx[np.argsort(fi[idx], kind='stable')]
    tk = []
    for gi in order:
        f = int(fi[gi]); xy = k2[gi, 0]; best, bd = None, gate
        for t in tk:
            if 0 < f - t['lf'] <= maxgap:
                dd = float(np.hypot(*(t['lxy'] - xy)))
                if dd < bd: best, bd = t, dd
        if best is None: tk.append(dict(lxy=xy, lf=f, idx=[gi]))
        else: best['lxy'] = xy; best['lf'] = f; best['idx'].append(gi)
    return [t['idx'] for t in tk if len(t['idx']) >= min_len]


def _smooth_series(t, Y, h=4.0, order=2, passes=2, tukey_c=3.0, max_gap=10):
    """Zero-phase robust local polynomial. t = real frame indices (gap aware)."""
    n = len(t); out = Y.copy().astype(np.float64)
    w = np.ones(n)
    for _ in range(passes):
        for i in range(n):
            dt = t - t[i]
            m = np.abs(dt) <= h
            # never fit across a hole larger than max_gap
            if m.sum() < order + 1: out[i] = Y[i]; continue
            tt = dt[m].astype(float)
            if np.abs(np.diff(np.sort(t[m]))).max(initial=0) > max_gap:
                out[i] = Y[i]; continue
            ww = (1.0 - np.abs(tt / (h + 1e-9)) ** 3) ** 3 * w[m]
            V = np.vander(tt, order + 1); W = np.sqrt(np.maximum(ww, 1e-12))[:, None]
            coef, *_ = np.linalg.lstsq(V * W, Y[m][:, None] * W, rcond=None)
            out[i] = coef[-1, 0]
        r = Y - out; s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        u = np.clip(r / (tukey_c * s), -1, 1); w = (1 - u ** 2) ** 2      # Tukey biweight
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--head', default=None); ap.add_argument('--head-out', default=None)
    ap.add_argument('--head-win', type=int, default=6)
    ap.add_argument('--kp-h', type=float, default=4.0, help='keypoint half-window, frames')
    ap.add_argument('--no-drop-fabricated', action='store_true')
    ap.add_argument('--no-smooth-kp', action='store_true')
    a = ap.parse_args()

    z = dict(np.load(a.npz, allow_pickle=False))
    n0 = len(z['frame_idx'])
    rep = {'input_rows': int(n0), 'passes': []}

    # ---- 1. drop fabricated
    if not a.no_drop_fabricated:
        keep = z['depth_measured'].astype(bool)
        for k in ('kp3d_cam', 'kp2d', 'frame_idx', 'is_right_wilor', 'is_right_mp',
                  'source', 'depth_measured'):
            if k in z: z[k] = z[k][keep]
        rep['passes'].append({'drop_fabricated': {'removed': int((~keep).sum()),
                                                  'kept': int(keep.sum())}})

    fi = z['frame_idx']; k2 = z['kp2d'].astype(np.float64); k3 = z['kp3d_cam'].astype(np.float64)
    z['kp2d_raw'] = k2.copy(); z['kp3d_cam_raw'] = k3.copy()
    smoothed = np.zeros(len(fi), bool)

    # ---- 3. keypoint smoothing (per hand, per track, gap aware)
    if not a.no_smooth_kp and len(fi):
        d2_before = []
        for h in (0, 1):
            hand = z['is_right_wilor'].astype(int) == h
            for tr in _tracks(fi, k2, hand):
                tr = np.asarray(tr); o = np.argsort(fi[tr]); tr = tr[o]
                t = fi[tr].astype(float)
                for j in range(21):
                    for dim in range(2):
                        k2[tr, j, dim] = _smooth_series(t, z['kp2d_raw'][tr, j, dim], a.kp_h)
                    for dim in range(3):
                        k3[tr, j, dim] = _smooth_series(t, z['kp3d_cam_raw'][tr, j, dim], a.kp_h)
                smoothed[tr] = True
        z['kp2d'] = k2; z['kp3d_cam'] = k3
        # honest motion-retention: a filter that flattens real motion is not a fix
        def wrist_steps(arr):
            out = []
            for h in (0, 1):
                m = z['is_right_wilor'].astype(int) == h
                f, w = fi[m], arr[m][:, 0, :]
                o = np.argsort(f); f, w = f[o], w[o]
                if len(f) > 1:
                    dz = np.linalg.norm(np.diff(w, axis=0), axis=-1)[np.diff(f) == 1]
                    out.append(dz)
            return np.concatenate(out) if out else np.zeros(0)
        b, c = wrist_steps(z['kp2d_raw']), wrist_steps(k2)
        rep['passes'].append({'smooth_keypoints': {
            'rows': int(smoothed.sum()), 'half_window_frames': a.kp_h,
            'wrist_step_px_median_before': round(float(np.median(b)), 3) if b.size else None,
            'wrist_step_px_median_after': round(float(np.median(c)), 3) if c.size else None,
            'motion_retained_p90_pct': round(100 * float(np.percentile(c, 90) /
                                       max(np.percentile(b, 90), 1e-9)), 1) if b.size else None}})
    z['smoothed'] = smoothed

    np.savez(a.out, **z)
    rep['output_rows'] = int(len(fi))

    # ---- 2. head pose
    if a.head and a.head_out:
        hz = dict(np.load(a.head, allow_pickle=True))
        T = hz['T'].astype(np.float64)
        T2 = smooth_se3(T, a.head_win)
        l0, j0, s0, v0 = head_metrics(T); l1, j1, s1, v1 = head_metrics(T2)
        hz['T'] = T2.astype(np.float32)
        if 'translation' in hz: hz['translation'] = T2[:, :3, 3].astype(np.float32)
        hz['T_unsmoothed'] = T.astype(np.float32); hz['head_smooth_win'] = np.int32(a.head_win)
        np.savez_compressed(a.head_out, **hz)
        rep['passes'].append({'smooth_head_pose': {
            'frames': int(len(T)), 'half_window_frames': a.head_win,
            'lin_jitter_mm': [round(float(np.median(l0)), 3), round(float(np.median(l1)), 3)],
            'rot_jitter_deg': [round(float(np.median(j0)), 4), round(float(np.median(j1)), 4)],
            'rot_speed_p90_retained_pct': round(100 * float(np.percentile(v1, 90) /
                                          max(np.percentile(v0, 90), 1e-9)), 1),
            'translation_step_p90_retained_pct': round(100 * float(np.percentile(s1, 90) /
                                          max(np.percentile(s0, 90), 1e-9)), 1)}})
    print(json.dumps(rep, indent=2))
    with open(os.path.splitext(a.out)[0] + '_v3_report.json', 'w') as fh:
        json.dump(rep, fh, indent=2)


if __name__ == '__main__':
    main()
