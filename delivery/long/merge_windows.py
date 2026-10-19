#!/usr/bin/env python
"""Merge per-window HaWoR raw npz files (run_hawor_clip.py output) into one clip-level raw npz.

Long clips (5-10 min) are run as overlapping ~60 s windows (see run_v3_long.sh) because one T4 /
16 GB host cannot hold a whole clip through motion estimation and DROID-SLAM. Each window covers
[start, end) of the clip's frames; only its CORE rows [core_start, core_end) are kept, so every
frame comes from exactly one window and the overlap only gives the window edges temporal context.

Output has the same keys as run_hawor_clip.py's npz, with frame_idx in CLIP frame numbers, plus
`window` (index of the window each row came from). Width/height/fps are the full video's.
"""
import argparse, json, os
import numpy as np

ROW_KEYS = ('kp3d_cam', 'kp2d', 'frame_idx', 'is_right_wilor', 'is_right_mp', 'source', 'depth_measured')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', required=True, help='windows.json written by run_v3_long.sh')
    ap.add_argument('--work', required=True, help='dir holding <name>.out/<name>_raw_hand21_keypoints.npz')
    ap.add_argument('--out', required=True, help='merged raw npz')
    ap.add_argument('--sibling', required=True, help='merged <clip>_3d_keypoints.npz (coverage denominator)')
    ap.add_argument('--meta-out', required=True, help='merged per-window metadata json')
    a = ap.parse_args()

    W = json.load(open(a.windows))
    N, width, height, fps = W['frames'], W['width'], W['height'], W['fps']
    parts, metas, K = {k: [] for k in ROW_KEYS}, [], None
    covered = np.zeros(N, dtype=np.int32)
    for i, w in enumerate(W['windows']):
        d = os.path.join(a.work, w['name'] + '.out')
        z = np.load(os.path.join(d, f"{w['name']}_raw_hand21_keypoints.npz"))
        if K is None:
            K = z['K']
        assert np.allclose(z['K'], K), f"K differs in window {w['name']}"
        f = z['frame_idx'].astype(np.int64) + w['start']
        keep = (f >= w['core_start']) & (f < w['core_end'])
        for k in ROW_KEYS:
            parts[k].append(f[keep] if k == 'frame_idx' else z[k][keep])
        parts.setdefault('window', []).append(np.full(int(keep.sum()), i, dtype=np.int16))
        covered[w['core_start']:w['core_end']] += 1
        mp = os.path.join(d, f"{w['name']}_raw_hawor_metadata.json")
        m = json.load(open(mp)) if os.path.exists(mp) else {}
        metas.append(dict(window=w, rows_kept=int(keep.sum()), rows_total=int(len(f)),
                          roundtrip=m.get('roundtrip'), hawor_frames=m.get('frames'),
                          slam_frames=m.get('slam_frames'), intrinsics=m.get('intrinsics'),
                          repo_commit=m.get('repo_commit')))
    assert (covered == 1).all(), 'window cores must tile the clip exactly once'

    out = {k: np.concatenate(v) for k, v in parts.items()}
    order = np.lexsort((out['is_right_wilor'], out['frame_idx']))
    out = {k: v[order] for k, v in out.items()}
    np.savez(a.out, **out, K=K, width=width, height=height, fps=fps, step=1)
    np.savez(a.sibling, processed_frames=np.arange(N, dtype=np.int64), frame_idx=out['frame_idx'],
             K=K, width=width, height=height, fps=fps, step=1)
    json.dump(dict(model='HaWoR', windowed=True, frames=N, window_len=W['window_len'],
                   margin=W['margin'], windows=metas, rows=int(len(order)),
                   rows_measured=int(out['depth_measured'].sum())), open(a.meta_out, 'w'), indent=2)
    print(f"merged {len(W['windows'])} windows -> {len(order)} rows "
          f"({int(out['depth_measured'].sum())} measured) over {N} frames")


if __name__ == '__main__':
    main()
