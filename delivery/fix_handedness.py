#!/usr/bin/env python
"""Correct per-TRACK handedness using an independent second model, and persist it for the renderer.

Two distinct defects, measured on episode_047_v3:

A. HaWoR'S LABEL IS WRONG FOR A WHOLE TRACK.  Handedness is a per-box class from the WiLoR YOLO
   detector. detect_track's track-id vote (HAWOR_TRACK_FIX) makes it *consistent* per track, which
   is right, but a consistently-wrong track stays consistently wrong and no self-consistency check
   can see it. Reviewers found the right hand drawn as left at frames 712-742.

B. THE RENDERER RELABELS AND FLIPS.  render_v3.assign_handedness ignores is_right_* entirely and
   labels per frame by wrist x (leftmost = left), then takes a per-track majority over only the
   frames where TWO hands were present. After v3 drops fabricated rows most frames have one hand,
   so that vote rests on very few frames and can inx  vert a whole track: at 795-807 HaWoR says
   R=13/14 and the renderer draws L=13/14. MediaPipe detects nothing there, so no second model can
   help - the fix is to stop the renderer guessing.

This script fixes A and emits what is needed to fix B.

MEDIAPIPE'S CONVENTION.  MediaPipe assigns handedness assuming the image is MIRRORED (front-facing
selfie). This footage is world-facing, so its labels must be swapped. Verified, not assumed:
agreement with HaWoR is 12.8% raw and 87.2% swapped, and an independent geometric check (on
two-hand frames, is HaWoR's 'left' the leftmost?) holds 99.4% of the time - so HaWoR is broadly
right and it is the raw MediaPipe label that is inverted.

VOTING.  Per track, not per frame, so one bad frame cannot flip a hand. A track is flipped only on
strong, consistent evidence, because a wrong flip is worse than the original error.
"""
import argparse, json
import numpy as np

WRIST = 0


def build_tracks(fi, k2, gate=150.0, maxgap=10, min_len=1):
    """Same greedy wrist-proximity tracker the delivery renderer uses, so tracks line up."""
    order = np.argsort(fi, kind='stable'); tk = []
    for gi in order:
        f = int(fi[gi]); xy = k2[gi, WRIST]; best, bd = None, gate
        for t in tk:
            if 0 < f - t['lf'] <= maxgap:
                d = float(np.hypot(*(t['lxy'] - xy)))
                if d < bd: best, bd = t, d
        if best is None: tk.append(dict(lxy=xy, lf=f, idx=[gi]))
        else: best['lxy'] = xy; best['lf'] = f; best['idx'].append(gi)
    return [np.asarray(t['idx']) for t in tk if len(t['idx']) >= min_len]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--mp', required=True, help='mp_handedness.py output')
    ap.add_argument('--no-swap-mp', action='store_true',
                    help='do NOT swap MediaPipe L/R (only for genuinely mirrored footage)')
    ap.add_argument('--gate', type=float, default=250.0,
                    help='wrist-proximity gate for the VOTING tracker, px. Deliberately looser than '
                         'the renderer\'s 150: handedness belongs to a physical hand over a long '
                         'span, and fast motion fragments a hand into pieces too small to vote on '
                         '(episode_047 712-742 was 1 of 5 fragments). Too loose over-merges two '
                         'different hands - at gate 500 this clip fused frames 390-826 into one.')
    ap.add_argument('--maxgap', type=int, default=20)
    ap.add_argument('--min-score', type=float, default=0.80)
    ap.add_argument('--min-votes', type=int, default=5)
    ap.add_argument('--min-frac', type=float, default=0.70)
    a = ap.parse_args()

    z = dict(np.load(a.npz, allow_pickle=False))
    fi, k2 = z['frame_idx'], z['kp2d']
    hand = z['is_right_wilor'].astype(int).copy()
    palm = float(np.median(np.linalg.norm(k2[:, 17, :] - k2[:, 5, :], axis=-1)))

    m = np.load(a.mp)
    mf, mw, msc = m['frame'], m['wrist'], m['score']
    mr = m['is_right'].astype(int)
    if not a.no_swap_mp:
        mr = 1 - mr
    by = {}
    for j in range(len(mf)): by.setdefault(int(mf[j]), []).append(j)

    # match each row to at most one MediaPipe detection, gated on palm width
    match = {}
    for i in range(len(fi)):
        cands = by.get(int(fi[i]), [])
        if not cands: continue
        dd = sorted((float(np.hypot(*(mw[j] - k2[i, WRIST]))), j) for j in cands)
        if dd[0][0] <= 1.5 * palm: match[i] = dd[0][1]

    tracks = build_tracks(fi, k2, gate=a.gate, maxgap=a.maxgap)
    flipped_tracks = 0; flipped_rows = 0; details = []
    for tr in tracks:
        votes = [(mr[match[i]], float(msc[match[i]])) for i in tr
                 if i in match and msc[match[i]] >= a.min_score]
        if len(votes) < a.min_votes: continue
        cur = int(np.round(hand[tr].mean()))
        disagree = sum(w for v, w in votes if v != cur)
        total = sum(w for _, w in votes)
        if total > 0 and disagree / total >= a.min_frac:
            hand[tr] = 1 - cur
            flipped_tracks += 1; flipped_rows += len(tr)
            details.append({'frames': [int(fi[tr].min()), int(fi[tr].max())], 'rows': int(len(tr)),
                            'was': 'right' if cur else 'left', 'now': 'left' if cur else 'right',
                            'mp_votes': len(votes),
                            'mp_disagreement': round(float(disagree / total), 3)})

    z['is_right_wilor'] = hand.astype(np.int8)
    z['is_right_mp'] = hand.astype(np.int8)
    # `hand` is what a patched renderer reads to stop re-deriving handedness from wrist x
    z['hand'] = hand.astype(np.int8)
    np.savez(a.out, **z)
    rep = {'rows': int(len(fi)), 'mp_matched': len(match),
           'mp_swapped': not a.no_swap_mp, 'tracks': len(tracks),
           'tracks_flipped': flipped_tracks, 'rows_flipped': flipped_rows, 'detail': details}
    print(json.dumps(rep, indent=2))
    with open(a.out.replace('.npz', '_handedness_report.json'), 'w') as fh:
        json.dump(rep, fh, indent=2)


if __name__ == '__main__':
    main()
