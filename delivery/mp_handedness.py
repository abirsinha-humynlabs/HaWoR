#!/usr/bin/env python
"""Independent per-frame hand handedness from MediaPipe Hands.

HaWoR's handedness is a per-box class from the WiLoR YOLO detector, and it is sometimes wrong for a
whole track at a time - which the track-id vote in detect_track then propagates consistently, so the
error is stable rather than flickering and cannot be caught by self-consistency. A genuinely
independent model is the only way to catch it.

Emits, per frame, every detected hand's wrist pixel plus MediaPipe's own left/right label and score.
Nothing is decided here; fix_handedness.py does the matching and voting.

MediaPipe's Left/Right refers to the ANATOMICAL hand and assumes a non-mirrored image, which is
correct for an egocentric (world-facing) camera.

Runs in its own env (numpy 2.x) because mediapipe's numpy pin conflicts with HaWoR's 1.26.4.
"""
import argparse, json
import cv2
import numpy as np
import mediapipe as mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--min-det', type=float, default=0.3)
    ap.add_argument('--max-frames', type=int, default=0)
    a = ap.parse_args()

    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                     min_detection_confidence=a.min_det,
                                     min_tracking_confidence=a.min_det)
    cap = cv2.VideoCapture(a.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rows = []
    f = 0
    while True:
        ok, img = cap.read()
        if not ok or (a.max_frames and f >= a.max_frames):
            break
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                c = hd.classification[0]
                w = lm.landmark[0]
                rows.append({'frame': f,
                             'is_right': int(c.label == 'Right'),
                             'score': float(c.score),
                             'wrist_x': float(w.x * W), 'wrist_y': float(w.y * H)})
        f += 1
    cap.release(); hands.close()
    np.savez(a.out,
             frame=np.array([r['frame'] for r in rows], np.int64),
             is_right=np.array([r['is_right'] for r in rows], np.int8),
             score=np.array([r['score'] for r in rows], np.float32),
             wrist=np.array([[r['wrist_x'], r['wrist_y']] for r in rows], np.float32),
             width=W, height=H, frames=f)
    print(json.dumps({'frames_read': f, 'mp_detections': len(rows),
                      'frames_with_a_hand': len({r['frame'] for r in rows}),
                      'right': int(sum(r['is_right'] for r in rows)),
                      'left': int(len(rows) - sum(r['is_right'] for r in rows))}, indent=2))


if __name__ == '__main__':
    main()
