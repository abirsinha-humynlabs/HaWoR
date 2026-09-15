# Delivery-format driver for HaWoR

Runs HaWoR on egocentric clips and emits the 21-keypoint delivery npz consumed by the
`viz_delivery` renderer and `fusion/evaluate_run.py` scorer.

| file | what it does |
|---|---|
| `run_hawor_clip.py` | calls HaWoR's four stages unmodified, then maps WORLD -> CAMERA and writes the npz |
| `qc_hawor.py` | QC numbers plus drawn skeleton frames (aggregate metrics alone are not enough) |
| `postprocess_v3.py` | v3 post-processing: drop fabricated rows, smooth keypoints, smooth head pose |

## Versions

* **v1** raw HaWoR.
* **v2** `HAWOR_TRACK_FIX=1` — track-id handedness in `detect_track` (see its docstring). Still raw
  model output; the flag only stops the pipeline discarding detections it already has.
* **v3** v2 plus `postprocess_v3.py`. **Not raw model output.** Every changed value is flagged
  (`smoothed`) and the original kept (`kp2d_raw`, `kp3d_cam_raw`).

## WORLD -> CAMERA

`hawor_infiller` returns MANO params in DROID-SLAM world space; the delivery format needs camera
space, so joints are mapped back with the same extrinsics `demo.py --vis_mode cam` uses:
`X_cam = R_w2c @ X_world + t_w2c`. demo.py additionally left-multiplies cameras and vertices by
`R_x = diag(1,-1,-1)`, which cancels. `--check-roundtrip` verifies this numerically against the
pre-SLAM `cam_space/*.json` instead of trusting the algebra: measured max error 1.7e-7 m.

## Measured on episode_047

|  | rows | fabricated | renderer L/R flips | jumps >100 px | palm max px |
|---|---|---|---|---|---|
| v1 | 3646 | 979 | 133 | 587 | 5419 |
| v2 | 3646 | 885 | 129 | 453 | 1891 |
| v3 | 2761 | 0 | 6 | 34 | 369 |

The left/right flipping is the RENDERER's, not HaWoR's: `render_v3.assign_handedness` ignores
`is_right_*` and relabels per frame by wrist x. A fabricated hand crossing the midline swaps both
labels. Removing fabricated rows removes the cause.

## Credit

The head-pose and keypoint smoothing in `postprocess_v3.py` take their core ideas from
https://github.com/Maiemdiab/egocentric-hand-stabilisation — specifically smoothing SE(3) rotations
as matrices with SVD re-projection, zero-phase local polynomial fits evaluated at the sample, and
reporting motion retention so a filter that eats real motion cannot pass as a fix. Bone-length
rigidification from that repo is deliberately NOT applied: it would drive `bone_cv_median` to zero
and make that metric uninformative.
