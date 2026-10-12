> **Superseded by `docs/HANDOFF.md` (2026-10-12).** Kept verbatim for history; some statements here were later corrected (e.g. the S3 env snapshot is the Sep-16 fleet env, not the v3 env; v3 ran in conda torch 1.13+cu117).

# HaWoR — handoff

Everything established about HaWoR in the prior session, so you don't rediscover it.
Repo: `https://github.com/abirsinha-humynlabs/HaWoR.git`

Written 2026-10-07. Facts here were verified against S3 and the papers; where something is an
inference rather than a measurement it says so.

---

## 1. Start here: there is already a working environment snapshot

**`s3://stage-humyn-egocentric-stereo-data/labelling_results/hand_pose_HaWoR/_env_snapshot/hawor_env_snapshot.tgz`** (763 KB)

This is the single most valuable artefact. It contains a *working* HaWoR setup from a previous run:

| File | What it gives you |
| --- | --- |
| `requirements_hawor_venv.txt` | The exact pinned env — `torch==2.8.0+cu128`, 16 `nvidia-*-cu12` wheels, `lietorch==0.2`, `pytorch3d` at a pinned commit |
| `requirements_render_venv.txt` | A separate venv for rendering |
| `batch_hawor.sh` | The fleet runner: per-clip scratch dir, resumable via `.ok` markers, uploads then deletes |
| `jobs.json` | The clip list with `clip`, `input_dir`, `img_focal`, `K` per entry |
| `patches/*.patch` | Four patches against the vendored DROID-SLAM / LieTorch |
| `logs/` | Build logs: `lietorch_build`, `droid_build`, `droid_build2`, `setup_*`, `smoke`, `batch` |
| `contested.py`, `qc_overlay.py`, `panels.tsv` | QC and overlay tooling |

**Do not rebuild from scratch before reading this.** The CUDA extension builds are the hard part.

### What the patches actually are

All four are **torch 2.8 API-compatibility fixes only** — `a.type()` → `a.scalar_type()`, because
torch 2.8 removed the deprecated call. Nobody was changing behaviour. They apply to:
`lietorch_cpu.cpp`, `lietorch_gpu.cu`, `correlation_kernels.cu`, `altcorr_kernel.cu`.

### Two commit hashes that disagree — resolve this first

- `patches/repo_commit.txt` → `883479d638a3e202ee35597b577276a8c5012a97`
- run metadata `repo_commit` → `66c7d4108d58a716deccd192cb7645170cdc7bd7`

I did not determine which is the HaWoR repo and which is a vendored dependency. Check both against
the fork before assuming either is the version that produced the published outputs.

---

## 2. Hard requirements

**GPU is mandatory.** HaWoR vendors DROID-SLAM at
`HaWoR/thirdparty/DROID-SLAM/thirdparty/lietorch/`, and builds two extensions:

| Extension | CPU path? |
| --- | --- |
| `lietorch_backends` | **yes** — `lietorch_cpu.cpp` exists |
| `droid_backends` | **no** — only `.cu` sources (`correlation_kernels.cu`, `altcorr_kernel.cu`) |

`droid_slam/motion_filter.py` is also decorated `@torch.cuda.amp.autocast(enabled=True)`
unconditionally. There is no `--cpu` flag, no device switch, no fallback anywhere in the scripts.

**Architecturally** HaWoR is two stacked stages: a camera-frame hand ViT + infiller (pure PyTorch,
*would* run on CPU slowly), and world-space lifting via DROID-SLAM (**CUDA-only**). If you only need
camera-frame `kp3d_cam`, a CPU-only variant is conceivable by skipping SLAM — but that is untested,
`run_clip.py` may not expose it, and world-space is HaWoR's entire contribution.

**CUDA 12.8 / torch 2.8+cu128.** Note this is *incompatible* with the EgoForce env (12.6 / cu126).
Separate venvs; they cannot share one.

**Disk.** `batch_hawor.sh` warns: HaWoR extracts every frame to jpg *and* writes a frames × H × W
bool mask memmap, **~3.8 GB scratch per 60 s 1080p clip**. It works on NVMe (`/opt/dlami/nvme`) and
deletes each clip's dir immediately after upload. Plan accordingly.

**Previous runs used a Deep Learning AMI** (`/opt/dlami/…`, `/home/ec2-user/hawor_runs/`). That is
why the HaWoR setup avoided the driver pain that cost a full day on a bare Amazon Linux 2023 box
(versioned `kernel6.12-devel` packages, a 20 GB default root volume, and an instance role that could
not read AWS's S3 driver bucket). **Use a DLAMI.**

---

## 3. What has already been run

**20 clips** under `s3://…/labelling_results/hand_pose_HaWoR/`:

`episode_001, 002, 002_v2, 009, 046, 047, 047_v2, 047_v3, 048, 048_v2, 052`, plus named work clips:
`Hospitality_plating_salad_vegetables_64`, `Hospitality_pouring_pink_yogurt_into_bowls_73`,
`Hospitality_pouring_yogurrt_into_bowls_72`, `Others_blowing_190`,
`construction_cutting_ply_using_machine_516`, `construction_mixing_cement_…_503`,
`construction_removing_grooting_using_hand_blade_506`, `construction_sanding_pipe_507`.

Each clip directory holds: `_hand21_keypoints.npz`, `_3d_keypoints.npz`, `_hawor_metadata.json`,
`_hawor_wrist_traj_panels.mp4`, `eval/report.{json,md}`, `eval_v1_vs_v2/report.{json,md}`, and a
`qc/` directory with `DRIFT_qc.json` and drawn `DRIFT_f*.png` frames.

All inputs come from `validation-result/ZED/<region>/<clip>/left_eye.mp4` — a rectified pinhole
stereo rig used as a mono left eye. **All hand-pose work to date is ZED-only**, though the capture
programme also covers akai, Oak-D, bitrobot, DAS-EGO and imucam.

---

## 4. How HaWoR performed — it is the strongest candidate so far

From `episode_048_v2`'s own `qc/DRIFT_qc.json` (raw output, **no post-processing**):

| Check | Result |
| --- | --- |
| Joints behind the lens | **0** |
| Joints under 5 cm | **0** |
| Joints over 3 m | **0** |
| Wrist Z | 0.31–0.59 m, median **0.44** |
| Finger chains (cm) | thumb 12.97, index 16.56, **middle 17.15**, ring 15.95, pinky 14.04 |
| Palm width | 6.71 cm |
| Bone-length CV (median) | 0.0336 |
| Jitter (px, median) | right 6.02, left 6.33 |
| Evidence-backed rows | **3628 / 3646 = 99.5%** |

Those finger chains are **anatomically normal adult dimensions**. Contrast EgoForce on episode_047:
minimum depth **−0.419 m** (behind camera) and 11.5% of joints under 5 cm; and POEM-v2, which failed
its own gates on both hands with a minimum joint Z of **−21 m**. On physical plausibility HaWoR is in
a different class.

### vs the incumbent (WiLoR + MediaPipe pipeline, with ADAPT smoothing), episode_048

| | WiLoR-pipeline + ADAPT | HaWoR |
| --- | --- | --- |
| Rows with image evidence | 3277 (89.9%) | **3628 (99.5%)** |
| Right hand, frames with evidence | 85.0% | **99.3%** |
| Bone-length CV (median) | **0.0051** | 0.0336 |
| 2D acceleration (jitter) | **1.06 px** | ~6.0 px |
| Negative depth / under 5 cm | 0% / 0% | 0% / 0% |

**Read this correctly.** HaWoR wins coverage decisively, and coverage cannot be manufactured by
post-processing. HaWoR *loses* on stability — but it is **raw** while the incumbent is smoothed and
rigidified. Those two rows measure the post-processing, not the model. A same-level comparison has
never been run.

---

## 5. Traps and gotchas, all verified

**Row order is not shared across models.** HaWoR's npz and the WiLoR fused npz agree on handedness
at the same row index only **44.8%** of the time. Joining by row position mismatches hands on more
than half the rows and inflates apparent disagreement ~4× (117 px vs the true 26 px). **Always key on
`(frame, hand)`.**

**`depth_measured=False` means no image evidence.** From HaWoR's own provenance note: `True` means
the ViT ran on that image frame; `False` rows are the **transformer infiller's output, with no image
evidence**. On episode_048 that is 18 of 3646 rows. HaWoR also interpolates bounding boxes across
detector gaps, so `final_valid_frames` (1823/1823) is wider than `detector_fired_frames`
(left 1817 / right 1811). Counting infilled rows as tracking flatters the model.

**Reprojection is ~0 by construction, not by validation.** HaWoR's own QC says it: `kp2d` is
*defined* as `project(kp3d_cam, K)`. Do not cite a 0 px reprojection as evidence of accuracy.

**HaWoR assumes the principal point is the image centre.** `run_clip.py` takes a single
`--img_focal`. The metadata records the gap: true rectified K is fx=fy=1065.0692, cx=959.691,
cy=541.633, while HaWoR internally assumes (960, 540) — a `principal_point_offset_px` of
`[-0.309, 1.633]`. Small on this rig; check it on any other.

**Smoothing hides failure.** Running POEM-v2's raw output through temporal smoothing + rigidify
flipped it from FAIL on every gate to **PASS on every gate**, without recovering any accuracy — its
own report says so. The physical gates measure self-consistency, and smoothing manufactures
self-consistency directly. Treat any stabilised output with this in mind.

### Paper limitations (arXiv 2501.02973, PDF at `hand21kp/papers/2501.02973v1.pdf`)

HaWoR's stated weaknesses are **not** the ones that broke EgoForce and POEM-v2:

> reliance on hand-tracking outputs from an off-the-shelf method … particularly in cases of
> **tracking identity failures** … left/right hand tracking is incorrect

> HaWoR models each hand independently, without any inter-penetration constraints. This can cause
> **self-penetrations** when the two hands interact.

Plus runtime: 40 ms/frame, which the paper calls "far from real-time". Note `postprocess.py` in the
`hand21kp` repo already has machinery for left/right repair, so the first limitation is partly
covered downstream.

---

## 6. Tooling that already exists — reuse, don't rewrite

- **`hand21kp/local_dump/compare_mint_vs_hawor.py`** — model-vs-model comparison on GT-free criteria
  (coverage, reprojection, depth sanity, bone-length CV, hand scale, jitter), keyed on `(frame, hand)`.
  Re-runnable; regenerates `comparison_report.md`, `comparison.json`, `per_keypoint.csv`.
- **`hand21kp/local_dump/render_side_by_side.py`** — two models' skeletons on the same frames, with
  no-evidence rows drawn faint and tagged `INFILLED`.
- **`EgoForce/fusion/evaluate_run.py`** — the three delivery metrics (M1 coverage/recovery exact,
  M2 geometric validity exact, M3 fingertip agreement **proxy**). HaWoR's `eval/report.md` is already
  in this format, so the harness was reused.
- **`EgoForce/fusion/metrics.py`** — the underlying measures, with proxies clearly labelled.

---

## 7. Open questions worth picking up

1. **A same-level comparison.** Put HaWoR through the same stabilise stage the incumbent uses
   (`EgoForce` branches `run/type1-egoforce-only` or `run/type2-fusion` carry `fusion/stabilise/`),
   or compare raw-vs-raw. Until then no statement about underlying accuracy is supported.
2. **The three-way on identical clips.** HaWoR, EgoForce and POEM-v2 all ran on **episode_047 and
   episode_048**. That comparison costs no GPU time — the npz files are all in S3.
3. **Does `eval_v1_vs_v2` show a real improvement?** Those reports exist for several clips and I
   never read them.
4. **Ground truth.** `fusion/metrics.py::fingertip_error` is written and tested and has nothing to
   consume. Every accuracy claim so far is a proxy.
5. **A wider-baseline rig.** POEM-v2 failed because a ZED's two eyes sit ~12 cm apart facing the same
   direction, making triangulation near-degenerate along the depth axis. Oak-D stereo already exists
   in `raw/` — if a stereo model is wanted, the baseline is the thing to check first.

---

## 8. Write boundary

Previous sessions were restricted to writing **only** under
`s3://stage-humyn-egocentric-stereo-data/labelling_results/hand_pose_EgoForce/`. For HaWoR the
analogous prefix is `…/labelling_results/hand_pose_HaWoR/`. **Confirm this with the user before
writing anything to S3** — do not assume, and do not write to `viz_delivery/`, `hand_pose_results/`
or any other prefix.

AWS access is via SSO: `aws sso login --profile stage`, then `AWS_PROFILE=stage`. The token expires
roughly daily.
