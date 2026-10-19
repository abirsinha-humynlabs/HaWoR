# HaWoR — handoff (state as of 2026-10-12)

Read this before doing anything on a new instance. It supersedes `docs/history/HaWoR_HANDOFF_2026-10-07.md`
(kept for history; several of its claims were corrected below).

## 1. Where things stand — the last thing we did

**The v3 pipeline was run on all 15 videos that had ever been through HaWoR, and published.**
Batch ran 2026-10-09 08:50 → 19:32 UTC on a g4dn.xlarge (T4), 15/15 succeeded, 0 failed, every clip passed the
QC gate (reprojection 0.0000 px, 0 % joints behind the lens). Each upload was size-verified against S3.

Output: `s3://stage-humyn-egocentric-stereo-data/labelling_results/hand_pose_HaWoR/v3/<input path>/`, mirroring the
input below `validation-result/` (e.g. `…/v3/ZED/home/episode_047/`; moved there from `v3/<clip>/` on 2026-10-18,
keys and sizes verified) — 16 files per clip
(241 objects, 1.16 GB). Run logs + per-clip timings: `docs/history/runs/2026-10-09_v3_batch/` in this repo.

| clip | run min | rows kept | bone CV median |
|---|---|---|---|
| episode_001 | 51.2 | 3621 | 0.0413 |
| episode_002 | 62.9 | 3245 | 0.0538 |
| episode_009 | 53.6 | 2159 | 0.0519 |
| episode_046 (31 s) | 24.9 | 1756 | 0.0299 |
| episode_047 | (reused, see below) | 2763 | 0.0608 |
| episode_048 | 46.1 | 3628 | 0.0384 |
| episode_052 | 47.3 | 3424 | 0.0413 |
| Hospitality_plating_salad_vegetables_64 | 40.8 | 2404 | 0.0329 |
| Hospitality_pouring_pink_yogurt_into_bowls_73 | 47.5 | 3606 | 0.0322 |
| Hospitality_pouring_yogurrt_into_bowls_72 | 47.9 | 3631 | 0.0326 |
| Others_blowing_190 | 42.1 | 1788 | 0.0325 |
| construction_cutting_ply_using_machine_516 | 48.3 | 3639 | **0.0047** ← see §8 |
| construction_mixing_cement_using_equipment_cement_chemicali_mixing_503 | 42.0 | 1765 | 0.0424 |
| construction_removing_grooting_using_hand_blade_506 | 35.7 | 1815 | 0.0297 |
| construction_sanding_pipe_507 | 44.8 | 3617 | 0.0288 |

episode_047 was not re-run in the batch: its v3 output is the 2026-10-07 run on the same box/env, which was
parity-checked against the originally published `hand_pose_HaWoR/episode_047_v3/` (final keypoints 0.2 mm median /
2.1 mm p95 3D; identical handedness flip; bit-identical head-pose smoothing).

Inputs: `validation-result/ZED/<group>/<clip>/left_eye.mp4` (+ `calibration.json`, `imu_accel.csv`) and the head
pose from `labelling_results/6dof_head_pose_v2/…`; exact keys per clip in `delivery/batch/v3_jobs.tsv`.

## 2. What "v3" is

HaWoR's model, weights and inference are unchanged. On top (all on branch `my-changes`, `delivery/README.md`):

- **v2 — `HAWOR_TRACK_FIX=1`** (`lib/pipeline/tools.py`): stock `detect_track` keeps the first box labelled
  "right" and the first labelled "left" per frame, so when the detector labels both hands "right" the real right
  hand is discarded and the infiller fabricates that frame (13 % of frames on episode_048). The fix votes
  handedness once per tracker id, then keeps ≤1 box per side per frame by track, ties broken on confidence.
- **v3 — `delivery/postprocess_v3.py`**: drop infiller rows (`depth_measured=False`, no image evidence); zero-phase
  local-quadratic smoothing of 3D keypoints (±4 frames, robust, per track), 2D re-projected from the smoothed 3D;
  SE(3) head-pose smoothing (±3). Raw values kept as `kp2d_raw` / `kp3d_cam_raw`. No bone rigidification.
- **v3 — handedness**: `mp_handedness.py` (MediaPipe as an independent opinion, labels swapped because MediaPipe
  assumes a mirrored image) + `fix_handedness.py` (per-physical-hand vote → `npz['hand']`);
  `render_v4_panels_handfix.py` honours `npz['hand']` instead of re-deriving L/R from wrist x.
- Driver: `delivery/run_v3_clip.sh <clip> <group> [upload] [--fresh]` (6 stages; 3 conda envs).

Trade-offs: v3 is not raw model output; coverage drops (fabricated rows removed); bone CV rises slightly from
per-coordinate smoothing; agreement with an independent 2D model (M3 proxy) is unchanged. No ground-truth accuracy
has ever been measured on our footage.

## 3. Repositories

| repo | branch / commit | notes |
|---|---|---|
| `abirsinha-humynlabs/HaWoR` | **`my-changes`** | everything lives here. `main` is behind (only has PR #1); merging `my-changes` → `main` is still an open decision |
| `abirsinha-humynlabs/EgoForce` | `run/type2-fusion` @ `9edd455` | used read-only: `fusion/` (calibration, topology, evaluate_run) and `viz_delivery/` (render_v2/v3) |
| `abirsinha-humynlabs/hand21kp` | private | referenced in old notes; not needed to run v3 |

Changes made 2026-10-07…12 on `my-changes` (this commit): two T4 compatibility fixes, batch tooling, setup
script, these docs.
- `thirdparty/DROID-SLAM/setup.py`: add `sm_75` gencode (the branch only built `sm_86`; T4 is sm_75).
- `thirdparty/Metric3D/.../RAFTDepthNormalDPTDecoder5.py`: `bfloat16` → `float16` inside an autocast with
  `enabled=False` — torch 1.13 rejects bf16 at construction on GPUs without bf16 (T4). The dtype is never used
  when disabled, so numerics are unchanged (verified by the parity check).

## 4. Setting up a new instance

1. Launch **"Deep Learning OSS Nvidia Driver AMI GPU PyTorch (Amazon Linux 2023)"** on a GPU instance **with an NVMe
   instance store** (g4dn.xlarge was used: T4 16 GB, 4 vCPU, 16 GB RAM; g5.xlarge also fine). Root ≥ 50 GB.
   A bare AL2023 AMI cost a prior session a day on drivers — use the DLAMI.
2. `git clone -b my-changes https://github.com/abirsinha-humynlabs/HaWoR.git ~/HaWoR`
3. AWS SSO profile `stage` (ask the user for the start URL and account id — deliberately not in this public repo):
   `aws configure sso --profile stage` with SSO session name `HLStagePowerUserAccess`, region `us-east-1`, role
   `HLStagePowerUserAccess`; then `aws sso login --profile stage --use-device-code`. The token expires ~daily.
   The instance role cannot read the bucket; SSO is required.
4. `bash ~/HaWoR/setup/setup_instance.sh` — idempotent, ~45–60 min on a g4dn.xlarge (mostly compiling). It:
   installs the boot unit that mounts the instance store at `/opt/dlami/nvme`; builds `~/projects/{HaWoR,EgoForce}`
   and `~/hawor_runs`; inits submodules; downloads weights from Hugging Face (size-checked); fetches MANO from
   `s3://…/labelling_results/implementation/mano/`; installs Miniforge at `~/miniconda3` and ninja; builds the
   three envs; ends with `VERIFY_OK` after real CUDA-kernel checks. `bash setup/setup_instance.sh verify` re-checks.

Environments (all exact pins from captured files, except as noted):
- `hawor` — py 3.10.21, torch 1.13.0+cu117, nvcc 11.7 from conda (`environment/hawor.conda.yml` + `hawor.pip.txt`).
  Compiled locally: chumpy@580566ea, torch-scatter 2.1.2, droid_backends + lietorch (arch 7.5 + 8.6), pytorch3d
  v0.7.5 (the prebuilt wheel host now returns 403). All 131 pip packages match the capture.
- `mp310` — MediaPipe 0.10.14 (`environment/mp310.conda.yml`).
- `egoforce` — **subset only**: what `run_v3_clip.sh` stage 6 needs (numpy 2.2.6, opencv 4.12.0.88, matplotlib
  3.10.6, scipy 1.15.3, ffmpeg 6.1.1 gpl), exact versions from EgoForce `environment/egoforce.*`. The full EgoForce
  env (CUDA 12.6 + torch 2.8 + TensorRT, ~12–15 GB) does not fit a 50 GB root alongside the rest; needed only to
  run EgoForce itself or the renderer's `--clip-foot`.

The old S3 `_env_snapshot/hawor_env_snapshot.tgz` is a *different* env (torch 2.8+cu128 venv, Sep-16 fleet batch,
driver `run_clip.py` never saved). It did not produce v3 and is not used.

## 5. Running

- One clip, detached (survives SSH/VS Code drops): `~/HaWoR/delivery/batch/hawor_tmux.sh <clip> <group> [upload]`
  (`upload` here means `run_v3_clip.sh`'s own upload to `hand_pose_HaWoR/<clip>_v3/` — prefer the batch runner).
- A list of clips → `v3/<input path>/` (e.g. `v3/ZED/home/episode_047/`): `JOBS=<tsv> ~/HaWoR/delivery/batch/v3_prefetch.sh` (pulls all inputs to NVMe so
  an SSO expiry cannot stop inference), then in tmux: `JOBS=<tsv> ~/HaWoR/delivery/batch/v3_batch.sh`.
  Resumable; per-clip state in `~/hawor_runs/v3_state/`; logs in `~/hawor_runs/logs/v3batch/`. If SSO expired
  mid-run: log in, then `v3_batch.sh --upload-only`. Note `v3_state/` marks clips uploaded — for a new list on a
  new box it starts empty.
- Cost on a T4: ~45–55 min per 60 s 1080p clip (detect 3, motion estimation 13–18, SLAM 3–8, post 2,
  render ~18). One clip at a time: SLAM peaks at 13.4 of 15 GB GPU and motion estimation at ~12 GB host RAM.
  ~3.8 GB scratch per clip on NVMe (the batch runner deletes it after upload).
- **Long clips (minutes) from the prod bucket** — `delivery/long/` (added 2026-10-18), keypoints only, no render yet:
  `make_jobs.sh <clips.csv> > jobs.tsv` (clip `a__b__c` → source `chunking/a/b/c/`, output `v3/a/b/c/`; checks every
  source exists), then in tmux `JOBS=<tsv> delivery/long/long_batch.sh` (`--upload-only` to retry uploads). Reads prod
  with AWS profile `prod` (read-only role), writes only to the stage `v3/` prefix. `run_v3_long.sh` extracts frames
  once with HaWoR's own ffmpeg command, runs `run_hawor_clip.py` unchanged on ~60 s windows (1800 frames, 30-frame
  overlap each side, only window cores kept — `merge_windows.py`), then post-process / MediaPipe / handedness vote /
  `pp_correct.py` / QC once on the merged clip. No head pose exists for these sources. State in
  `~/hawor_runs/long_state/`, logs in `~/hawor_runs/logs/long/`; job lists live in `~/hawor_runs/jobs/` (not in git —
  they name customer sites). T4 cost: ~14–18 min per window; 48–164 min per 5–5.5 min clip depending on how often
  hands are in view.
- **Principal point**: HaWoR places hands with a camera centred on the image (`hawor_video.py:96`); only the focal
  comes from the calibration. `pp_correct.py` moves kp3d/kp2d onto the true principal point and stores
  `pp_offset_px`. Offsets: ZED and bitrobot < 2 px (the published ZED v3 is unaffected), akai up to (+47, −25) px.
  `fix_uploaded.sh` applies it to a clip uploaded without it.

## 6. Provenance facts (settled)

- The two commit hashes in old notes are both HaWoR-fork commits: `66c7d41` (last upstream) stamps the Sep 14–15
  delivery runs v1/v2/v3, which ran the track fix as *uncommitted* edits; `883479d` (merge of PR #1) stamps the Sep 16
  fleet batch (torch 2.8 venv, `track_fix: true`, raw only).
- Before this work only **episode_047** had ever been through v3.
- Old S3 folders `hand_pose_HaWoR/episode_048/` and `episode_002/` mix runs: npz + metadata are from the Sep 16 fleet
  batch, eval/qc/panels beside them are from Sep 14 v1. Bucket versioning is off, so the v1 npz are gone.
- Fleet vs delivery env on the same clip/code (A10G): 2.3 mm median / 1.4 cm p95 joint difference — the run-to-run
  scale to expect. Infilled rows (no image evidence) differ by up to ~40 cm between runs; v3 drops them.

## 7. Hard constraints (from the user)

- S3 writes **only** to `s3://stage-humyn-egocentric-stereo-data/labelling_results/hand_pose_HaWoR/v3/`
  (confirmed 2026-10-09). Anything else — confirm with the user first. Never `viz_delivery/`, `hand_pose_results/`.
- Never force-push. Never commit model weights, videos, MANO files or credentials (this repo is public — also keep
  AWS account ids / SSO URLs out).
- Long runs go in tmux.

## 8. Open items / next steps

1. **construction_cutting_ply_using_machine_516** has bone CV 0.0047 and jitter 1–2 px, ~10× steadier than every
   other clip; anatomy is normal and both hands are present ~100 %. Probably hands held still on the machine — check
   its panel video before trusting it.
2. **14 clips the fleet planned but never ran** — ready job list: `delivery/batch/v3_jobs_next14.tsv`.
3. **Same-level comparison** vs the incumbent (WiLoR + MediaPipe + ADAPT smoothing): v3 vs incumbent on the same
   clips, keyed on `(frame, hand)` — never done; HaWoR's earlier "jitter disadvantage" compared raw vs smoothed.
4. Merge `my-changes` → `main` (PR) so a default clone gets the pipeline.
5. Ground truth: every accuracy statement so far is a proxy.
6. Full `egoforce` env would need a larger root volume.
7. **Prod long-clip run** (2026-10-18): 10 clips (5 akai + 5 bitrobot, `~/hawor_runs/jobs/prod10.tsv`) of the 117 in
   the user's `hand21kp_rendered_clips_table.csv`, to tally HaWoR against the other models' npz in prod
   `work_items/hand_detection/model_output_30fps/<clip>/`. Compare hands by position, not only `(frame, hand)`: the
   other model labels many hands 2 (= other / bystander) where HaWoR always says L or R. Render comes later.

## 9. What is NOT preserved

- Raw stage-1 intermediates of the 2026-10-09 batch (all-row raw npz incl. infilled rows, MediaPipe npz) lived on the
  NVMe instance store and were erased by the 2026-10-12 stop/start. Not part of any delivery; regenerate by
  re-running a clip. The published v3 npz keeps raw values for every delivered row, and `<clip>_v3_3d_keypoints.npz`
  is the raw HaWoR 3D output.
- Run logs/timings are preserved in git: `docs/history/runs/2026-10-09_v3_batch/` (batch.log, per-clip logs, timing.tsv).
