# SETUP.md — set up this repo and find your way around

For a Claude agent (or a person) starting cold. Read `CLAUDE.md` (the user's hard rules), then this file
(setup, repo map, how to run, traps), then `docs/HANDOFF.md` (state of the work, history, open items).
Facts here were checked on 2026-10-19 against the code and the g4dn.xlarge the work ran on.

## 0. Start here

- **Existing instance** (the repo is already at `~/HaWoR`): run `bash ~/HaWoR/setup/setup_instance.sh verify`.
  If it prints `VERIFY_OK`, skip §3–§6. Before launching anything, run `tmux ls` and `nvidia-smi`: one
  clip at a time fits on a T4, so never start a second GPU job.
- **New instance:** §3 → §4 → §5 (needs the user) → §6, then §8.
- **What to work on:** `docs/HANDOFF.md` §1 and §8. As of 2026-10-19 the prod long-clip run is stopped
  at 5 of 10 clips (§8 below says how to resume), and the delivered long-clip npz files have a known
  bug in their `window` key (§9).

## 1. What this repo is

A fork of [HaWoR](https://github.com/ThunderVVV/HaWoR) (egocentric 3D hand reconstruction) plus our
delivery layer. HaWoR's pipeline per video:

```
frames -> hand detector (YOLO, left/right class) -> tracker (ultralytics) -> left/right hand slots
       -> 3D hand model (ViT + temporal modules, on detected frames) -> DROID-SLAM + Metric3D (camera, metric scale)
       -> infiller (fills frames with no detection, from motion only)
```

Our layer ("v3") runs that with the track-ID handedness fix switched on (`HAWOR_TRACK_FIX=1`), then
post-processes: drops infilled rows, smooths keypoints, fixes handedness with a MediaPipe second
opinion, QC, optional render, and uploads.

- Branch: **`my-changes`**. `main` is behind; it only has PR #1 (the track-ID fix, commit `d704d14`).
- Remote: `https://github.com/abirsinha-humynlabs/HaWoR.git` — **public**.

## 2. Rules you must not break (also in `CLAUDE.md`)

- S3 writes go **only** under `s3://stage-humyn-egocentric-stereo-data/labelling_results/hand_pose_HaWoR/v3/`.
  Anything else: ask the user first. The prod bucket (`prod-egc-stereo-v2-data`) is **read-only**.
- Legacy scripts write **outside** `v3/` — never use them to upload:
  - `delivery/run_v3_clip.sh` and `delivery/batch/hawor_tmux.sh` with the `upload` argument
    (→ `hand_pose_HaWoR/<clip>_v3/`);
  - everything in `delivery/v1v2/` (`run_one_clip.sh` and `run_queue.sh` upload unconditionally to
    `hand_pose_HaWoR/<clip>/`).
  Upload only through `delivery/batch/v3_batch.sh` or `delivery/long/long_batch.sh`.
- Never force-push. Never commit weights, videos, MANO `.pkl`, credentials, AWS account ids, SSO start
  URLs, or job lists / CSVs that name customer sites. Keep those in `~/hawor_runs/jobs/`; stage commits
  by explicit path, never `git add -A` / `git add .`.
- Long jobs run in tmux. The user's SSH / VS Code connection drops often.

## 3. Instance

- AMI: **"Deep Learning OSS Nvidia Driver AMI GPU PyTorch (Amazon Linux 2023)"**. A bare AL2023 AMI cost a
  previous session a day on drivers.
- Instance: a GPU **with an NVMe instance store** — g4dn.xlarge (T4 16 GB, 4 vCPU, 16 GB RAM) is what we
  use; g5.xlarge also works.
- Root volume: **100 GB**. The HaWoR stack takes about 17 GB on top of the AMI (conda envs 11 GB, weights
  3.5 GB, thirdparty 1.7 GB, EgoForce 0.5 GB).
- NVMe instance store: mounted at `/opt/dlami/nvme` by `hawor-scratch.service` (installed by setup).
  **Wiped on every stop/start.** Inputs, frames and scratch live there; outputs and state live on root.
- The current box also has an 8 GB `/swapfile` that setup did not create; it has not been shown to be needed.
- `~/miniforge3`, `~/Dyn-HaMR` and `~/dynhamr_dl` on the current box belong to another project. Leave them alone.

## 4. Get the code

```bash
git clone -b my-changes https://github.com/abirsinha-humynlabs/HaWoR.git ~/HaWoR
```

If `main` was cloned by mistake: `git -C ~/HaWoR checkout my-changes`.

## 5. AWS access (SSO only)

The instance role cannot read the buckets. Two SSO profiles are needed. The account ids and the SSO
start URL are deliberately **not** in this repo: ask the user. Template for `~/.aws/config` (chmod 600):

```ini
[profile stage]
sso_session = <stage session name>
sso_account_id = <ask the user>
sso_role_name = HLStagePowerUserAccess
[sso-session <stage session name>]
sso_start_url = <ask the user>
sso_region = us-east-1
sso_registration_scopes = sso:account:access

[profile prod]
sso_session = <prod session name>
sso_account_id = <ask the user>
sso_role_name = HLProdReadonlyAccess
[sso-session <prod session name>]
sso_start_url = <ask the user; same portal as stage>
sso_region = us-east-1
sso_registration_scopes = sso:account:access
```

The user can also create the prod profile in their own terminal: `aws configure sso --profile prod --use-device-code`.

- Log in with `aws sso login --profile stage --use-device-code --no-browser` (same for `prod`). The device
  code expires in about 10 minutes, so have the user run it themselves (`! aws sso login ...`) and approve
  it at once; relaying a code from a background shell has failed twice.
- Tokens expire about daily. Check with `AWS_PROFILE=stage aws sts get-caller-identity` (and `prod`).
- `stage` = read/write (writes still only to `v3/`); `prod` = read-only role.
- The stage bucket is in `ap-south-1` and the prod bucket in `us-east-1`; this matters only for console links.

## 6. Build the environment

```bash
tmux new -d -s setup "bash ~/HaWoR/setup/setup_instance.sh 2>&1 | tee ~/setup_instance.log; bash"
tail -f ~/setup_instance.log                    # ends with: VERIFY_OK
bash ~/HaWoR/setup/setup_instance.sh verify     # re-check any time, e.g. after a stop/start
```

- Needs stage SSO first (MANO comes from S3). Took about 1.5 h on a g4dn.xlarge, mostly compiling
  (pytorch3d is the longest). Per-step logs: `/opt/dlami/nvme/logs/setup_*.log` (lost on stop/start).
- It installs the NVMe boot unit; creates `~/projects/{HaWoR,EgoForce}`, `~/hawor_runs/{out,logs,scripts}`
  (with the `scripts/qc_hawor.py` symlink both drivers need) and the NVMe symlinks
  `~/hawor_runs/{clips,work,scratch,render}`; inits submodules; downloads weights (Hugging Face,
  size-checked) and MANO (S3); installs Miniforge at `~/miniconda3`; and builds three conda envs:
  - `hawor` — Python 3.10, torch 1.13.0+cu117, compiled DROID-SLAM/lietorch, pytorch3d 0.7.5, ultralytics 8.4.150;
  - `mp310` — MediaPipe 0.10.14 (handedness second opinion);
  - `egoforce` — a subset of the EgoForce env, for the renderer and scoring only.
- `VERIFY_OK` means: CUDA kernels ran in `hawor` (lietorch, the DROID correlation kernel, torch_scatter;
  pytorch3d is imported); `mp310` and `egoforce` import cleanly and ffmpeg has libx264; weights and MANO
  files are present; the NVMe is mounted.
- Re-running is safe: finished steps skip themselves. Two exceptions: if a stop/start interrupted a
  `mamba env create`, remove that env first (`~/miniconda3/bin/conda env remove -n <env>`), because a
  half-built env counts as done; and a re-run resets a clean `~/projects/EgoForce` to the tip of
  `origin/run/type2-fusion`. v3 used commit `9edd455`, so read the `NOTE:` line setup prints.
- Rebuild envs only through `setup/setup_instance.sh`. `environment/README.md` predates it and its
  commands no longer work.
- Ad hoc use (inspect an npz, re-run QC): `source ~/miniconda3/etc/profile.d/conda.sh && conda activate hawor`,
  or call `~/miniconda3/envs/hawor/bin/python` directly. A plain login shell has neither on its PATH.

## 7. Repo map

| Path | What it is |
| --- | --- |
| `lib/`, `hawor/`, `infiller/`, `scripts/scripts_test_video/` | Upstream HaWoR. `lib/pipeline/tools.py` also holds our `HAWOR_TRACK_FIX` path (`d704d14`). |
| `thirdparty/` | DROID-SLAM (we added `sm_75` for the T4) and Metric3D (bf16 → fp16 fix for the T4). |
| `delivery/run_hawor_clip.py` | Runs HaWoR's 4 stages on one video; writes the hand21 npz in camera frame with the true K. |
| `delivery/postprocess_v3.py` | Drops infilled rows, smooths 3D keypoints (±4 frames, robust), re-projects 2D. |
| `delivery/mp_handedness.py`, `fix_handedness.py` | MediaPipe second opinion and the per-track handedness vote. |
| `delivery/qc_hawor.py` | QC report and sample frames. |
| `delivery/render_v4_panels_handfix.py` | Panel video renderer (ZED clips only so far). |
| `delivery/run_v3_clip.sh` | One ~60 s ZED clip end to end (6 stages). |
| `delivery/batch/` | ZED batch: `v3_jobs*.tsv`, `v3_prefetch.sh`, `v3_batch.sh`, `hawor_tmux.sh`. |
| `delivery/long/` | Long prod clips (akai, bitrobot): `make_jobs.sh`, `long_batch.sh`, `run_v3_long.sh`, `merge_windows.py`, `pp_correct.py`, `fix_uploaded.sh`. |
| `delivery/v1v2/` | History only. Never run (see §2). |
| `setup/`, `environment/` | Instance setup, and the env captures it builds from. |
| `docs/HANDOFF.md`, `docs/history/` | State of the work, decisions, open items, batch logs. |

`~/hawor_runs/jobs/` is made by hand; `~/hawor_runs/v3_state/` and `long_state/` appear on a batch's first use.

## 8. Running

**One ZED clip (~60 s):** `~/HaWoR/delivery/batch/hawor_tmux.sh <clip> "<group>"` → `~/hawor_runs/out/<clip>_v3/`.
- `<group>` = column 2 of `delivery/batch/v3_jobs*.tsv` without `validation-result/ZED/` and `/<clip>`, e.g.
  `home`, `hospitality`, or `industrial/industrial noida` (quote it: it can contain a slash and a space).
- Needs a valid stage SSO token even without uploading. Never pass `upload` (§2).
- It finds the head pose by name under `labelling_results/6dof_head_pose_v2/` and fails for a few clips
  (truncated folder names, or a head pose under `6dof_head_pose_v3`). For clips in a job list, prefer the
  batch below, whose prefetch stages the exact files.

**ZED batch → `v3/<input path below validation-result/>/`** (e.g. `v3/ZED/home/episode_047/`):
```bash
JOBS=~/HaWoR/delivery/batch/<jobs>.tsv ~/HaWoR/delivery/batch/v3_prefetch.sh      # inputs to NVMe
tmux new -d -s v3batch "JOBS=$HOME/HaWoR/delivery/batch/<jobs>.tsv $HOME/HaWoR/delivery/batch/v3_batch.sh 2>&1 | tee -a $HOME/hawor_runs/logs/v3batch/batch.log; bash"
JOBS=<same tsv> ~/HaWoR/delivery/batch/v3_batch.sh --upload-only                   # after an SSO expiry
```
About 45–55 min per 60 s 1080p clip on a T4 (including the render).

**Long prod clips (akai / bitrobot) → `v3/<path below chunking/>/` in the stage bucket**
(prod `chunking/akai/2026-07-22/<session>/004/seg_000/` → `v3/akai/2026-07-22/<session>/004/seg_000/`):
```bash
AWS_PROFILE=prod aws sts get-caller-identity                                       # make_jobs needs prod SSO
mkdir -p ~/hawor_runs/jobs
~/HaWoR/delivery/long/make_jobs.sh <clips.csv> > ~/hawor_runs/jobs/<name>.tsv
grep -vc '^#' ~/hawor_runs/jobs/<name>.tsv                                         # must equal the CSV's row count
tmux new -d -s hawor_long "JOBS=$HOME/hawor_runs/jobs/<name>.tsv $HOME/HaWoR/delivery/long/long_batch.sh 2>&1 | tee -a $HOME/hawor_runs/logs/long_batch.log; bash"
tail -f ~/hawor_runs/logs/long_batch.log
JOBS=<same tsv> ~/HaWoR/delivery/long/long_batch.sh --upload-only                 # retry uploads after a stage SSO expiry
```
- The CSV needs a header row and the clip name in column 1. A clip name `a__b__c` maps to `chunking/a/b/c/`.
  `make_jobs.sh` silently drops clips whose sources it cannot list, so an expired prod token yields an
  empty job list.
- Inputs fetched per clip: `left_rectified.mp4`, `calibration.json` and `meta.json` (30 fps, 1920×1080).
  If any fetch fails — an expired prod token or a missing file — the batch waits and retries every
  10 minutes.
- Each clip runs as windows of at most 1800 frames (in practice ~1,560–1,690 frames, about 55 s) with
  30-frame margins; post-processing runs once on the merged clip. Keypoints only, no render yet; no head
  pose exists for these sources.
- Cost on a T4: 14–27 min per window, so about 100–165 min per 5-minute clip depending on how often
  hands are in view. One clip at a time: a window peaks near 13 GB GPU and 12 GB host RAM.
- State: `~/hawor_runs/long_state/<clip>.{done,uploaded,failed}`. A failed clip is skipped on reruns;
  delete its `.failed` file to retry. If stage SSO expires, finished clips are marked done and their
  uploads are retried before each next clip.
- The job list is read once at start. To change it, stop the batch right after a `wNNN … min` line in
  `~/hawor_runs/logs/long/<clip>.log` and restart; finished windows stay cached on NVMe until the clip
  is uploaded.

**Resuming the stopped prod run (5 of 10 left):**
- On the current box: start the tmux command above with `JOBS=$HOME/hawor_runs/jobs/prod10.tsv`;
  `long_state/` skips the 5 finished clips.
- On a new box: the job list, the clips CSV and `long_state/` are not in git. Get the TSV/CSV from the
  user, then mark the 5 finished clips (listed in `docs/HANDOFF.md` §8 by job-list position) with
  `touch ~/hawor_runs/long_state/<clip>.done ~/hawor_runs/long_state/<clip>.uploaded`, or remove them
  from the TSV. Otherwise they re-run and re-upload, about 10 h of T4 time.

**Stopping a batch safely:**
```bash
PP=$(tmux list-panes -t <session> -F '#{pane_pid}'); pstree -p $PP   # note the PIDs first
tmux kill-session -t <session>
kill <the noted PIDs>                                                # if any are still alive
```
Never use `pkill -f <pattern>`: it also matches your own shell and the tmux server whenever the pattern
appears in their command lines. Stopping mid-window throws that window's work away.

## 9. Outputs

Long prod clips, 13 files per S3 folder:

| File | Contents |
| --- | --- |
| `<clip>_v3_hand21_keypoints.npz` | The deliverable (keys below). |
| `<clip>_v3_3d_keypoints.npz` | Coverage denominator only (`processed_frames`, `frame_idx`, `K`, …). It holds **no** 3D keypoints, whatever older notes say. |
| `<clip>_v3_v3_report.json` | Post-processing report: rows dropped as infilled, smoothing stats. |
| `<clip>_v3_handedness_report.json` | MediaPipe vote: matched rows, tracks and rows flipped, with frame ranges. |
| `<clip>_v3_hawor_metadata.json` | Per-window run metadata, incl. each window's frame range and the world-to-camera round-trip check. |
| `qc/` (8 files) | `<clip>_v3_qc.json` and 7 sample frames with keypoints drawn. |

ZED clips, 16 files: the hand21 and 3d npz, `<clip>_v3_hawor_wrist_traj_panels.mp4`, the v3 and
handedness reports, `head_pose_6dof_smoothed.npz`, `qc/` (8 files) and `eval/` (`report.json`,
`report.md`). ZED folders have no `_hawor_metadata.json`.

Keys of `<clip>_v3_hand21_keypoints.npz` (one row per hand per frame, OpenPose-21 joint order):
- `frame_idx`; `hand` (0 = left, 1 = right; the final label); `kp2d` (px, true K); `kp3d_cam` (m, camera
  frame); `kp2d_raw` / `kp3d_cam_raw` (before smoothing); `depth_measured` (all True in v3); `smoothed`.
- `is_right_wilor` and `is_right_mp` are overwritten to equal `hand`. `source` is always `'wilor'`, which
  is a leftover name, not provenance.
- Per clip: `K`, `width`, `height`, `fps`, `step`. Long clips add `pp_offset_px` and `pp_corrected`.
- **Known bug:** in long-clip files `window` still has its length from before the infilled rows were
  dropped, so it is not aligned to rows. Derive a row's window from `frame_idx` and the window ranges in
  `_hawor_metadata.json`. (Cause: `postprocess_v3.py` filters a fixed key list that lacks `window`.)
- `(frame_idx, hand)` is not unique: the MediaPipe vote can leave two same-side hands in a frame (712
  frames over the 5 prod clips). Pair hands by 2D wrist position when comparing models.

The QC gate (reprojection median < 1 px) is ~0 by construction, because `kp2d` is defined as the
projection of `kp3d_cam`. It catches broken plumbing, not inaccuracy.

## 10. Traps (verified 2026-10-19)

- HaWoR's 3D model runs only on frames where the hand has a box: a box the tracker kept, or any box in a
  frame where the tracker returned nothing (those get fallback ids 5000/10000). Box interpolation in
  `hawor_video.py` never fires on video, despite comments in `run_hawor_clip.py` and older notes. Frames
  without a box are filled by the infiller from motion only; v3 drops those rows.
- The tracker is ultralytics' default, **TrackTrack** (ultralytics 8.4.150): a new track needs a box
  scoring above 0.7 and counts after 3 frames; boxes at 0.25–0.6 only extend live tracks; boxes below 0.25
  never enter it; in any frame with a live track every untracked box is discarded. The detector's own
  threshold (0.2) matters far less.
- HaWoR assumes the principal point is the image centre. `delivery/long/pp_correct.py` corrects the
  delivered long-clip keypoints (akai cameras are up to (+47, −25) px off-centre; ZED and bitrobot under
  2 px). The ZED pipeline does not apply it.
- `fix_handedness.py` overwrites the original labels; they survive only as flip ranges in the report.
- There is no ground truth on our footage; every accuracy statement so far is a proxy.

## 11. Committing

- This box has no git identity configured. Commit with the user's name and email (ask; their earlier
  commits show them): `git -c user.name='<name>' -c user.email='<email>' commit ...`.
- Stage explicit paths and check `git diff --cached --name-only` before committing. An untracked
  `hand21kp_rendered_clips_table.csv` (customer clip list) sits in the repo root on the current box and
  must never be committed.
- `git push origin my-changes` works from the session the user connects through. Never force-push.
