# HaWoR fork — notes for Claude

Read `SETUP.md` first (setup on a new instance, repo map, how to run, traps), then `docs/HANDOFF.md` (current state,
what was done last, open items).

## Branch
Work on **`my-changes`**. `main` is behind and lacks the v3 pipeline (`delivery/`), the batch tooling and the setup.

## Hard constraints (user's rules)
- S3 writes only to `s3://stage-humyn-egocentric-stereo-data/labelling_results/hand_pose_HaWoR/v3/`.
  Any other prefix: ask the user first. Reads elsewhere in the bucket are fine.
- Never force-push. Never commit weights, videos, MANO `.pkl`, credentials. This repo is **public**: keep AWS account
  ids and SSO start URLs out of it. Never commit job lists or CSVs that name customer sites (keep them in
  `~/hawor_runs/jobs/`); stage commits by explicit path.
- Prod bucket `prod-egc-stereo-v2-data`: read-only, via SSO profile `prod` (needed for long clips).
- Never pass `upload` to `run_v3_clip.sh` / `hawor_tmux.sh`, and never run `delivery/v1v2/`: they write outside `v3/`.
- AWS is SSO only (`aws sso login --profile stage --use-device-code`, expires ~daily); the instance role cannot read
  the bucket. If the `stage` profile is missing, ask the user for the SSO start URL / account id.
- Long-running jobs go in tmux (`delivery/batch/hawor_tmux.sh`, or `tmux new -d -s ...`).

## New instance
GPU DLAMI (OSS Nvidia Driver, PyTorch, Amazon Linux 2023) with an NVMe instance store, repo at `~/HaWoR` on
`my-changes`, SSO logged in, then `bash setup/setup_instance.sh` (idempotent, ~1.5 h, ends in `VERIFY_OK`).

## Layout after setup
- `~/projects/HaWoR` → this repo; `~/projects/EgoForce` (branch `run/type2-fusion`) — `delivery/` scripts hardcode both.
- `~/miniconda3` envs: `hawor` (inference, torch 1.13+cu117), `mp310` (MediaPipe), `egoforce` (render/score subset).
- `~/hawor_runs/{out,logs,scripts}` on root; `~/hawor_runs/{clips,work,scratch,render}` → `/opt/dlami/nvme` (wiped on
  every stop/start; `hawor-scratch.service` re-mounts it).

## Run
- One clip: `delivery/batch/hawor_tmux.sh <clip> <group>` → `~/hawor_runs/out/<clip>_v3/`.
- Batch → `v3/<input path below validation-result/>/` (e.g. `v3/ZED/home/episode_047/`): `JOBS=<tsv> delivery/batch/v3_prefetch.sh`, then `JOBS=<tsv> delivery/batch/v3_batch.sh` in tmux.
- Long prod clips (akai/bitrobot) → `v3/<path below chunking/>/`: `delivery/long/make_jobs.sh`, then `JOBS=<tsv> delivery/long/long_batch.sh` in tmux (HANDOFF §5).
