# Running HaWoR without a GPU

Stock HaWoR cannot run on CPU. This fork adds a CPU path for the parts that are plain PyTorch, and
a way around the part that is not.

## What is and is not CPU-capable

HaWoR is two stacked stages:

| Stage | What it does | CPU? |
| --- | --- | --- |
| Detection + camera-frame hand | YOLO detector, hand ViT, CMIB infiller | **yes**, after the changes below |
| World-space lifting | masked DROID-SLAM + Metric3D | **no** |

DROID-SLAM's `droid_backends` extension has only `.cu` sources — `correlation_kernels.cu`,
`altcorr_kernel.cu` — and no CPU implementation. `droid_slam/motion_filter.py` is also decorated
`@torch.cuda.amp.autocast(enabled=True)` unconditionally. There is no flag that makes it run on CPU
and none can be added without reimplementing the correlation volume.

**So: you cannot get world-space hands out of HaWoR alone on CPU.** You can get camera-frame hands,
and you can get world-space hands if you already have a camera trajectory from somewhere else.

## The useful case: you already have head pose

HaWoR consumes SLAM through exactly one contract, in `lib/eval_utils/custom_utils.load_slam_cam`:

```
pred_cam['traj']   (N, 7) float   [tx, ty, tz, qx, qy, qz, qw]   camera-to-world
pred_cam['scale']  scalar         multiplied into the translation
```

That is all. Anything that produces a camera-to-world trajectory can stand in — including a 6DoF
head pose solved by a separate system. `scripts/scripts_test_video/make_camera_npz.py` writes that
file from an npz holding `T` of shape `(N, 4, 4)`:

```bash
python demo.py --video_path clip.mp4 --img_focal 1065.07 \
    --device cpu --skip-slam --camera-npz head_pose_6dof.npz
```

The conversion is verified against HaWoR's own `load_slam_cam`: feeding a trajectory in and reading
it back recovers `R_c2w` and `t_c2w` to ~2e-7.

## Camera-frame only

Without `--camera-npz` an identity trajectory is written, so HaWoR's "world" equals its camera
frame and the output is camera-space hand pose:

```bash
python demo.py --video_path clip.mp4 --img_focal 1065.07 --device cpu --skip-slam
```

The run prints a warning so this is not mistaken for world-space output.

## New flags

| Flag | Effect |
| --- | --- |
| `--device {auto,cpu,cuda,cuda:N,mps}` | Pins the device. `auto` picks CUDA when available. Also settable as `HAWOR_DEVICE`. |
| `--skip-slam` | Do not run DROID-SLAM. Required on CPU. Writes a stand-in trajectory instead. |
| `--camera-npz PATH` | A 6DoF npz with `T` (N,4,4) camera-to-world, used in place of SLAM. |
| `--no-vis` | Skip the aitviewer render and write the motion only. |

Running on CPU without `--skip-slam` exits immediately with an explanation rather than failing
deep inside a CUDA kernel launch.

## What changed

| File | Change |
| --- | --- |
| `hawor/utils/device.py` | **new.** Resolves the device once: explicit call → `HAWOR_DEVICE` → CUDA if present → CPU. MPS is accepted by name but never auto-selected, because parts of the MANO/pytorch3d path behave differently on Metal. |
| `hawor/utils/process.py` | 4 hard-coded `.cuda()` → `.to(get_device())` in `run_mano` / `run_mano_left`. |
| `infiller/hand_utils/process.py` | Same, 4 sites. |
| `lib/models/hawor.py` | `inference(..., device='cuda')` → `device=None`, resolved at call time. The only caller never passed one, so this was a hard CPU blocker. |
| `scripts/scripts_test_video/hawor_video.py` | 5 `.cuda()` → `.to(device)`; both entry points take `get_device()`; the SLAM-mask block is gated on `HAWOR_SKIP_MASKS`. |
| `scripts/scripts_test_video/make_camera_npz.py` | **new.** Writes a SLAM-compatible trajectory from identity or an external 6DoF npz. |
| `demo.py` | The four flags above; `hawor_slam` import made lazy so a machine without the CUDA extensions can still run `--skip-slam`. |

### A side effect worth knowing

`--skip-slam` also skips the hand-mask render inside `hawor_motion_estimation`. That block exists
*only* to feed masked DROID-SLAM, and it is expensive: a pytorch3d render for every frame plus a
`frames × H × W` bool memmap — **~3.8 GB of scratch per 60 s 1080p clip**, per the project's own
batch script. Skipping SLAM therefore removes a large amount of compute and disk, not just the
SLAM itself.

## Expect it to be slow

The paper quotes 40 ms/frame on a GPU and calls that "far from real-time". On CPU the hand ViT will
be far slower again. Benchmark a short clip before committing to a batch. Nothing here changes the
model's arithmetic — only where it runs.

## Status of these changes

Verified:

- The quaternion conversion in `make_camera_npz.py` round-trips 500 random rotations at 1.1e-15 max
  error, with all four Shepperd branches exercised.
- Both modes round-trip through the real `load_slam_cam` (headpose to ~2e-7, identity to exactly
  R=I, t=0), with `R_w2c @ R_c2w == I`.
- All patched files byte-compile.

**Not verified:** end-to-end inference. That needs the model weights and a working `smplx`, which
would not build on the machine these changes were written on. The device plumbing is mechanical and
the touched lines are listed above, but the first real CPU run may still surface a tensor that lands
on the wrong device. If it does, the error will name the file and line.
