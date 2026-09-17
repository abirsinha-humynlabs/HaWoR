# Resolved environments for the HaWoR pipeline

`requirements.txt` at the repo root is the upstream declaration. These files are different:
they are the **resolved** environments that the deliveries were actually produced with,
captured with `conda env export` / `pip freeze` on the run machine before it was torn down.
Use them to rebuild rather than resolving from scratch — the CUDA/torch pinning below is the
part that is slow to rediscover.

| file | env | Python | torch |
| --- | --- | --- | --- |
| `hawor.conda.yml` + `hawor.pip.txt` | `hawor` — inference | 3.10.21 | 1.13.0+cu117 |
| `mp310.conda.yml` | `mp310` — MediaPipe handedness (`delivery/mp_handedness.py`) | 3.10 | n/a |

The renderer and scorer do **not** run in `hawor`; they run in the `egoforce` env
(torch 2.8.0+cu126), which is captured in the EgoForce repo under its own `env/`. The delivery
scripts activate both in turn, which is why two incompatible torch builds are expected here and
not a mistake.

## Rebuilding

    conda env create -f env/hawor.conda.yml

`hawor.pip.txt` is a `pip freeze` of the same env, kept because it records the exact commits of
the dependencies installed straight from git — `chumpy@580566ea` in particular. Entries of the
form `pkg @ file:///home/conda/feedstock_root/...` are conda-supplied packages that `pip freeze`
renders as local paths; take those from the conda yml, not from the pip file.

Built and verified on an NVIDIA A10G, driver 615.71.09.
