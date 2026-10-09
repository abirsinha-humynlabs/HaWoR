"""Central device selection, so the CPU path is one setting rather than scattered edits.

Stock HaWoR hard-codes ``.cuda()`` in the MANO helpers, the infiller helpers and the video
pipeline. That makes a CPU run impossible even though those parts are plain PyTorch. This module
resolves the device once and everything else asks it.

Resolution order:

1. an explicit ``set_device(...)`` call (what ``demo.py --device`` does)
2. the ``HAWOR_DEVICE`` environment variable
3. CUDA if available, otherwise CPU

MPS is accepted if asked for by name but is NOT auto-selected: several ops in the MANO/pytorch3d
path fall back or differ numerically on Metal, so opting in should be deliberate.

What this does NOT make CPU-capable: DROID-SLAM. Its ``droid_backends`` extension has only ``.cu``
sources and no CPU implementation, so the world-space stage still needs a GPU. Use
``demo.py --skip-slam`` for camera-frame output, or ``--camera-npz`` to supply a trajectory that
was computed elsewhere.
"""

import os

import torch

_DEVICE = None


def _auto():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_device(spec):
    """Pin the device for this process. ``spec`` is 'auto', 'cpu', 'cuda', 'cuda:N' or 'mps'."""
    global _DEVICE
    if spec is None or spec == "auto":
        _DEVICE = _auto()
    else:
        _DEVICE = torch.device(spec)
    if _DEVICE.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("device 'cuda' requested but torch.cuda.is_available() is False")
    if _DEVICE.type == "mps" and not getattr(torch.backends, "mps", None):
        raise SystemExit("device 'mps' requested but this torch has no MPS backend")
    return _DEVICE


def get_device():
    """The device everything should run on."""
    global _DEVICE
    if _DEVICE is None:
        env = os.environ.get("HAWOR_DEVICE")
        _DEVICE = set_device(env) if env else _auto()
    return _DEVICE


def to_device(x):
    """``x.to(get_device())`` for anything with a ``.to``; passes other values through."""
    return x.to(get_device()) if hasattr(x, "to") else x


def is_cpu():
    return get_device().type == "cpu"
