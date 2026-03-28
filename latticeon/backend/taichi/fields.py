from __future__ import annotations

import taichi as ti
from taichi.lang import impl


_initialized_arch: str | None = None


def ensure_taichi_initialized(arch: ti.lang.core.Arch | None = None) -> None:
    global _initialized_arch

    if impl.get_runtime().prog is not None:
        return

    resolved_arch = ti.cpu if arch is None else arch
    ti.init(arch=resolved_arch)
    _initialized_arch = str(resolved_arch)
