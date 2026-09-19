"""macOS 兼容：XGBoost wheel 通过 @rpath 找 libomp.dylib，在部分环境
（pyenv Python + Apple Silicon）rpath 解析不到，需在 import xgboost 前
把 Homebrew 的 libomp 预加载进进程，dyld 随后会复用已加载镜像。

导入本模块即完成预加载；非 macOS 或无库时静默跳过。
"""

import ctypes
import sys
from pathlib import Path

_CANDIDATES = (
    "/opt/homebrew/opt/libomp/lib/libomp.dylib",  # Apple Silicon Homebrew
    "/usr/local/opt/libomp/lib/libomp.dylib",  # Intel Homebrew
)


def preload_libomp() -> bool:
    if sys.platform != "darwin":
        return False
    for path in _CANDIDATES:
        if Path(path).exists():
            ctypes.CDLL(path)
            return True
    return False
