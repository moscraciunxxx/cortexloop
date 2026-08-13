from __future__ import annotations

import ctypes
import os
from pathlib import Path

from . import CLASS_NAMES

_LIB = None


class Perception(ctypes.Structure):
    _fields_ = [
        ("class_id", ctypes.c_int),
        ("confidence", ctypes.c_float),
        ("logits", ctypes.c_float * 8),
        ("preprocess_us", ctypes.c_double),
        ("infer_us", ctypes.c_double),
        ("total_us", ctypes.c_double),
    ]


def _candidates() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    names = ["libcortexloop.dylib", "libcortexloop.so", "cortexloop.dll"]
    dirs = [
        root / "build",
        root / "build" / "Release",
        root / "build" / "lib",
        Path(os.environ.get("CORTEXLOOP_LIB", root / "build")),
    ]
    out: list[Path] = []
    for d in dirs:
        for n in names:
            out.append(d / n)
    return out


def load(model_path: str | os.PathLike | None = None):
    global _LIB
    if _LIB is not None:
        return _LIB
    lib_path = None
    for p in _candidates():
        if p.is_file():
            lib_path = p
            break
    if lib_path is None:
        raise FileNotFoundError(
            "Native library not found. Build with: cmake -S . -B build && cmake --build build"
        )
    lib = ctypes.CDLL(str(lib_path))
    lib.cl_init.argtypes = [ctypes.c_char_p]
    lib.cl_init.restype = ctypes.c_int
    lib.cl_shutdown.argtypes = []
    lib.cl_ready.restype = ctypes.c_int
    lib.cl_set_backend.argtypes = [ctypes.c_int]
    lib.cl_set_backend.restype = ctypes.c_int
    lib.cl_get_backend.restype = ctypes.c_int
    lib.cl_backend_name.argtypes = [ctypes.c_int]
    lib.cl_backend_name.restype = ctypes.c_char_p
    lib.cl_backend_available.argtypes = [ctypes.c_int]
    lib.cl_backend_available.restype = ctypes.c_int
    lib.cl_cpu_features.restype = ctypes.c_uint32
    lib.cl_cpu_features_string.restype = ctypes.c_char_p
    lib.cl_class_name.argtypes = [ctypes.c_int]
    lib.cl_class_name.restype = ctypes.c_char_p
    lib.cl_perceive.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int]
    lib.cl_perceive.restype = Perception
    lib.cl_bench_json.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.cl_bench_json.restype = ctypes.c_int
    lib.cl_model_nbytes.restype = ctypes.c_size_t
    lib.cl_version.restype = ctypes.c_char_p

    root = Path(__file__).resolve().parents[1]
    model = Path(model_path) if model_path else root / "models" / "warehouse_int8.bin"
    rc = lib.cl_init(str(model).encode())
    if rc != 0:
        raise RuntimeError(f"cl_init failed ({rc}) for {model}. Train first: python -m cortexloop.train")
    _LIB = lib
    return lib


def perceive(rgb: bytes, width: int, height: int) -> dict:
    lib = load()
    buf = (ctypes.c_uint8 * len(rgb)).from_buffer_copy(rgb)
    p = lib.cl_perceive(buf, width, height)
    return {
        "class_id": int(p.class_id),
        "class_name": CLASS_NAMES[p.class_id] if 0 <= p.class_id < 8 else "unknown",
        "confidence": float(p.confidence),
        "logits": [float(p.logits[i]) for i in range(8)],
        "preprocess_us": float(p.preprocess_us),
        "infer_us": float(p.infer_us),
        "total_us": float(p.total_us),
    }


def set_backend(name: str) -> str:
    lib = load()
    mapping = {"scalar": 0, "neon_dotprod": 1, "neon": 1, "neon_i8mm": 2, "i8mm": 2, "kleidiai": 3}
    key = name.lower().strip()
    if key not in mapping:
        raise ValueError(f"unknown backend {name}")
    rc = lib.cl_set_backend(mapping[key])
    if rc != 0:
        raise RuntimeError(f"backend {name} is not available on this CPU")
    return lib.cl_backend_name(lib.cl_get_backend()).decode()


def backend_name() -> str:
    lib = load()
    return lib.cl_backend_name(lib.cl_get_backend()).decode()


def cpu_features() -> str:
    lib = load()
    return lib.cl_cpu_features_string().decode()


def available_backends() -> list[str]:
    lib = load()
    names = []
    for i in range(4):
        if lib.cl_backend_available(i):
            names.append(lib.cl_backend_name(i).decode())
    return names


def bench_json(m: int = 256, n: int = 256, k: int = 256, iters: int = 24) -> str:
    lib = load()
    buf = ctypes.create_string_buffer(8192)
    lib.cl_bench_json(m, n, k, iters, buf, 8192)
    return buf.value.decode()
