import numpy as np
import ctypes
import os
import time

# Argument Types
wxb_argtypes = [
    ctypes.POINTER(ctypes.c_float),   # W
    ctypes.POINTER(ctypes.c_float),   # X
    ctypes.POINTER(ctypes.c_float),   # b
    ctypes.POINTER(ctypes.c_float),   # Y
    ctypes.c_int, ctypes.c_int, ctypes.c_int,  # out_features, in_features, batch_size
    ctypes.POINTER(ctypes.c_double)   # time_ms
]

# ----- Load CPU library -----
libcpu = ctypes.CDLL(os.path.abspath("./libwxb_cpu.so"))
libcpu.wxb_cpu.argtypes = wxb_argtypes
# ---- Load CPU Libary (C) ----
libcpuc = ctypes.CDLL(os.path.abspath("./libwxb_cpuc.so"))
libcpuc.wxb_cpuc.argtypes = wxb_argtypes 

# ----- Load GPU library -----
libgpu = ctypes.CDLL(os.path.abspath("./libwxb_gpu.so"))
libgpu.wxb_gpu.argtypes = [
    ctypes.POINTER(ctypes.c_float),   # W
    ctypes.POINTER(ctypes.c_float),   # X
    ctypes.POINTER(ctypes.c_float),   # b
    ctypes.POINTER(ctypes.c_float),   # Y
    ctypes.c_int, ctypes.c_int, ctypes.c_int,  # out_features, in_features, batch_size
    ctypes.POINTER(ctypes.c_float)   # time_ms
]


def wxb_cpu(W, X, b):
    out_features, in_features = W.shape
    batch_size = X.shape[0]
    Y = np.zeros((batch_size, out_features), dtype=np.float32)
    time_ms = ctypes.c_double()
    libcpu.wxb_cpu(
        W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        Y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_features, in_features, batch_size,
        ctypes.byref(time_ms)
    )
    return Y, time_ms.value


def wxb_gpu(W, X, b):
    out_features, in_features = W.shape
    batch_size = X.shape[0]
    Y = np.zeros((batch_size, out_features), dtype=np.float32)
    time_ms = ctypes.c_float()
    libgpu.wxb_gpu(
        W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        Y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_features, in_features, batch_size,
        ctypes.byref(time_ms)
    )
    return Y, time_ms.value


if __name__ == "__main__":
    # Example sizes
    out_features = 10
    in_features = 5
    batch_size = [1, 10, 100, 1000, 10000, 50000, 100_000, 1_000_000, 5_000_000, 10_000_000, 1_000_000_000]
    for i in batch_size:
        start = time.time()

        # Random data
        W = np.random.rand(out_features, in_features).astype(np.float32)
        X = np.random.rand(i, in_features).astype(np.float32)
        b = np.random.rand(out_features).astype(np.float32)

        print("\nW:", len(W), len(W[0]))
        print("X:", len(X), len(X[0]))
        print("b:", len(b))

        # CPU
        Y_cpu, t_cpu = wxb_cpu(W, X, b)
        print(f"\nCPU [C++] wx+b took {t_cpu:.6f} ms")
        # print("CPU Output:\n", Y_cpu[0][0])

        # CPU in C
        Y_cpuc, t_cpuc = wxb_cpu(W, X, b)
        print(f"CPU [C] wx+b took {t_cpuc:.6f} ms")
        # print("CPU Output:\n", Y_cpuc[0][0])

        # GPU
        Y_gpu, t_gpu = wxb_gpu(W, X, b)
        print(f"GPU wx+b took {t_gpu:.6f} ms")
        # print("GPU Output:\n", Y_gpu[0][0])

        diff1 = np.max(np.abs(Y_cpu - Y_gpu))
        print(f"\nMax Difference CPU [C++] vs GPU: {diff1}")

        diff2 = np.max(np.abs(Y_cpuc - Y_gpu))
        print(f"Max Difference CPU [C] vs GPU: {diff2}")

        diff3 = np.max(np.abs(Y_cpu - Y_cpuc))
        print(f"Max Difference CPU [c++] vs CPU [C]: {diff3}")

        end = time.time()
        print("\nTime taken at the orchestrator level(Python):", end - start, " s")
