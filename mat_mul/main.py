import numpy as np
import ctypes
import os
import time

cpu_argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double)
]

# Load CPU shared library
libcpu = ctypes.CDLL(os.path.abspath("./matmul_cpu.so"))
libcpu.matmul_cpu.argtypes = cpu_argtypes

libcpuc = ctypes.CDLL(os.path.abspath('./matmul_cpuc.so'))
libcpuc.matmul_cpuc.argtypes = cpu_argtypes

# Load GPU shared library
libgpu = ctypes.CDLL(os.path.abspath("./matmul_gpu.so"))
libgpu.matmul_gpu.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float)
]

def matmul_cpu(A, B):
    rowsA, colsA = A.shape
    rowsB, colsB = B.shape
    assert colsA == rowsB
    C = np.zeros((rowsA, colsB), dtype=np.float32)
    time_ms = ctypes.c_double()

    libcpu.matmul_cpu(
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        rowsA, colsA, colsB,
        ctypes.byref(time_ms)
    )
    return C, time_ms.value

def matmul_gpu(A, B):
    rowsA, colsA = A.shape
    rowsB, colsB = B.shape
    assert colsA == rowsB
    C = np.zeros((rowsA, colsB), dtype=np.float32)
    time_ms = ctypes.c_float()

    libgpu.matmul_gpu(
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        rowsA, colsA, colsB,
        ctypes.byref(time_ms)
    )
    return C, time_ms.value

if __name__ == "__main__":
    N = [16, 32, 64, 128, 256, 512, 1000, 1024, 2000, 2048]
    for i in N:
        start = time.time()
        A = np.random.rand(i, i).astype(np.float32)
        B = np.random.rand(i, i).astype(np.float32)

        # print("Matrix A:\n",A, "\n\n","Matrix B: \n", B, "\n")

        C_cpu, time_cpu = matmul_cpu(A, B)
        print(f"\nN = {i}\nCPU Matrix [c++] Multiplication took {time_cpu:.8f} ms")

        C_cpuc, time_cpuc = matmul_cpu(A, B)
        print(f"CPU Matrix [c] Multiplication took {time_cpuc:.8f} ms")
        
        C_gpu, time_gpu = matmul_gpu(A, B)
        print(f"GPU Matrix Multiplication took {time_gpu:.8f} ms")

        print(f"Difference CPU vs GPU: {np.max(np.abs(C_cpu - C_gpu))}")
        end = time.time()

        print("Time Taken at the orchestrator level (Python):", end - start, " s")
