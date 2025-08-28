import ctypes
import numpy as np
import time

cpu_lib = ctypes.CDLL("./libadd_cpu1.so")
cpuc_lib = ctypes.CDLL("./libadd_cpu_c1.so")
gpu_lib = ctypes.CDLL("./libadd_gpu.so")

for lib in (cpu_lib, cpuc_lib ,gpu_lib):
    cpu_lib.add_matrix_cpu.argtypes = [
        np.ctypeslib.ndpointer(np.float32),
        np.ctypeslib.ndpointer(np.float32),
        np.ctypeslib.ndpointer(np.float32),
        ctypes.c_int
    ]
    cpuc_lib.add_matrix_cpu.argtypes = [
        np.ctypeslib.ndpointer(np.float32),
        np.ctypeslib.ndpointer(np.float32),
        np.ctypeslib.ndpointer(np.float32),
        ctypes.c_int
    ]
    gpu_lib.add_matrix_gpu.argtypes =  [
        np.ctypeslib.ndpointer(np.float32),
        np.ctypeslib.ndpointer(np.float32),
        np.ctypeslib.ndpointer(np.float32),
        ctypes.c_int
    ]


# N = 100_000_000
# N = 999_999_999
N = 800_000_000

A = np.random.rand(N).astype(np.float32)
B = np.random.rand(N).astype(np.float32)

C_cpu = np.zeros_like(A)
C_cpuc = np.zeros_like(A)
C_gpu = np.zeros_like(A)

print("N = 1B \n")
start = time.time()
print("Running CPU in C++")
cpu_lib.add_matrix_cpu(A, B, C_cpu, N)

print("Running CPU in C")
cpuc_lib.add_matrix_cpu(A,B, C_cpuc,N)

print("Running GPU in CUDA")
gpu_lib.add_matrix_gpu(A, B, C_gpu, N)

end = time.time()
print("Overall Time taken: ", (end-start)*1000, "milliseconds")
# Verify correctness
print("CPU C++ vs CPU C match?", np.allclose(C_cpu, C_cpuc))
print("CPU C++ vs GPU match?", np.allclose(C_cpu, C_gpu))


# print("\n Checking the outputs of some random index in array")
# print(A[100], B[100])
# print(C_cpu[100], C_cpuc[100] ,C_gpu[100])
