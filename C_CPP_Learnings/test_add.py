import ctypes
import numpy as np

# Load library
lib = ctypes.CDLL("objects/libtensor.so")

# Setup prototype: void tensor_add(void* a, void* b, void* out, int dtype, int64_t n)
lib.tensor_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                           ctypes.c_int, ctypes.c_longlong]

# Prepare test arrays
N = 5
a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
b = np.array([10, 20, 30, 40, 50], dtype=np.float32)
out = np.zeros_like(a)

# Call C ABI dispatcher (dtype=0 means float32)
lib.tensor_add(a.ctypes.data, b.ctypes.data, out.ctypes.data, 0, N)

print("a:", a)
print("b:", b)
print("out:", out)
