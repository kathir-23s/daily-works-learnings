import numpy as np, time

def mm_classic(A, B):
    return A @ B

def strassen(A, B, b=128):
    n = A.shape[0]
    if n <= b:
        return A @ B
    k = n // 2
    A11,A12,A21,A22 = A[:k,:k], A[:k,k:], A[k:,:k], A[k:,k:]
    B11,B12,B21,B22 = B[:k,:k], B[:k,k:], B[k:,:k], B[k:,k:]

    M1 = strassen(A11 + A22, B11 + B22, b)
    M2 = strassen(A21 + A22, B11, b)
    M3 = strassen(A11, B12 - B22, b)
    M4 = strassen(A22, B21 - B11, b)
    M5 = strassen(A11 + A12, B22, b)
    M6 = strassen(A21 - A11, B11 + B12, b)
    M7 = strassen(A12 - A22, B21 + B22, b)

    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    C = np.empty_like(A)
    C[:k,:k], C[:k,k:], C[k:,:k], C[k:,k:] = C11, C12, C21, C22
    return C

def time_one(fn, A, B, reps=3):
    t = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        C = fn(A, B)
        t = min(t, time.perf_counter() - t0)
    return t

for n in [128, 256, 512, 1024, 2048]:
    A = np.random.randn(n, n).astype(np.float64)
    B = np.random.randn(n, n).astype(np.float64)
    t_base = time_one(mm_classic, A, B)
    t_s128 = time_one(lambda X,Y: strassen(X,Y,b=128), A, B)
    t_s256 = time_one(lambda X,Y: strassen(X,Y,b=256), A, B)
    print(f"n={n:4} | classic {t_base:.3f}s | S(b=128) {t_s128:.3f}s | S(b=256) {t_s256:.3f}s")
