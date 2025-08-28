N = 32
CPU Matrix Multiplication took 0.01692000 ms
N = 32
CPU Matrix Multiplication took 0.01693400 ms
N = 32
GPU Matrix Multiplication took 0.01024000 ms
Difference CPU vs GPU: 1.9073486328125e-06
N = 64
CPU Matrix Multiplication took 0.05558600 ms
N = 64
CPU Matrix Multiplication took 0.06901900 ms
N = 64
GPU Matrix Multiplication took 0.00716800 ms
Difference CPU vs GPU: 3.814697265625e-06
N = 128
CPU Matrix Multiplication took 0.66503000 ms
N = 128
CPU Matrix Multiplication took 0.67194600 ms
N = 128
GPU Matrix Multiplication took 0.01331200 ms
Difference CPU vs GPU: 7.62939453125e-06
N = 256
CPU Matrix Multiplication took 6.49773800 ms
N = 256
CPU Matrix Multiplication took 6.53340200 ms
N = 256
GPU Matrix Multiplication took 0.12288000 ms
Difference CPU vs GPU: 2.288818359375e-05
N = 512
CPU Matrix Multiplication took 53.12393900 ms
N = 512
CPU Matrix Multiplication took 52.99649100 ms
N = 512
GPU Matrix Multiplication took 0.37878400 ms
Difference CPU vs GPU: 4.57763671875e-05
N = 1024
CPU Matrix Multiplication took 2135.39621800 ms
N = 1024
CPU Matrix Multiplication took 2122.16340300 ms
N = 1024
GPU Matrix Multiplication took 2.76185608 ms
Difference CPU vs GPU: 9.1552734375e-05
N = 2048
CPU Matrix Multiplication took 16490.89647900 ms
N = 2048
CPU Matrix Multiplication took 16737.17771300 ms
N = 2048
GPU Matrix Multiplication took 26.91027260 ms
Difference CPU vs GPU: 0.00018310546875
N = 3200
CPU Matrix Multiplication took 19763.77736500 ms
N = 3200
CPU Matrix Multiplication took 19554.68274600 ms
N = 3200
GPU Matrix Multiplication took 84.36441803 ms
Difference CPU vs GPU: 0.00030517578125




N = 16
CPU Matrix [c++] Multiplication took 0.00183500 ms
CPU Matrix [c] Multiplication took 0.00173900 ms
GPU Matrix Multiplication took 0.18403199 ms
Difference CPU vs GPU: 9.5367431640625e-07
Time Taken at the orchestrator level (Python): 0.27387213706970215  s

N = 32
CPU Matrix [c++] Multiplication took 0.00693200 ms
CPU Matrix [c] Multiplication took 0.00695900 ms
GPU Matrix Multiplication took 0.00409600 ms
Difference CPU vs GPU: 1.9073486328125e-06
Time Taken at the orchestrator level (Python): 0.0005974769592285156  s

N = 64
CPU Matrix [c++] Multiplication took 0.05542500 ms
CPU Matrix [c] Multiplication took 0.05533100 ms
GPU Matrix Multiplication took 0.00512000 ms
Difference CPU vs GPU: 3.814697265625e-06
Time Taken at the orchestrator level (Python): 0.0003566741943359375  s

N = 128
CPU Matrix [c++] Multiplication took 0.66805900 ms
CPU Matrix [c] Multiplication took 0.65593400 ms
GPU Matrix Multiplication took 0.01024000 ms
Difference CPU vs GPU: 1.1444091796875e-05
Time Taken at the orchestrator level (Python): 0.0017673969268798828  s

N = 256
CPU Matrix [c++] Multiplication took 6.47092600 ms
CPU Matrix [c] Multiplication took 6.49503700 ms
GPU Matrix Multiplication took 0.11942400 ms
Difference CPU vs GPU: 2.288818359375e-05
Time Taken at the orchestrator level (Python): 0.014115095138549805  s

N = 512
CPU Matrix [c++] Multiplication took 52.40413800 ms
CPU Matrix [c] Multiplication took 52.47567000 ms
GPU Matrix Multiplication took 0.37897599 ms
Difference CPU vs GPU: 4.57763671875e-05
Time Taken at the orchestrator level (Python): 0.10890412330627441  s

N = 1000
CPU Matrix [c++] Multiplication took 379.72060000 ms
CPU Matrix [c] Multiplication took 380.64587000 ms
GPU Matrix Multiplication took 2.55468798 ms
Difference CPU vs GPU: 9.1552734375e-05
Time Taken at the orchestrator level (Python): 0.777167558670044  s

N = 1024
CPU Matrix [c++] Multiplication took 2085.82702600 ms
CPU Matrix [c] Multiplication took 2079.78182300 ms
GPU Matrix Multiplication took 2.76499200 ms
Difference CPU vs GPU: 9.1552734375e-05
Time Taken at the orchestrator level (Python): 4.179962635040283  s

N = 2000
CPU Matrix [c++] Multiplication took 6892.97179500 ms
CPU Matrix [c] Multiplication took 3227.87716400 ms
GPU Matrix Multiplication took 21.07977676 ms
Difference CPU vs GPU: 0.00018310546875
Time Taken at the orchestrator level (Python): 10.185903310775757  s

N = 2048
CPU Matrix [c++] Multiplication took 16591.95966200 ms
CPU Matrix [c] Multiplication took 16702.95096900 ms
GPU Matrix Multiplication took 26.12928009 ms
Difference CPU vs GPU: 0.00018310546875
Time Taken at the orchestrator level (Python): 33.37999367713928  s




| N     | CPU [C++] Time (ms) | CPU [C] Time (ms) | GPU Time (ms) | Max Diff CPU vs GPU        | Python Orchestrator Time (s) |
|-------|---------------------|-------------------|---------------|----------------------------|------------------------------|
| 16    | 0.00183500           | 0.00173900        | 0.18403199    | 9.5367431640625e-07         | 0.27387213706970215           |
| 32    | 0.00693200           | 0.00695900        | 0.00409600    | 1.9073486328125e-06         | 0.0005974769592285156         |
| 64    | 0.05542500           | 0.05533100        | 0.00512000    | 3.814697265625e-06          | 0.0003566741943359375         |
| 128   | 0.66805900           | 0.65593400        | 0.01024000    | 1.1444091796875e-05         | 0.0017673969268798828         |
| 256   | 6.47092600           | 6.49503700        | 0.11942400    | 2.288818359375e-05          | 0.014115095138549805          |
| 512   | 52.40413800          | 52.47567000       | 0.37897599    | 4.57763671875e-05           | 0.10890412330627441           |
| 1000  | 379.72060000         | 380.64587000      | 2.55468798    | 9.1552734375e-05            | 0.777167558670044             |
| 1024  | 2085.82702600        | 2079.78182300     | 2.76499200    | 9.1552734375e-05            | 4.179962635040283             |
| 2000  | 6892.97179500        | 3227.87716400     | 21.07977676   | 0.00018310546875            | 10.185903310775757            |
| 2048  | 16591.95966200       | 16702.95096900    | 26.12928009   | 0.00018310546875            | 33.37999367713928             |

