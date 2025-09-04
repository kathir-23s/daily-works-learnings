#include <stdio.h>

int detect_overflow(int a, int b) {
    int sum = a + b;
    if (((a ^ sum) & (b ^ sum)) < 0) return 1;
    return 0;
}