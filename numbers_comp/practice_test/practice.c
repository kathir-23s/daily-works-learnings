#include <stdio.h>
#include <stdint.h>
#include <string.h>

unsigned int base36_to_dec(const char *s) {
    unsigned int res = 0;
    while (*s) {
        char c = *s++;
        int val = (c >= '0' && c <= '9') ? (c - '0') : (c - 'A' + 10);
        res = res * 36 + val;
    }
    return res;
}

unsigned int binary_to_bcd(unsigned int x) {
    unsigned int bcd = 0, shift = 0;
    while (x > 0) {
        unsigned int digit = x % 10;
        bcd |= (digit << (shift * 4));
        x /= 10;
        shift++;
    }
    return bcd;
}

unsigned int mul_shift_add(unsigned int a, unsigned int b) {
    unsigned int res = 0;
    while (b > 0) {
        if (b & 1) res += a;
        a <<= 1;
        b >>= 1;
    }
    return res;
}

void restoring_division(unsigned int dividend, unsigned int divisor) {
    unsigned int quotient = 0, remainder = 0;
    for (int i = 31; i >= 0; i--) {
        remainder = (remainder << 1) | ((dividend >> i) & 1);
        if (remainder >= divisor) {
            remainder -= divisor;
            quotient |= (1U << i);
        }
    }
    printf("Q=%u R=%u\n", quotient, remainder);
}

int detect_overflow(int a, int b) {
    int sum = a + b;
    if (((a ^ sum) & (b ^ sum)) < 0) return 1;
    return 0;
}

uint32_t float_to_bits(float f) {
    union { float f; uint32_t u; } x;
    x.f = f;
    return x.u;
}

void encode_ieee(float val) {
    uint32_t bits = float_to_bits(val);
    int sign = (bits >> 31) & 1;
    int exp  = (bits >> 23) & 0xFF;
    uint32_t mant = bits & 0x7FFFFF;
    printf("Sign=%d Exp=%d Mant=0x%X\n", sign, exp, mant);
}

uint32_t rotate_right(uint32_t x, int k) {
    return (x >> k) | (x << (32 - k));
}

int has_alternating_bits(uint32_t x) {
    uint32_t y = x ^ (x >> 1);
    return (y & (y + 1)) == 0;
}

int count_set_bits(uint32_t x) {
    int c = 0;
    while (x) {
        x &= (x - 1);
        c++;
    }
    return c;
}

int main() {
    printf("Q1: %u\n", base36_to_dec("1Z3"));
    printf("Q2: %u\n", binary_to_bcd(0b101011));
    printf("Q3: %u\n", mul_shift_add(13, 9));
    printf("Q4: "); restoring_division(53, 5);
    printf("Q5: Overflow=%d\n", detect_overflow(2000000000, 2000000000));
    printf("Q6: "); encode_ieee(-19.625f);
    printf("Q7: 0x%X\n", rotate_right(0xF1234567, 8));
    printf("Q8: %d\n", has_alternating_bits(0b10101010));
    printf("Q9: %d\n", count_set_bits(0xF0F0F0F0));
    printf("Q10: "); encode_ieee(156.375f);
    return 0;
}
