#include <stdio.h>
#include <math.h>

double ibm_to_ieee(__uint64_t n){
    int s, e, m;
    double ans;

    s = (n & (1 << 31)) != 0;
    e = (n >> 24) & 0x7f;
    m = n & 0xffffff;

    ans = m * pow(16, -6) * pow(16, e-64);
    return (s==1) ? -ans: ans;

}

int main() {
    __uint64_t ibm_num;
    double ieee_num;

    // Example IBM hex floating point number (change as needed)
    ibm_num = 0x411f0d0000000000ULL;

    ieee_num = ibm_to_ieee(ibm_num);

    printf("IBM hex: 0x%016lx\n", ibm_num);
    printf("IEEE double: %f\n", ieee_num);

    return 0;
}