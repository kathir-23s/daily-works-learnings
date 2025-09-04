#include <stdio.h>

int main() {
    // Write C code here
    signed char m = 3;
    signed int a, b;
    a = m << 9;
    b = (-m) << 9;
    
    printf("%d", a);
    printf("\n%d", b);
}