#include <stdio.h>

int main() {
    printf("Size of char: %zu byte(s)\t %zu bit(s)\n", sizeof(char), sizeof(char)*8);
    printf("Size of short: %zu byte(s)\t %zu bit(s)\n", sizeof(short), sizeof(short)*8);
    printf("Size of int: %zu byte(s)\t %zu bit(s)\n", sizeof(int), sizeof(int)*8);
    printf("Size of long: %zu byte(s)\t %zu bit(s)\n", sizeof(long), sizeof(long)*8);
    printf("Size of long long: %zu byte(s)\t %zu bit(s)\n", sizeof(long long), sizeof(long long)*8);
    printf("Size of float: %zu byte(s)\t %zu bit(s)\n", sizeof(float), sizeof(float)*8);
    printf("Size of double: %zu byte(s)\t %zu bit(s)\n", sizeof(double), sizeof(double)*8);
    printf("Size of long double: %zu byte(s)\t %zu bit(s)\n", sizeof(long double), sizeof(long double)*8);
    printf("Size of _Bool: %zu byte(s)\t %zu bit(s)\n", sizeof(_Bool), sizeof(_Bool)*8);
    
    return 0;
}
