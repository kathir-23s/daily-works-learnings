#include <stdio.h>

unsigned char sqrt(unsigned char n) {
unsigned char c=0, p=1;
while (n >= p) {
n -= p;
p += 2;
c++;
}
return c;
}

int main(){
    unsigned char in = 27;
    unsigned res = sqrt(in);
    printf("%d\n", res);
}