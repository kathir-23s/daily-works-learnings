#include <stdio.h>

int main() {
	unsigned char a = 0x8a; //INPUTS
	
    // SOLUTIONS
    unsigned char a1 = a | 0x48; 
    unsigned char a2 = a & 0x0f;
    unsigned char a3;
    if ((a & 0x08) ){
        a3 = 0x00;
    }
    else {
        a3 = 0x08;
    }
    unsigned char a4 = a & 0xfd;
    unsigned char a5 = a ^ 0x18;

    printf("A = %02X\n", a);
    printf("Set bit 3 and 6: %02X\n", a1);
    printf("Low nibble of A: %02X\n", a2);
    printf("Set b to 100 if b3 in a is not set: %02X\n", a3);
    printf("Clear bit 1 of A: %02X\n", a4);
    printf("Toggle bits 4 and 5: %02X\n", a5);

	return 0;
}
