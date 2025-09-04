#include <stdio.h>
#include <stdlib.h>

// void dec_to_bin(unsigned char x, char out[9]){
//     for (int i = 7; i >=0; --i)
//         out[7-i] = ((x >> i) & 1) ? '1' : '0';
//     out[8] = '\0';
// }

void dec_to_bin(int n, char out[9]){
   for (int i=7; i>=0; --i){
    out[i] = (n%2) ? '1': '0';
    n = n/2;
   }
   out[8] = '\0';
    
}
int main(){
    char s[9];
    // int in = 47;
    dec_to_bin(47, s);
    printf("%s\n", s);
}