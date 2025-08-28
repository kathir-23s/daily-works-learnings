#include <stdio.h>
#include <stdlib.h>

int main() {
    int D1 = 2, D2 = 3, D3 = 4;

    // Allocate contiguous block of memory for all elements
    float* tensor = (float*)malloc(D1 * D2 * D3 * sizeof(float));

    // Set values in tensor[i][j][k]
    for (int i = 0; i < D1; i++) {
        for (int j = 0; j < D2; j++) {
            for (int k = 0; k < D3; k++) {
                int index = i * D2 * D3 + j * D3 + k;  // flatten index
                tensor[index] = (float)(index);
            }
        }
    }

    // Print values
    for (int i = 0; i < D1; i++) {
        printf("Block %d:\n", i);
        for (int j = 0; j < D2; j++) {
            for (int k = 0; k < D3; k++) {
                int index = i * D2 * D3 + j * D3 + k;
                printf("%.1f ", tensor[index]);
            }
            printf("\n");
        }
        printf("\n");
    }

    free(tensor);
    return 0;
}
