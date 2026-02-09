void gemmV0(int* A, int* B, int* C, int n, int m, int k) {
    for (int i = 0; i < n; i++) {
        for (int t = 0; t < k; t++) {
            for (int j = 0; j < m; j++) {
                C[i * m + j] += A[i * k + t] * B[t * m + j]; 
            }
        }
    }
}

