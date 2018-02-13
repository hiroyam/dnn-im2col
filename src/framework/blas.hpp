namespace di {
void axpy(const int      N,
          const float_t  alpha,
          const float_t *x,
          const int      incx,
          float_t       *y,
          const int      incy) {
    for (int n = 0; n < N; n++) {
        y[n * incy] += alpha * x[n * incx];
    }
}

void gemv(char           trans,
          const int      M,
          const int      N,
          const float_t  alpha,
          const float_t *A,
          const int      lda,
          const float_t *x,
          const int      incx,
          const float_t  beta,
          float_t       *y,
          const int      incy) {
    if (trans == 'N') {
        for (int m = 0; m < M; m++) {
            float_t sum = 0.0;
            for (int n = 0; n < N; n++) {
                sum += A[m + n * lda] * x[n * incy];
            }
            y[m * incy] = alpha * sum + beta * y[m * incy];
        }
    }

    if (trans == 'T') {
        for (int m = 0; m < M; m++) {
            float_t sum = 0.0;
            for (int n = 0; n < N; n++) {
                sum += A[n + m * lda] * x[n * incy];
            }
            y[m * incy] = alpha * sum + beta * y[m * incy];
        }
    }
}

void gemm(char           major,
          char           transa,
          char           transb,
          const int      M,
          const int      N,
          const int      K,
          const float_t  alpha,
          const float_t *A,
          const int      lda,
          const float_t *B,
          const int      ldb,
          const float_t  beta,
          float_t       *C,
          const int      ldc) {
    //
    // RowMajor
    //
    if (major == 'R') {
        if (transa == 'N' && transb == 'N') {
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float_t sum = 0.0;
                    for (int k = 0; k < K; k++) {
                        float_t tmp = A[k + m * lda] * B[n + k * ldb];
                        sum += tmp;
                    }
                    C[n + m * ldc] = alpha * sum + beta * C[n + m * ldc];
                }
            }
        }

        if (transa == 'T' && transb == 'N') {
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float_t sum = 0.0;
                    for (int k = 0; k < K; k++) {
                        float_t tmp = A[m + k * lda] * B[n + k * ldb];
                        sum += tmp;
                    }
                    C[n + m * ldc] = alpha * sum + beta * C[n + m * ldc];
                }
            }
        }

        if (transa == 'N' && transb == 'T') {
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float_t sum = 0.0;
                    for (int k = 0; k < K; k++) {
                        float_t tmp = A[k + m * lda] * B[k + n * ldb];
                        sum += tmp;
                    }
                    C[n + m * ldc] = alpha * sum + beta * C[n + m * ldc];
                }
            }
        }

        if (transa == 'T' && transb == 'T') {
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float_t sum = 0.0;
                    for (int k = 0; k < K; k++) {
                        float_t tmp = A[m + k * lda] * B[k + n * ldb];
                        sum += tmp;
                    }
                    C[n + m * ldc] = alpha * sum + beta * C[n + m * ldc];
                }
            }
        }
    }

    //
    // ColMajor
    //
    if (major == 'C') {
        if (transa == 'N' && transb == 'N') {
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float_t sum = 0.0;
                    for (int k = 0; k < K; k++) {
                        float_t tmp = A[m + k * lda] * B[k + n * ldb];
                        sum += tmp;
                    }
                    C[m + n * ldc] = alpha * sum + beta * C[m + n * ldc];
                }
            }
        }

        if (transa == 'T' && transb == 'N') {
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float_t sum = 0.0;
                    for (int k = 0; k < K; k++) {
                        float_t tmp = A[k + m * lda] * B[k + n * ldb];
                        sum += tmp;
                    }
                    C[m + n * ldc] = alpha * sum + beta * C[m + n * ldc];
                }
            }
        }

        if (transa == 'N' && transb == 'T') {
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float_t sum = 0.0;
                    for (int k = 0; k < K; k++) {
                        float_t tmp = A[m + k * lda] * B[n + k * ldb];
                        sum += tmp;
                    }
                    C[m + n * ldc] = alpha * sum + beta * C[m + n * ldc];
                }
            }
        }

        if (transa == 'T' && transb == 'T') {
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float_t sum = 0.0;
                    for (int k = 0; k < K; k++) {
                        float_t tmp = A[k + m * lda] * B[n + k * ldb];
                        sum += tmp;
                    }
                    C[m + n * ldc] = alpha * sum + beta * C[m + n * ldc];
                }
            }
        }
    }
}
} // namespace di
