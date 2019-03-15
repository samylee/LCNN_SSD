#include "math_functions.h"

void caffe_copy(const int N, const float* X, float* Y) {
  if (X != Y) 
  {
	  memcpy(Y, X, sizeof(float) * N);  // NOLINT(caffe/alt_fn)
  }
}

void caffe_set(const int N, const float alpha, float* Y) {
	if (alpha == 0) {
		memset(Y, 0, sizeof(float) * N);  // NOLINT(caffe/alt_fn)
		return;
	}
	for (int i = 0; i < N; ++i) {
		Y[i] = alpha;
	}
}

void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
	const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
	const float alpha, const float* A, const float* B, const float beta,
	float* C) {
	int lda = (TransA == CblasNoTrans) ? K : M;
	int ldb = (TransB == CblasNoTrans) ? N : K;
	cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
		ldb, beta, C, N);
}

void caffe_sqr(const int N, const float* a, float* y){
	for (int i = 0; i < N; ++i) {
		y[i] = a[i] * a[i];
	}
}

float caffe_cpu_asum(const int n, const float* x) {
	return cblas_sasum(n, x, 1);
}

void caffe_cpu_scale(const int n, const float alpha, const float *x, float* y) {
	cblas_scopy(n, x, 1, y, 1);
	cblas_sscal(n, alpha, y, 1);
}

void caffe_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M,
	const int N, const float alpha, const float* A, const float* x,
	const float beta, float* y) {
	cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

void caffe_powx(const int n, const float* a, const float b, float* y) {
	for (int i = 0; i < n; ++i){
		y[i] = pow(a[i], b);
	}
}

void caffe_div(const int n, const float* a, const float* b, float* y) {
	for (int i = 0; i < n; ++i) {
		y[i] = a[i] / b[i];
	}
}

void caffe_scal(const int N, const float alpha, float *X) {
	cblas_sscal(N, alpha, X, 1);
}

void caffe_mul(const int n, const float* a, const float* b, float* y) {
	for (int i = 0; i < n; ++i){
		y[i] = a[i] * b[i];
	}
}

void caffe_exp(const int n, const float* a, float* y) {
	for (int i = 0; i < n; ++i) {
		y[i] = exp(a[i]);
	}
}