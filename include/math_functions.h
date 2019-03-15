#ifndef MATH_FUNCTIONS_H
#define MATH_FUNCTIONS_H

#include "string.h"
#include <math.h>

extern "C" {
#include <cblas.h>
}

void caffe_copy(const int N, const float *X, float *Y);

void caffe_set(const int N, const float alpha, float *X);

void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
	const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
	const float alpha, const float* A, const float* B, const float beta,
	float* C);

void caffe_sqr(const int N, const float* a, float* y);

float caffe_cpu_asum(const int n, const float* x);

void caffe_cpu_scale(const int n, const float alpha, const float *x, float* y);

void caffe_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M,
	const int N, const float alpha, const float* A, const float* x,
	const float beta, float* y);

void caffe_powx(const int n, const float* a, const float b, float* y);

void caffe_div(const int n, const float* a, const float* b, float* y);

void caffe_scal(const int N, const float alpha, float *X);

void caffe_mul(const int n, const float* a, const float* b, float* y);

void caffe_exp(const int n, const float* a, float* y);

#endif
