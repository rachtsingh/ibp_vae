#include <THC/THC.h>
#include <stdbool.h>
#include <stdio.h>
#include "functions_cuda_kernel.h"

#define real float

extern THCState *state;

int polygamma_cuda(int n, THCudaTensor *input, THCudaTensor *output) {
  float *input_data, *output_data;
  input_data = THCudaTensor_data(state, input);
  output_data = THCudaTensor_data(state, output);

  int input_strideHeight = THCudaTensor_stride(state, input, 0);
  int input_strideWidth = THCudaTensor_stride(state, input, 1);

  int output_strideHeight = THCudaTensor_stride(state, output, 0);
  int output_strideWidth = THCudaTensor_stride(state, output, 1);

  int height = THCudaTensor_size(state, input, 0);
  int width = THCudaTensor_size(state, input, 1);

  if (input->nDimension != 2) {
    printf("Warning: polygamma input is supposed to be 2-dimensional\n");
  }

  int error = 0;
  error = polygamma_cuda_wrapped(n, input_strideHeight, input_strideWidth, output_strideHeight, output_strideWidth, height, width, input_data, output_data);
  if (error) {
    THError("aborting, cuda kernel failed.\n");
  }
  return 0;
}

int polygamma_cuda_dbl(int n, THCudaDoubleTensor *input, THCudaDoubleTensor *output) {
  double *input_data, *output_data;
  input_data = THCudaDoubleTensor_data(state, input);
  output_data = THCudaDoubleTensor_data(state, output);

  int input_strideHeight = THCudaDoubleTensor_stride(state, input, 0);
  int input_strideWidth = THCudaDoubleTensor_stride(state, input, 1);

  int output_strideHeight = THCudaDoubleTensor_stride(state, output, 0);
  int output_strideWidth = THCudaDoubleTensor_stride(state, output, 1);

  int height = THCudaDoubleTensor_size(state, input, 0);
  int width = THCudaDoubleTensor_size(state, input, 1);

  if (input->nDimension != 2) {
    printf("Warning: polygamma input is supposed to be 2-dimensional\n");
  }

  int error = 0;
  error = polygamma_cuda_dbl_wrapped(n, input_strideHeight, input_strideWidth, output_strideHeight, output_strideWidth, height, width, input_data, output_data);
  if (error) {
    THError("aborting, cuda kernel failed.\n");
  }
  return 0;
}

int lgamma_cuda(THCudaTensor *input, THCudaTensor *output) {
  float *input_data, *output_data;
  input_data = THCudaTensor_data(state, input);
  output_data = THCudaTensor_data(state, output);

  int input_strideHeight = THCudaTensor_stride(state, input, 0);
  int input_strideWidth = THCudaTensor_stride(state, input, 1);

  int output_strideHeight = THCudaTensor_stride(state, output, 0);
  int output_strideWidth = THCudaTensor_stride(state, output, 1);

  int height = THCudaTensor_size(state, input, 0);
  int width = THCudaTensor_size(state, input, 1);

  if (input->nDimension != 2) {
    printf("Warning: polygamma input is supposed to be 2-dimensional\n");
  }

  int error = 0;
  error = lgamma_cuda_wrapped(input_strideHeight, input_strideWidth, output_strideHeight, output_strideWidth, height, width, input_data, output_data);
  if (error) {
    THError("aborting, cuda kernel failed.\n");
  }
  return 0;
}

int lgamma_cuda_dbl(THCudaDoubleTensor *input, THCudaDoubleTensor *output) {
  double *input_data, *output_data;
  input_data = THCudaDoubleTensor_data(state, input);
  output_data = THCudaDoubleTensor_data(state, output);

  int input_strideHeight = THCudaDoubleTensor_stride(state, input, 0);
  int input_strideWidth = THCudaDoubleTensor_stride(state, input, 1);

  int output_strideHeight = THCudaDoubleTensor_stride(state, output, 0);
  int output_strideWidth = THCudaDoubleTensor_stride(state, output, 1);

  int height = THCudaDoubleTensor_size(state, input, 0);
  int width = THCudaDoubleTensor_size(state, input, 1);

  if (input->nDimension != 2) {
    printf("Warning: polygamma input is supposed to be 2-dimensional\n");
  }

  int error = 0;
  error = lgamma_cuda_dbl_wrapped(input_strideHeight, input_strideWidth, output_strideHeight, output_strideWidth, height, width, input_data, output_data);
  if (error) {
    THError("aborting, cuda kernel failed.\n");
  }
  return 0;
}

int lbeta_cuda(THCudaTensor *a, THCudaTensor *b, THCudaTensor *output) {
  float *a_data, *b_data, *output_data;
  a_data = THCudaTensor_data(state, a);
  b_data = THCudaTensor_data(state, b);
  output_data = THCudaTensor_data(state, output);

  int a_strideHeight = THCudaTensor_stride(state, a, 0);
  int a_strideWidth = THCudaTensor_stride(state, a, 1);

  int b_strideHeight = THCudaTensor_stride(state, b, 0);
  int b_strideWidth = THCudaTensor_stride(state, b, 1);

  int output_strideHeight = THCudaTensor_stride(state, output, 0);
  int output_strideWidth = THCudaTensor_stride(state, output, 1);

  int height = THCudaTensor_size(state, a, 0);
  int width = THCudaTensor_size(state, a, 1);

  if (a->nDimension != 2 || b->nDimension != 2) {
    THError("Error: polygamma input is supposed to be 2-dimensional\n");
  }

  if (THCudaTensor_size(state, b, 0) != height || THCudaTensor_size(state, b, 1) != width) {
    THError("Error: a and b are not the same shape.\n");
  }

  int error = 0;
  error = lbeta_cuda_wrapped(a_strideHeight, a_strideWidth, b_strideHeight, b_strideWidth, output_strideHeight, output_strideWidth, height, width, a_data, b_data, output_data);
  if (error) {
    THError("aborting, cuda kernel failed.\n");
  }
  return 0;
}

int lbeta_cuda_dbl(THCudaDoubleTensor *a, THCudaDoubleTensor *b, THCudaDoubleTensor *output) {
  double *a_data, *b_data, *output_data;
  a_data = THCudaDoubleTensor_data(state, a);
  b_data = THCudaDoubleTensor_data(state, b);
  output_data = THCudaDoubleTensor_data(state, output);

  int a_strideHeight = THCudaDoubleTensor_stride(state, a, 0);
  int a_strideWidth = THCudaDoubleTensor_stride(state, a, 1);

  int b_strideHeight = THCudaDoubleTensor_stride(state, b, 0);
  int b_strideWidth = THCudaDoubleTensor_stride(state, b, 1);

  int output_strideHeight = THCudaDoubleTensor_stride(state, output, 0);
  int output_strideWidth = THCudaDoubleTensor_stride(state, output, 1);

  int height = THCudaDoubleTensor_size(state, a, 0);
  int width = THCudaDoubleTensor_size(state, a, 1);

  if (a->nDimension != 2 || b->nDimension != 2) {
    THError("Error: polygamma input is supposed to be 2-dimensional\n");
  }

  if (THCudaDoubleTensor_size(state, b, 0) != height || THCudaDoubleTensor_size(state, b, 1) != width) {
    THError("Error: a and b are not the same shape.\n");
  }

  int error = 0;
  error = lbeta_cuda_dbl_wrapped(a_strideHeight, a_strideWidth, b_strideHeight, b_strideWidth, output_strideHeight, output_strideWidth, height, width, a_data, b_data, output_data);
  if (error) {
    THError("aborting, cuda kernel failed.\n");
  }
  return 0;
}

void init_random(void) {
  init_rand();
}

int sample_gamma_dbl(THCudaDoubleTensor *a, THCudaDoubleTensor *output) {
  double *a_data, *output_data;
  a_data = THCudaDoubleTensor_data(state, a);
  output_data = THCudaDoubleTensor_data(state, output);

  int height = THCudaDoubleTensor_size(state, a, 0);
  int width = THCudaDoubleTensor_size(state, a, 1);

  int error = 0;
  error = sample_gamma_dbl_wrapped(height, width, a_data, output_data);
  if (error) {
    THError("aborting, cuda kernel failed.\n");
  }
  return 0;
}

// only used if sampling using the above function doesn't work for numerical reasons
int sample_beta_dbl(THCudaDoubleTensor *a, THCudaDoubleTensor *b, THCudaDoubleTensor *output) {
  double *a_data, *b_data, *output_data;
  a_data = THCudaDoubleTensor_data(state, a);
  b_data = THCudaDoubleTensor_data(state, b);
  output_data = THCudaDoubleTensor_data(state, output);

  int height = THCudaDoubleTensor_size(state, a, 0);
  int width = THCudaDoubleTensor_size(state, a, 1);

  int error = 0;
  error = sample_beta_dbl_wrapped(height, width, a_data, b_data, output_data);
  if (error) {
    THError("aborting, cuda kernel failed.\n");
  }
  return 0;
}
