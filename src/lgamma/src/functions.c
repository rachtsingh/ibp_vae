#include <TH/TH.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include "internals_s.h"

int polygamma(int n, THFloatTensor *input, THFloatTensor *output) {
  float *input_data, *output_data;
  input_data = THFloatTensor_data(input);
  output_data = THFloatTensor_data(output);

  int input_strideHeight = input->stride[0];
  int input_strideWidth = input->stride[1];

  int output_strideHeight = output->stride[0];
  int output_strideWidth = output->stride[1];

  int height = input->size[0];
  int width = input->size[1];

  if (input->nDimension != 2) {
    printf("Warning: polygamma input is supposed to be 2-dimensional\n");
  }
  
  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      const int out_address = x * output_strideWidth + y * output_strideHeight;
      const int in_address = x * input_strideWidth + y * input_strideHeight;
      output_data[out_address] = polygamma_impl(n, input_data[in_address]);
    }
  }

  return 1;
}

int polygamma_dbl(int n, THDoubleTensor *input, THDoubleTensor *output) {
  double *input_data, *output_data;
  input_data = THDoubleTensor_data(input);
  output_data = THDoubleTensor_data(output);

  int input_strideHeight = input->stride[0];
  int input_strideWidth = input->stride[1];

  int output_strideHeight = output->stride[0];
  int output_strideWidth = output->stride[1];

  int height = input->size[0];
  int width = input->size[1];

  if (input->nDimension != 2) {
    printf("Warning: polygamma input is supposed to be 2-dimensional\n");
  }

  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      const int out_address = x * output_strideWidth + y * output_strideHeight;
      const int in_address = x * input_strideWidth + y * input_strideHeight;
      output_data[out_address] = polygamma_impl_dbl(n, input_data[in_address]);
    }
  }

  return 1;
}

// lgamma, or log(gamma(x)) - we need the suffix py because the lgamma function
// is defined in math.h
int lgamma_py(THFloatTensor *input, THFloatTensor *output) {
  float *input_data, *output_data;
  input_data = THFloatTensor_data(input);
  output_data = THFloatTensor_data(output);

  int input_strideHeight = input->stride[0];
  int input_strideWidth = input->stride[1];

  int output_strideHeight = output->stride[0];
  int output_strideWidth = output->stride[1];

  int height = input->size[0];
  int width = input->size[1];

  if (input->nDimension != 2) {
    printf("Warning: lgamma is only implemented for 2-dimensional vectors\n");
  }

  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      const int out_address = x * output_strideWidth + y * output_strideHeight;
      const int in_address = x * input_strideWidth + y * input_strideHeight;
      output_data[out_address] = lgammaf(input_data[in_address]);
    }
  }

  return 1;
}

int lgamma_dbl_py(THDoubleTensor *input, THDoubleTensor *output) {
  double *input_data, *output_data;
  input_data = THDoubleTensor_data(input);
  output_data = THDoubleTensor_data(output);

  int input_strideHeight = input->stride[0];
  int input_strideWidth = input->stride[1];

  int output_strideHeight = output->stride[0];
  int output_strideWidth = output->stride[1];

  int height = input->size[0];
  int width = input->size[1];

  if (input->nDimension != 2) {
    printf("Warning: lgamma is only implemented for 2-dimensional vectors\n");
  }

  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      const int out_address = x * output_strideWidth + y * output_strideHeight;
      const int in_address = x * input_strideWidth + y * input_strideHeight;
      output_data[out_address] = lgamma(input_data[in_address]);
    }
  }

  return 1;
}

int lbeta(THFloatTensor *a, THFloatTensor *b, THFloatTensor *output) {
  float *a_data, *b_data, *output_data;
  a_data = THFloatTensor_data(a);
  b_data = THFloatTensor_data(b);
  output_data = THFloatTensor_data(output);

  int a_strideHeight = a->stride[0];
  int a_strideWidth = a->stride[1];

  int b_strideHeight = b->stride[0];
  int b_strideWidth = b->stride[1];

  int output_strideHeight = output->stride[0];
  int output_strideWidth = output->stride[1];

  // choosing arbitrarily
  int height = a->size[0];
  int width = a->size[1];

  if (a->nDimension != 2 || b->nDimension != 2) {
    printf("Warning: lbeta is only implemented for 2-dimensional vectors\n");
  }

  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      const int out_address = x * output_strideWidth + y * output_strideHeight;
      const int a_address = x * a_strideWidth + y * a_strideHeight;
      const int b_address = x * b_strideWidth + y * b_strideHeight;
      output_data[out_address] =
          lbeta_impl(a_data[a_address], b_data[b_address]);
    }
  }

  return 1;
}

int lbeta_dbl(THDoubleTensor *a, THDoubleTensor *b, THDoubleTensor *output) {
  double *a_data, *b_data, *output_data;
  a_data = THDoubleTensor_data(a);
  b_data = THDoubleTensor_data(b);
  output_data = THDoubleTensor_data(output);

  int a_strideHeight = a->stride[0];
  int a_strideWidth = a->stride[1];

  int b_strideHeight = b->stride[0];
  int b_strideWidth = b->stride[1];

  int output_strideHeight = output->stride[0];
  int output_strideWidth = output->stride[1];

  // choosing arbitrarily
  int height = a->size[0];
  int width = a->size[1];

  if (a->nDimension != 2 || b->nDimension != 2) {
    printf("Warning: lbeta is only implemented for 2-dimensional vectors\n");
  }

  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      const int out_address = x * output_strideWidth + y * output_strideHeight;
      const int a_address = x * a_strideWidth + y * a_strideHeight;
      const int b_address = x * b_strideWidth + y * b_strideHeight;
      output_data[out_address] =
          lbeta_impl_dbl(a_data[a_address], b_data[b_address]);
    }
  }

  return 1;
}
