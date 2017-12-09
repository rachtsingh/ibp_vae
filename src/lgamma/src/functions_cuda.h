// float
int polygamma_cuda(int n, THCudaTensor *input, THCudaTensor *output);
int lgamma_cuda(THCudaTensor *input, THCudaTensor *output);
int lbeta_cuda(THCudaTensor *a, THCudaTensor *b, THCudaTensor *output);

// double
int polygamma_cuda_dbl(int n, THCudaDoubleTensor *input, THCudaDoubleTensor *output);
int lgamma_cuda_dbl(THCudaDoubleTensor *input, THCudaDoubleTensor *output);
int lbeta_cuda_dbl(THCudaDoubleTensor *a, THCudaDoubleTensor *b, THCudaDoubleTensor *output);

void init_random(void);
int sample_gamma_dbl(THCudaDoubleTensor *a, THCudaDoubleTensor *output);
int sample_beta_dbl(THCudaDoubleTensor *a, THCudaDoubleTensor *b, THCudaDoubleTensor *output);
