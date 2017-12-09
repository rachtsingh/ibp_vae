#ifdef __cplusplus
extern "C" {
#endif

// float
int polygamma_cuda_wrapped(int n, int input_sheight, int input_swidth, int output_sheight, int output_swidth, int height, int width, float *input_data, float *output_data);
int lgamma_cuda_wrapped(int input_strideHeight, int input_strideWidth, int output_strideHeight, int output_strideWidth, int height, int width, float *input_data, float *output_data);
int lbeta_cuda_wrapped(int a_strideHeight, int a_strideWidth, int b_strideHeight, int b_strideWidth, int output_strideHeight, int output_strideWidth, int height, int width, float *a_data, float *b_data, float *output_data);

// double
int polygamma_cuda_dbl_wrapped(int n, int input_sheight, int input_swidth, int output_sheight, int output_swidth, int height, int width, double *input_data, double *output_data);
int lgamma_cuda_dbl_wrapped(int input_strideHeight, int input_strideWidth, int output_strideHeight, int output_strideWidth, int height, int width, double *input_data, double *output_data);
int lbeta_cuda_dbl_wrapped(int a_strideHeight, int a_strideWidth, int b_strideHeight, int b_strideWidth, int output_strideHeight, int output_strideWidth, int height, int width, double *a_data, double *b_data, double *output_data);

int sample_gamma_dbl_wrapped(int height, int width, double *a_data, double *output_data);
int sample_beta_dbl_wrapped(int height, int width, double *a_data, double *b_data, double *output_data);
void init_rand(void);

#ifdef __cplusplus
}
#endif
