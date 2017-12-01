// polygamma
int polygamma(int n, THFloatTensor* input, THFloatTensor* output);
int polygamma_dbl(int n, THDoubleTensor* input, THDoubleTensor* output);

// lgamma
int lgamma_py(THFloatTensor *input, THFloatTensor *output);
int lgamma_dbl_py(THDoubleTensor *input, THDoubleTensor *output);

// lbeta
int lbeta(THFloatTensor *a, THFloatTensor *b, THFloatTensor *output);
int lbeta_dbl(THDoubleTensor *a, THDoubleTensor *b, THDoubleTensor *output);
