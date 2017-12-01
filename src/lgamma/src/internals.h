// polygamma float
__BOTH__ int zeta_impl_series(float *a, float *b, float *s, const float x, const float machep);
__BOTH__ float zeta_impl(float x, float q);
__BOTH__ float polygamma_impl(int n, float x);

// polygamma double
__BOTH__ int zeta_impl_series_dbl(double *a, double *b, double *s, const double x, const double machep);
__BOTH__ double zeta_impl_dbl(double x, double q);
__BOTH__ double polygamma_impl_dbl(int n, double x);

// beta
__BOTH__ double alnrel(double a);
__BOTH__ double algdiv(double a, double b);
__BOTH__ float lbeta_impl(float a, float b);
__BOTH__ double lbeta_impl_dbl(double a, double b);
