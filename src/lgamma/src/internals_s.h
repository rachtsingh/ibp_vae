// polygamma float
 int zeta_impl_series(float *a, float *b, float *s, const float x, const float machep);
 float zeta_impl(float x, float q);
 float polygamma_impl(int n, float x);

// polygamma double
 int zeta_impl_series_dbl(double *a, double *b, double *s, const double x, const double machep);
 double zeta_impl_dbl(double x, double q);
 double polygamma_impl_dbl(int n, double x);

// beta
 double alnrel(double a);
 double algdiv(double a, double b);
 float lbeta_impl(float a, float b);
 double lbeta_impl_dbl(double a, double b);
