#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include "internals.h"

/* #if __NVCC__ */
/*   #define __BOTH__ __device__ __host__ */
/*   #include <cfloat> */
/* #else */
/*   #define __BOTH__  */
/*   #include <float.h> */
/* #endif */

/* #include "internals.h" */

/* #if __CUDACC__ */
/* #else */
/*   #define __BOTH__ __device__  __host__ */
/* #endif */

__BOTH__ int zeta_impl_series(float *a, float *b, float *s, const float x,
                     const float machep) {
  int i = 0;
  while (i < 9) {
    i += 1;
    *a += 1.0f;
    *b = powf(*a, -x);
    *s += *b;
    if (fabsf(*b / *s) < machep) {
      return true;
    }
  }

  // Return whether we are done
  return false;
}

__BOTH__ int zeta_impl_series_dbl(double *a, double *b, double *s, const double x,
                         const double machep) {
  int i = 0;
  while ((i < 9) || (*a <= 9.0)) {
    i += 1;
    *a += 1.0;
    *b = pow(*a, -x);
    *s += *b;
    if (fabs(*b / *s) < machep) return true;
  }

  // Return whether we are done
  return false;
}

__BOTH__ float zeta_impl(float x, float q) {
  int i;
  float p, r, a, b, k, s, t, w;

  const float A[] = {
      12.0,
      -720.0,
      30240.0,
      -1209600.0,
      47900160.0,
      -1.8924375803183791606e9, /*1.307674368e12/691*/
      7.47242496e10,
      -2.950130727918164224e12,  /*1.067062284288e16/3617*/
      1.1646782814350067249e14,  /*5.109094217170944e18/43867*/
      -4.5979787224074726105e15, /*8.028576626982912e20/174611*/
      1.8152105401943546773e17,  /*1.5511210043330985984e23/854513*/
      -7.1661652561756670113e18  /*1.6938241367317436694528e27/236364091*/
  };

  const float maxnum = FLT_MAX;
  const float zero = 0.0, half = 0.5, one = 1.0;
  const float machep = 1e-15;

  if (x == one) return maxnum;

  if (x < one) {
    return zero;
  }

  if (q <= zero) {
    if (q == floorf(q)) {
      return maxnum;
    }
    p = x;
    r = floorf(p);
    if (p != r) return zero;
  }

  /* Permit negative q but continue sum until n+q > +9 .
   * This case should be handled by a reflection formula.
   * If q<0 and x is an integer, there is a relation to
   * the polygamma function.
   */
  s = powf(q, -x);
  a = q;
  b = zero;

  // Run the summation in a helper function that is specific to the floating
  // precision
  if (zeta_impl_series(&a, &b, &s, x, machep)) {
    return s;
  }

  w = a;
  s += b * w / (x - one);
  s -= half * b;
  a = one;
  k = zero;
  for (i = 0; i < 12; i++) {
    a *= x + k;
    b /= w;
    t = a * b / A[i];
    s = s + t;
    t = fabsf(t / s);
    if (t < machep) return s;
    k += one;
    a *= x + k;
    b /= w;
    k += one;
  }
  return s;
};

__BOTH__ double zeta_impl_dbl(double x, double q) {
  int i;
  double p, r, a, b, k, s, t, w;

  const double A[] = {
      12.0,
      -720.0,
      30240.0,
      -1209600.0,
      47900160.0,
      -1.8924375803183791606e9, /*1.307674368e12/691*/
      7.47242496e10,
      -2.950130727918164224e12,  /*1.067062284288e16/3617*/
      1.1646782814350067249e14,  /*5.109094217170944e18/43867*/
      -4.5979787224074726105e15, /*8.028576626982912e20/174611*/
      1.8152105401943546773e17,  /*1.5511210043330985984e23/854513*/
      -7.1661652561756670113e18  /*1.6938241367317436694528e27/236364091*/
  };

  const double maxnum = DBL_MAX;
  const double zero = 0.0, half = 0.5, one = 1.0;
  const double machep = 1e-15;

  if (x == one) return maxnum;

  if (x < one) {
    return zero;
  }

  if (q <= zero) {
    if (q == floor(q)) {
      return maxnum;
    }
    p = x;
    r = floor(p);
    if (p != r) return zero;
  }

  /* Permit negative q but continue sum until n+q > +9 .
   * This case should be handled by a reflection formula.
   * If q<0 and x is an integer, there is a relation to
   * the polygamma function.
   */
  s = pow(q, -x);
  a = q;
  b = zero;

  // Run the summation in a helper function that is specific to the doubleing
  // precision
  if (zeta_impl_series_dbl(&a, &b, &s, x, machep)) {
    return s;
  }

  w = a;
  s += b * w / (x - one);
  s -= half * b;
  a = one;
  k = zero;
  for (i = 0; i < 12; i++) {
    a *= x + k;
    b /= w;
    t = a * b / A[i];
    s = s + t;
    t = fabs(t / s);
    if (t < machep) return s;
    k += one;
    a *= x + k;
    b /= w;
    k += one;
  }
  return s;
};

__BOTH__ float polynomial_evaluation(float x, const float *f, int n) {
  float result = 0.0;
  for (int i = 0; i < n; i++) {
    result *= x;
    result += f[i];
  }
  return result;
}

__BOTH__ double polynomial_evaluation_dbl(double x, const double *f, int n) {
  double result = 0.0;
  for (int i = 0; i < n; i++) {
    result *= x;
    result += f[i];
  }
  return result;
}

__BOTH__ float digamma_impl_maybe_poly(const float s) {
  const float A[] = {-4.16666666666666666667E-3f, 3.96825396825396825397E-3f,
                     -8.33333333333333333333E-3f, 8.33333333333333333333E-2f};
  float z;
  if (s < 1.0e8f) {
    z = 1.0f / (s * s);
    return z * polynomial_evaluation(z, A, 4);
  } else {
    return 0.0f;
  }
}

__BOTH__ double digamma_impl_maybe_poly_dbl(const double s) {
  const double A[] = {8.33333333333333333333E-2, -2.10927960927960927961E-2,
                      7.57575757575757575758E-3, -4.16666666666666666667E-3,
                      3.96825396825396825397E-3, -8.33333333333333333333E-3,
                      8.33333333333333333333E-2};

  double z;
  if (s < 1.0e17) {
    z = 1.0 / (s * s);
    return z * polynomial_evaluation_dbl(z, A, 7);
  } else
    return 0.0;
}

__BOTH__ float digamma_impl(const float u) {
  float x = u;
  float p, q, nz, s, w, y;
  bool negative;

  const float maxnum = FLT_MAX;
  const float m_pi = M_PI;

  negative = 0;
  nz = 0.0;

  const float zero = 0.0;
  const float one = 1.0;
  const float half = 0.5;

  if (x <= zero) {
    negative = one;
    q = x;
    p = floorf(q);
    if (p == q) {
      return maxnum;
    }
    /* Remove the zeros of tan(m_pi x)
     * by subtracting the nearest integer from x
     */
    nz = q - p;
    if (nz != half) {
      if (nz > half) {
        p += one;
        nz = q - p;
      }
      nz = m_pi / tanf(m_pi * nz);
    } else {
      nz = zero;
    }
    x = one - x;
  }

  /* use the recurrence psi(x+1) = psi(x) + 1/x. */
  s = x;
  w = zero;
  while (s < 10.0) {
    w += one / s;
    s += one;
  }

  y = digamma_impl_maybe_poly(s);

  y = logf(s) - (half / s) - y - w;

  return (negative) ? y - nz : y;
}

__BOTH__ double digamma_impl_dbl(const double u) {
  double x = u;
  double p, q, nz, s, w, y;
  bool negative;

  const double maxnum = FLT_MAX;
  const double m_pi = M_PI;

  negative = 0;
  nz = 0.0;

  const double zero = 0.0;
  const double one = 1.0;
  const double half = 0.5;

  if (x <= zero) {
    negative = one;
    q = x;
    p = floor(q);
    if (p == q) {
      return maxnum;
    }
    /* Remove the zeros of tan(m_pi x)
     * by subtracting the nearest integer from x
     */
    nz = q - p;
    if (nz != half) {
      if (nz > half) {
        p += one;
        nz = q - p;
      }
      nz = m_pi / tan(m_pi * nz);
    } else {
      nz = zero;
    }
    x = one - x;
  }

  /* use the recurrence psi(x+1) = psi(x) + 1/x. */
  s = x;
  w = zero;
  while (s < 10.0) {
    w += one / s;
    s += one;
  }

  y = digamma_impl_maybe_poly_dbl(s);

  y = log(s) - (half / s) - y - w;

  return (negative) ? y - nz : y;
}

__BOTH__ float polygamma_impl(int n, float x) {
  if (n == 0) {
    return digamma_impl(x);
  }

  // dumb code to calculate factorials
  float factorial = 1.0;
  for (int i = 0; i < n; i++) {
    factorial *= (i + 1);
  }

  return powf(-1.0, n + 1) * factorial * zeta_impl(n + 1, x);
}

__BOTH__ double polygamma_impl_dbl(int n, double x) {
  if (n == 0) {
    return digamma_impl_dbl(x);
  }

  // dumb code to calculate factorials
  double factorial = 1.0;
  for (int i = 0; i < n; i++) {
    factorial *= (i + 1);
  }
  
  return pow(-1.0, n + 1) * factorial * zeta_impl_dbl(n + 1, x);
  /* return ((n + 1 % 2) ? -1.0 : 1.0) * factorial * zeta_impl_dbl(n + 1, x); */
}

__BOTH__ double alnrel(double a) {
  const double p[] = {1.0, -.129418923021993E1, .405303492862024E0,
                      -.178874546012214E-1};
  const double q[] = {1.0, -.162752256355323E1, .747811014037616E0,
                      -.845104217945565E-1};
  if (fabs(a) > 0.375) {
    return (1.0 + log(a));
  } else {
    double t = a / (a + 2.0);
    double t2 = t * t;
    double w = polynomial_evaluation_dbl(t2, p, 4) /
               polynomial_evaluation_dbl(t2, q, 4);
    return 2 * t * w;
  }
}

/* double algdiv(double a, double b) { */
/*    */
/* } */
/*  */
__BOTH__ double algdiv(double a, double b) {
  double cp[] = {.833333333333333E-01, -.277777777760991E-02,
                 .793650666825390E-03, -.595202931351870E-03,
                 .837308034031215E-03, -.165322962780713E-02};
  double h, c, x, d;
  if (a <= b) {
    h = a / b;
    c = h / (1.0 + h);
    x = 1.0 / (1.0 + h);
    d = b + (a - 0.5);
  } else {
    h = b / a;
    c = 1.0 / (1.0 + h);
    x = h / (1.0 + h);
    d = a + (b - 0.5);
  }
  double x2 = x * x;
  double s3 = 1.0E0 + (x + x2);
  double s5 = 1.0E0 + (x + x2 * s3);
  double s7 = 1.0E0 + (x + x2 * s5);
  double s9 = 1.0E0 + (x + x2 * s7);
  double s11 = 1.0E0 + (x + x2 * s9);

  double t = (1.0 / b) * (1.0 / b);
  double w =
      ((((cp[5] * s11 * t + cp[4] * s9) * t + cp[3] * s7) * t + cp[2] * s5) *
           t +
       cp[1] * s3) *
          t +
      cp[0];
  w = w * (c / b);

  double u = d * alnrel(a / b);
  double v = a * (log(b) - 1);
  if (u <= v) {
    return (w - u) - v;
  } else {
    return (w - v) - u;
  }
}

__BOTH__ float lbeta_impl(float a, float b) {
  a = fabsf(a);
  b = fabsf(b);
  return lgammaf(a) + lgammaf(b) - lgammaf(a + b);
  /* if (a >= 1.0) { */
  /*   if (a >= 2.0) { */
  /*     float n = a - 1.0; */
  /*     float w = 1.0; */
  /*     float h = 0.0; */
  /*     for (int i = 0; i < n; i++) { */
  /*       a = a - 1.0; */
  /*       h = a / b; */
  /*       w = w * (h / (1.0 + h)); */
  /*     } */
  /*     w = logf(w); */
  /*     if (b < 8) { */
  /*       float n = b - 1.0; */
  /*       float z = 1.0; */
  /*       for (int i = 0; i < n; i++) { */
  /*         b = b - 1.0; */
  /*         z = z * (b / (a + b)); */
  /*       } */
  /*       return w + logf(z) + (lgammaf(a) + (lgammaf(a) - lgammaf(a + b))); */
  /*     } else { */
  /*       return w + lgamma(b) + algdiv((double)a, (double)b); */
  /*     } */
  /*   } else if (b >= 2.0) { */
  /*     if (b < 8.0) { */
  /*       float n = b - 1.0; */
  /*       float z = 1.0; */
  /*       for (int i = 0; i < n; i++) { */
  /*         b = b - 1.0; */
  /*         z = z * (b / (a + b)); */
  /*       } */
  /*       return logf(z) + (lgammaf(a) + (lgammaf(b) - lgammaf(a + b))); */
  /*     } else */
  /*       return lgammaf(a) + algdiv((double)a, (double)b); */
  /*   } */
  /*   printf("using lgammaf\n"); */
  /*   return lgammaf(a) + lgammaf(b) - lgammaf(a + b); */
  /* } else { */
  /*   if (b >= 8.0) { */
  /*     return lgammaf(a) + algdiv((double)a, (double)b); */
  /*   } */
  /*   return lgammaf(a) + (lgammaf(b) - lgammaf(a + b)); */
  /* } */
}

__BOTH__ double lbeta_impl_dbl(double a, double b) {
  a = fabs(a);
  b = fabs(b);
  return lgamma(a) + lgamma(b) - lgamma(a + b);
}

/* double lbeta_impl_dbl(double a, double b) { */
/*   a = fabs(a); */
/*   b = fabs(b); */
/*   if (a >= 1.0) { */
/*     if (a >= 2.0) { */
/*       double n = a - 1.0; */
/*       double w = 1.0; */
/*       double h = 0.0; */
/*       for (int i = 0; i < n; i++) { */
/*         a = a - 1.0; */
/*         h = a / b; */
/*         w = w * (h / (1.0 + h)); */
/*       } */
/*       w = log(w); */
/*       if (b < 8) { */
/*         double n = b - 1.0; */
/*         double z = 1.0; */
/*         for (int i = 0; i < n; i++) { */
/*           b = b - 1.0; */
/*           z = z * (b / (a + b)); */
/*         } */
/*         return 10.0; */
/*         printf("a\n"); */
/*         return w + log(z) + (lgamma(a) + (lgamma(a) - lgamma(a + b))); */
/*       } else { */
/*         printf("b\n"); */
/*         return w + lgamma(b) + algdiv(a, b); */
/*       } */
/*     } else if (b >= 2.0) { */
/*       if (b < 8.0) { */
/*         double n = b - 1.0; */
/*         double z = 1.0; */
/*         for (int i = 0; i < n; i++) { */
/*           b = b - 1.0; */
/*           z = z * (b / (a + b)); */
/*         } */
/*         printf("c\n"); */
/*         return log(z) + (lgamma(a) + (lgamma(b) - lgamma(a + b))); */
/*       } */
/*       printf("d\n"); */
/*       return lgamma(a) + algdiv(a, b); */
/*     } */
/*     printf("using lgamma\n"); */
/*     return lgamma(a) + lgamma(b) - lgamma(a + b); */
/*   } else { */
/*     if (b >= 8.0) { */
/*       printf("e\n"); */
/*       return lgamma(a) + algdiv(a, b); */
/*     } */
/*     printf("f\n"); */
/*     return lgamma(a) + (lgamma(b) - lgamma(a + b)); */
/*   } */
/* } */

