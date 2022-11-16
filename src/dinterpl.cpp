#include "dinterpl.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <immintrin.h>
#include <mkl.h>
#include <mkl_lapacke.h>
#include <cmath>

/*
  Class Constructor / Destructor
*/

dinterpl::dinterpl(const double x_array[], const double y_array[], size_t size){

  using namespace std;

	/*
		Copy data for use
	*/
  this->x_array = (double*) _mm_malloc(sizeof(double)*size,64);
  this->y_array = (double*) _mm_malloc(sizeof(double)*size,64);
  this->dydx_array = (double*) _mm_malloc(sizeof(double)*(size-1),64);

  memcpy( this->x_array, x_array, sizeof(double)*size);
  memcpy( this->y_array, y_array, sizeof(double)*size);
  
  this->size = size;
  
  for (int i = 0; i < size-1 ; i++){
  	dydx_array[i] = (y_array[i+1]-y_array[i])/(x_array[i+1]-x_array[i]);
  }
  
  //Initialize cubic spline
  cspline_init(this->x_array, this->y_array, size);
}

dinterpl::~dinterpl(){
  free_memory();
}

void dinterpl::free_memory()
{
  _mm_free(this->x_array);
  _mm_free(this->y_array);
  _mm_free(this->cspline_a);
  _mm_free(this->cspline_b);
  _mm_free(this->cspline_c);
  _mm_free(this->cspline_d);
}


/*
  Initialize Cubic Spline Interpolation
*/

void dinterpl::cspline_init(const double x_array[], const double y_array[], size_t size){

  size_t N = size;

  //Allocate memory for cubic spline
  cspline_a = (double*) _mm_malloc(sizeof(double)*(N-1),64);
  cspline_b = (double*) _mm_malloc(sizeof(double)*(N-1),64);
  cspline_c = (double*) _mm_malloc(sizeof(double)*(N-1),64);
  cspline_d = (double*) _mm_malloc(sizeof(double)*(N-1),64);
  
  //Allocate memory for spline evaluation
  double* X = (double*) _mm_malloc(sizeof(double)*N,64);
  double* dl = (double*) _mm_malloc(sizeof(double)*(N-1),64);
  double* du = (double*) _mm_malloc(sizeof(double)*(N-1),64);
  double* d = (double*) _mm_malloc(sizeof(double)*N,64);
  double* du2 = (double*) _mm_malloc(sizeof(double)*(N-2),64);//N-2 dimension

  //Compute X
  X[0]=3.0*(y_array[1]-y_array[0]);
  
  for (int i=1;i<N-1;i++){
    X[i]=3.0*(y_array[i+1]-y_array[i-1]);
  }
  X[N-1]=3.0*(y_array[N-1]-y_array[N-2]);
  
  //Get dl,du  //N-1 off diagonal points
  for (int i=0;i<N-1;i++){
    dl[i] = 1.0;
    du[i] = 1.0;
  }

  //get d //total of N diagonal points
  d[0]=2.0;
  d[N-1]=2.0;
  for (int i=1;i<N-1;i++){ 
    d[i] = 4.0;
  }

  lapack_int* ipiv = (lapack_int*) _mm_malloc(sizeof(lapack_int)*(N),64);

  LAPACKE_dgttrf (N , dl , d , du , du2 , ipiv );

  LAPACKE_dgttrs (LAPACK_ROW_MAJOR , 'N' , N , 1 , dl , d , du , du2 , ipiv , X , 1 );

  #pragma ivdep
  for (int i=0;i<N-1;i++){
    cspline_a[i]=y_array[i];
    cspline_b[i]=X[i];
    cspline_c[i]=3.0*(y_array[i+1]-y_array[i])-2.0*X[i]-X[i+1];
    cspline_d[i]=2.0*(y_array[i]-y_array[i+1])+X[i]+X[i+1];
  }

  _mm_free(X);
  _mm_free(dl);
  _mm_free(du);
  _mm_free(d);
  _mm_free(du2);
  _mm_free(ipiv);
}

void dinterpl::cspline_filter(double threshold){

  /*
    Cubic spline with cuts placed at discontinuities in first derivatives
  */

  size_t N = size;
  
  //double first_deltax = (x_array[1]-x_array[0]);
  //double last_deltax = (x_array[N]-x_array[N-1]);
  //double min_dx = (first_deltax>last_deltax) ? last_deltax : first_deltax; 
  
  double dx = 1e-3;// = min_dx < 1e-3 ? min_dx : 1e-3;
 
  //Compute first derivative
	double dy2dx2_last=0.0f;
  for (int i = 1 ; i < N-1 ; i++){
    //dydx_array[i] = (y_array[i+1] - y_array[i])/(x_array[i+1] - x_array[i]);
    const double x_hi = x_array[i]+dx;
    const double x_lo = x_array[i]-dx;
    const double dydx_hi = (linear_eval(x_hi+dx)-linear_eval(x_hi-dx))/(dx*2.0);
    const double dydx_lo = (linear_eval(x_lo+dx)-linear_eval(x_lo-dx))/(dx*2.0);
    const double dy2dx2 = (dydx_hi-dydx_lo)/(dx*2.0);
    if ( fabs(dy2dx2-dy2dx2_last) > threshold ){ //Use linear interpolation at discontinuity
      cspline_b[i-1] = y_array[i] - y_array[i-1]; 
      cspline_c[i-1] = 0.0; 
      cspline_d[i-1] = 0.0;

      cspline_b[i] = y_array[i+1] - y_array[i]; 
      cspline_c[i] = 0.0; 
      cspline_d[i] = 0.0;    
      //printf("Discontinuity at point %d | x = %f\n",i,x_array[i]);
    }
  	dy2dx2_last = dy2dx2;
  }
}

/*
  Evaluate linear interpolation
*/

//Regular version using binary search
double dinterpl::linear_fast_eval (double x){

  double y;
  size_t index;
  const double xmin = x_array[0];
  const double xmax = x_array[size-1];

  if (x > xmax or x < xmin) {
    y = 0.0;
  }
  else {

    index = interp_bsearch (x_array, x, 0, size - 1);

    /* evaluate */
    y = y_array[index] + (x-x_array[index])*dydx_array[index];
  }
  return y;
}

//Regular version using binary search (fastest)
double dinterpl::linear_eval (double x){

  double x_lo, x_hi;
  double y_lo, y_hi;
  double y;
  size_t index;
  const double xmin = x_array[0];
  const double xmax = x_array[size-1];

  if (x > xmax or x < xmin) {
    y = 0.0;
  }
  else {

    index = interp_bsearch (x_array, x, 0, size - 1);

    /* evaluate */
    x_lo = x_array[index];
    x_hi = x_array[index + 1];
    y_lo = y_array[index];
    y_hi = y_array[index + 1];

    y = y_lo + (x - x_lo) / (x_hi - x_lo) * (y_hi - y_lo);
  }
  return y;
}

//Version using internal cache
double dinterpl::linear_cached_eval (double x){

  double x_lo, x_hi;
  double y_lo, y_hi;
  double y;
  size_t index;
  const double xmin = x_array[0];
  const double xmax = x_array[size-1];
  static size_t internal_cache;

  if (x > xmax or x < xmin) {
    y = 0.0;
  }
  else {

    index = interp_accel_find (internal_cache, x_array, size, x);
    internal_cache = index;

    /* evaluate */
    x_lo = x_array[index];
    x_hi = x_array[index + 1];
    y_lo = y_array[index];
    y_hi = y_array[index + 1];

    y = y_lo + (x - x_lo) / (x_hi - x_lo) * (y_hi - y_lo);
  }
  return y;
}

/*
  Evaluate Cubic spline interpolation
*/

//Regular version using binary search (fastest)
double dinterpl::cspline_eval(double x){

  double x_lo, x_hi;
  double y_lo, y_hi;
  double t;
  double y;
  size_t index;
  double xmin = x_array[0];
  double xmax = x_array[size-1];

  if (x > xmax or x < xmin) {
    y = 0.0;
  }
  else {
    index = interp_bsearch (x_array, x, 0, size - 1);
    /* evaluate */
    x_lo = x_array[index];
    x_hi = x_array[index + 1];
    t = (x-x_lo)/(x_hi-x_lo);

    y = cspline_eval_at_knot(index, t,cspline_a,cspline_b,cspline_c,cspline_d);
  }
  
  return y;
}

//Version using internal cache
double dinterpl::cspline_cached_eval(double x){

  double x_lo, x_hi;
  double y_lo, y_hi;
  double t;
  double y;
  size_t index;
  double xmin = x_array[0];
  double xmax = x_array[size-1];
  static size_t internal_cache;

  if (x > xmax or x < xmin) {
    y = 0.0;
  }
  else {
    index = interp_accel_find (internal_cache, x_array, size, x);
    internal_cache = index;
  
    /* evaluate */
    x_lo = x_array[index];
    x_hi = x_array[index + 1];
    t = (x-x_lo)/(x_hi-x_lo);

    y = cspline_eval_at_knot(index, t,cspline_a,cspline_b,cspline_c,cspline_d);
  }
  
  return y;
}


/*
  Static Version of linear interpolation
*/

double dinterpl::linear_eval (double x, const double x_array[], const double y_array[], size_t size){

  double x_lo, x_hi;
  double y_lo, y_hi;
  double y;
  size_t index;
  const double xmin = x_array[0];
  const double xmax = x_array[size-1];

  if (x > xmax or x < xmin) {
    y = 0.0;
  }
  else {
    index = interp_bsearch (x_array, x, 0, size - 1);

    /* evaluate */
    x_lo = x_array[index];
    x_hi = x_array[index + 1];
    y_lo = y_array[index];
    y_hi = y_array[index + 1];

    y = y_lo + (x - x_lo) / (x_hi - x_lo) * (y_hi - y_lo);
  }
  return y;
}

double dinterpl::linear_cached_eval (double x, const double x_array[], const double y_array[], size_t size){

  double x_lo, x_hi;
  double y_lo, y_hi;
  double y;
  size_t index;
  const double xmin = x_array[0];
  const double xmax = x_array[size-1];
  static size_t internal_cache;

  if (x > xmax or x < xmin) {
    y = 0.0;
  }
  else {

    index = interp_accel_find (internal_cache, x_array, size, x);
    internal_cache = index;

    /* evaluate */
    x_lo = x_array[index];
    x_hi = x_array[index + 1];
    y_lo = y_array[index];
    y_hi = y_array[index + 1];

    y = y_lo + (x - x_lo) / (x_hi - x_lo) * (y_hi - y_lo);
  }
  return y;
}

/*
  Misc functions
*/

void dinterpl::uniformize_grid(size_t s_size){
  /*
    process to create a new uniform grid of data using linear interpolation
    note: replaces the grid stored in class object with the new one and this can cause loss of information
  */  
  
  double* sx_array;
  double* sy_array;
  sx_array = (double*) _mm_malloc(sizeof(double)*s_size,64);
  sy_array = (double*) _mm_malloc(sizeof(double)*s_size,64);
  
  double xmin = x_array[0];
  double xmax = x_array[size-1];
  double xrange = xmax-xmin;
  double dx = ((double) s_size-1.0);
  
  for (int i=0;i<s_size;i++){
    double x = xmin+ xrange*((double) i)/dx;
    sx_array[i] = x;
    sy_array[i] = linear_eval (x);
  }
  
  
  //Free old grid and copy new grid to the class object
  free_memory();
	
  this->x_array = (double*) _mm_malloc(sizeof(double)*s_size,64);
  this->y_array = (double*) _mm_malloc(sizeof(double)*s_size,64);
  
  memcpy( this->x_array, sx_array, sizeof(double)*s_size);
  memcpy( this->y_array, sy_array, sizeof(double)*s_size);
  
  this->size = s_size;
  
  _mm_free(sx_array);
  _mm_free(sy_array);
  
  //Re-initialize the cubic spline
  cspline_init(this->x_array, this->y_array, this->size);
}
