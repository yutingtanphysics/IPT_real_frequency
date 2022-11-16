#include <cstdlib>
#include <cstring>

using namespace std;

class dinterpl{

  public:

    //Class constructor/destructor
    dinterpl(const double x_array[], const double y_array[], size_t size);
    ~dinterpl();
    
    //Interpolation Evaluation
    double linear_eval(double x);
    double linear_fast_eval (double x);
    double cspline_eval(double x);

    //Cached version
    double linear_cached_eval(double x);
    double cspline_cached_eval(double x);

    //Static Version of linear interpolation
    static double linear_eval (double x, const double x_array[], const double y_array[], size_t size);
    static double linear_cached_eval (double x, const double x_array[], const double y_array[], size_t size);
    
    //Misc Routines
    void cspline_filter(double threshold);
    void uniformize_grid(size_t s_size);
    
    //Moved to public (so we can access them)
    double* cspline_a;
    double* cspline_b;
    double* cspline_c;
    double* cspline_d;
    

  private:
  
    //Arrays stored in memory (must free them with destructor)
    double* x_array;
    double* y_array;
    double* dydx_array;

    size_t size;

    void cspline_init(const double x_array[], const double y_array[], size_t size);

    /*
      Inline search fuctions
    */
    inline static size_t interp_bsearch (const double x_array[], double x, size_t index_lo, size_t index_hi) 
    {
      size_t min = index_lo, max = index_hi;
      while (min + 1 < max)
      {
        size_t i = (min + max) >> 1;
        min = x > x_array[i] ? i : min;
        max = x > x_array[i] ? max : i;
      }
      return min;
    }

    inline static size_t interp_accel_find(size_t internal_cache, const double xa[], size_t len, double x)
    {
      size_t x_index = internal_cache;

      if(x < xa[x_index]) {
        internal_cache = interp_bsearch(xa, x, 0, x_index);
      }
      else if(x >= xa[x_index + 1]) {
        internal_cache = interp_bsearch(xa, x, x_index, len-1);
      }

      return internal_cache;
    }
    
    inline static double cspline_eval_at_knot(size_t i, double t
                                              ,const double cspline_a[],const double cspline_b[]
                                              ,const double cspline_c[],const double cspline_d[])
    {
      return cspline_a[i]+cspline_b[i]*t+cspline_c[i]*t*t+cspline_d[i]*t*t*t;
    }
    
    void free_memory();

};

