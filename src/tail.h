/*
	Extrapolation of hybridization function tail
*/

struct fit_params
{
  size_t max_iter; //Cap on number of iterations for curve fit
  size_t ntail; // number of points on the tail to fit
  size_t p; // fit order
  double xtol; // xtol for lm
  double gtol; // gtol for lm
  double ftol; // ftol for lm
  bool verbose; // verbose output
  bool quiet; // do not produce any output
  void setdefault(){
  	max_iter=20;
  	ntail=200;
  	p = 4;
  	xtol=1e-14;
  	gtol=1e-14;
  	ftol=1e-14;
  	verbose=false;
  	quiet=false;
  }
};

int fit_tail(double* omega,std::complex<double>* Delta, size_t N,double *L,double *R,size_t p, struct fit_params * params);


