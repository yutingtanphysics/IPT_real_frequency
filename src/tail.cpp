/*
	Extrapolation of hybridization function tail
*/

#include <stdlib.h>
#include <stdio.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlinear.h>
#include <complex>
#include "tail.h"



struct data
{
  double* w;
  double* y;
  size_t n;
  size_t p;
};

/* model function: A/(B+w) */
double reDeltatailfunc(const double LR[],int p, const double w)
{
	double res = 0.0;
	//res = LR[0]/pow(w,1)+LR[1]/pow(w,3)+LR[2]/pow(w,5)+LR[3]/pow(w,7);
	for (int i=0;i<p;i++) res += LR[i]/pow(w,2*i+1);
  return res;
}

int func_Deltatail (const gsl_vector * x, void *params, gsl_vector * f)
{
  struct data *d = (struct data *) params;
  size_t n = d->n;
  size_t p = d->p;
  
  double * LR = new double[p];
  //printf("p = %d\n",p);
  for (int i = 0; i < p; i++) {
  	LR[i] = gsl_vector_get(x, i);
  	//printf("LR[%d] = %f\n",i,LR[i]);
  }
  //double w = d->w[0];
	//printf("%f %f\n",w,1/pow(w,1)+1/pow(w,3)+1/pow(w,5)+1/pow(w,7));

  for (int i = 0; i < n; i++)
  {
    double wi = d->w[i];
    double yi = d->y[i];
    double y = reDeltatailfunc(LR,p,wi);
		//if (i==0) printf("%f %f %f %f\n",wi,yi,y,yi - y);
    gsl_vector_set(f, i, yi - y);
  }
	delete [] LR;
  return GSL_SUCCESS;
}

int dfunc_Deltatail (const gsl_vector * x, void *params, gsl_matrix * J)
{
  struct data *d = (struct data *) params;
  
  //double * LR = new double[d->p];
  //for (int i=0;i<d->p;i++) LR[i] = gsl_vector_get(x, i);

  for (int i = 0; i < d->n; i++) //#data
  {
    for (int j=0;j<d->p;j++){ //#fit order
      gsl_matrix_set (J, i, j, 1/pow(d->w[i],2*j+1));
    }
  }

	//delete [] LR;
  return GSL_SUCCESS;
}

void
callback(const size_t iter, void *params,
         const gsl_multifit_nlinear_workspace *w)
{
  gsl_vector *f = gsl_multifit_nlinear_residual(w);
  gsl_vector *x = gsl_multifit_nlinear_position(w);
  double avratio = gsl_multifit_nlinear_avratio(w);
  double rcond;

	struct fit_params *fp = (struct fit_params *) params;
	bool verbose = fp->verbose;
	bool quiet = fp->quiet;

  /* compute reciprocal condition number of J(x) */
  gsl_multifit_nlinear_rcond(&rcond, w);

	if (verbose && !quiet){
  fprintf(stderr, "iter %2zu:",iter);
  for (int i=0;i<fp->p;i++)
  fprintf(stderr, " L[%d] = %.4f,",i,gsl_vector_get(x, i));
  fprintf(stderr, " |A|/|v| = %.4f cond(J) = %8.4f, |f(x)| = %.4f\n",
          avratio,
          1.0 / rcond,
          gsl_blas_dnrm2(f)
          );
  }
}

void
solve_system(gsl_vector *x, gsl_multifit_nlinear_fdf *fdf,
             gsl_multifit_nlinear_parameters *params,struct fit_params * fparams)
{
  const gsl_multifit_nlinear_type *T = gsl_multifit_nlinear_trust;
  const size_t max_iter = fparams->max_iter;
  const double xtol = fparams->xtol;
  const double gtol = fparams->gtol;
  const double ftol = fparams->ftol;
  bool verbose = fparams->verbose;
  bool quiet = fparams->quiet;
  const size_t n = fdf->n;
  const size_t p = fdf->p;
  
  //params->factor_up = 10.0;
  
  //params->factor_down = 5.0;
  params->trs = gsl_multifit_nlinear_trs_lm;
  //params->trs = gsl_multifit_nlinear_trs_dogleg;
  //params->scale = gsl_multifit_nlinear_scale_levenberg;
  //printf("factor_up : %f\n",params->factor_up);
  //printf("factor_down : %f\n",params->factor_down);
  //printf("avmax : %f\n",params->avmax);
  
  
  gsl_multifit_nlinear_workspace *work =
    gsl_multifit_nlinear_alloc(T, params, n, p);
  gsl_vector * f = gsl_multifit_nlinear_residual(work);
  gsl_vector * y = gsl_multifit_nlinear_position(work);
  int info;
  double chisq0, chisq, rcond;
  
	//printf ("work is a '%s' solver\n", gsl_multifit_nlinear_name (work));
	//printf ("work is a '%s' solver\n", gsl_multifit_nlinear_trs_name (work));

	
	gsl_set_error_handler_off();
	
	
  /* initialize solver */
  gsl_multifit_nlinear_init(x, fdf, work);

  /* store initial cost */
  gsl_blas_ddot(f, f, &chisq0);

  /* iterate until convergence */
  gsl_multifit_nlinear_driver(max_iter, xtol, gtol, ftol,
                              callback, fparams, &info, work);

  /* store final cost */
  gsl_blas_ddot(f, f, &chisq);

  /* store cond(J(x)) */
  gsl_multifit_nlinear_rcond(&rcond, work);

  gsl_vector_memcpy(x, y);

  /* print summary */

	if (verbose && !quiet){
  fprintf(stderr, "NITER         = %zu\n", gsl_multifit_nlinear_niter(work));
  fprintf(stderr, "NFEV          = %zu\n", fdf->nevalf);
  fprintf(stderr, "NJEV          = %zu\n", fdf->nevaldf);
  fprintf(stderr, "NAEV          = %zu\n", fdf->nevalfvv);
  fprintf(stderr, "initial cost  = %.12e\n", chisq0);
  fprintf(stderr, "final cost    = %.12e\n", chisq);
  fprintf(stderr, "final x       = (");
  for (int i=0;i<p;i++)
  fprintf(stderr, "%.12e,",gsl_vector_get(x, i));
  fprintf(stderr,")\n");
  fprintf(stderr, "final cond(J) = %.12e\n", 1.0 / rcond);
  }

  gsl_multifit_nlinear_free(work);
}

int fit_tail(double* omega,std::complex<double>* Delta, size_t N,double *L,double *R,size_t p, struct fit_params * params)
{

  const size_t n = params->ntail;  /* number of data points to fit */
  //const size_t p = params->p;    /* number of model parameters */
  //printf("N : %lu , p : %lu\n",N,p);
  gsl_vector *f1 = gsl_vector_alloc(n);
  gsl_vector *x1 = gsl_vector_alloc(p);
  gsl_vector *f2 = gsl_vector_alloc(n);
  gsl_vector *x2 = gsl_vector_alloc(p);
  
  gsl_multifit_nlinear_fdf fdf1;
  gsl_multifit_nlinear_fdf fdf2;
  
  gsl_multifit_nlinear_parameters fdf_params1 =
    gsl_multifit_nlinear_default_parameters();
  
  gsl_multifit_nlinear_parameters fdf_params2 =
    gsl_multifit_nlinear_default_parameters();
    
  struct data fit_data1; //w,y,n,p
  struct data fit_data2; //w,y,n,p

	double* w1 = new double[n];
	double* y1 = new double[n];
	double* w2 = new double[n];
	double* y2 = new double[n];
  
  for (int i = 0; i < n; i++)
  {
  		const int i0=0;
      w1[i] = omega[i+i0];
      y1[i] = std::real(Delta[i+i0]);
      
      //printf("%f %f\n",w1[i],y1[i]);
  }
  
  for (int i = 0; i < n; i++)
  {
  		const int i0=0;
      w2[i] = omega[N-1-i-i0];
      y2[i] = std::real(Delta[N-1-i-i0]);
      //printf("%f %f\n",w2[i],y2[i]);
  }
  
  fit_data1.w = w1;
  fit_data1.y = y1;
  fit_data1.n = n;
  fit_data1.p = p;
  
  fit_data2.w = w2;
  fit_data2.y = y2;
  fit_data2.n = n;
  fit_data2.p = p;
    
  /* define function to be minimized */
  fdf1.f = func_Deltatail;
  fdf1.df = NULL; //dfunc_Deltatail;
  fdf1.fvv = NULL;
  fdf1.n = n;
  fdf1.p = p;
  fdf1.params = &fit_data1;
  
  fdf2.f = func_Deltatail;
  fdf2.df = NULL; //dfunc_Deltatail;
  fdf2.fvv = NULL;
  fdf2.n = n;
  fdf2.p = p;
  fdf2.params = &fit_data2;

  /* starting point */
  for (int i=0;i<p;i++) gsl_vector_set(x1, i, 1.0);
  for (int i=0;i<p;i++) gsl_vector_set(x2, i, 1.0);

  //fdf_params1.trs = gsl_multifit_nlinear_trs_lm;
  //fdf_params2.trs = gsl_multifit_nlinear_trs_lm;
  solve_system(x1, &fdf1, &fdf_params1,params);
  solve_system(x2, &fdf2, &fdf_params2,params);
  
  /* print data and model */
  for (int i=0;i<p;i++) L[i] = gsl_vector_get(x1, i);
  for (int i=0;i<p;i++) R[i] = gsl_vector_get(x2, i);
  
  gsl_vector_free(f1);
  gsl_vector_free(x1);
  gsl_vector_free(f2);
  gsl_vector_free(x2);
  
  delete [] w1;
  delete [] y1;
  delete [] w2;
  delete [] y2;

  return 0;
}
