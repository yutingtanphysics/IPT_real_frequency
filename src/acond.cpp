#include <stdlib.h>
#include <argp.h>
#include <cmath>
#include <stdio.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlinear.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "routines.h"

const char *argp_program_version =
  "Real-to-Matsubara Analytic Continuation 1.0";
const char *argp_program_bug_address =
  "<henrytsang222@gmail.com>";

/* Program documentation. */
static char doc[] =
  "Analytic Continuation for complex functions defined on the real axis to Matsubara frequencies";

/* A description of the arguments we accept. */
static char args_doc[] = "FUNCTION TEMPERATURE";

/* The options we understand. */
static struct argp_option options[] = {
  {"verbose",  'v', 0,      0,  "Produce verbose output" },
  {"quiet",    'q', 0,      0,  "Don't produce any output" },
  {"silent",   's', 0,      OPTION_ALIAS },
  {"ntail",   't', "COUNT", 0, "Number of points to sample on the tail" },
  {"output",   'o', "FILE", 0,
   "Output to FILE instead of standard output" },
  { 0 }
};

/* Used by main to communicate with parse_opt. */
struct arguments
{
  char *args[3];                /* arg1 & arg2 */
  int silent, verbose;
  int ntail,nmat;
  char *output_file;
};

/* Parse a single option. */
static error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  /* Get the input argument from argp_parse, which we
     know is a pointer to our arguments structure. */
  struct arguments *arguments = (struct arguments*)(state)->input;

  switch (key)
    {
    case 'q': case 's':
      arguments->silent = 1;
      break;
    case 'v':
      arguments->verbose = 1;
      break;
    case 'o':
      arguments->output_file = arg;
      break;
    case 't':
      arguments->ntail = arg ? atoi (arg) : 200;
      break;

    case ARGP_KEY_ARG:
      if (state->arg_num >= 3)
        /* Too many arguments. */
        argp_usage (state);

      arguments->args[state->arg_num] = arg;

      break;

    case ARGP_KEY_END:
      if (state->arg_num < 3)
        /* Not enough arguments. */
        argp_usage (state);
      break;

    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

/* Our argp parser. */
static struct argp argp = { options, parse_opt, args_doc, doc };

struct fit_data
{
  double* w;
  double* y;
  size_t n;
};

struct fit_params
{
  size_t max_iter;
  double xtol;
  double gtol;
  double ftol;
  bool verbose;
  bool quiet;
};

/* model function: C+A/(B+w) asymptotic behavior of reSigma */
double reSigma(const double A, const double B, const double C, const double w)
{
  return C+A/(w+B);
}


int func_Sigmatail (const gsl_vector * x, void *params, gsl_vector * f)
{
  struct fit_data *d = (struct fit_data *) params;
  double A = gsl_vector_get(x, 0);
  double B = gsl_vector_get(x, 1);
  double C = gsl_vector_get(x, 2);

  for (int i = 0; i < d->n; ++i)
  {
    double wi = d->w[i];
    double yi = d->y[i];
    double y = reSigma(A,B,C,wi);

    gsl_vector_set(f, i, yi - y);
  }

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
  //(void) params; /* not used */

  /* compute reciprocal condition number of J(x) */
  gsl_multifit_nlinear_rcond(&rcond, w);

	
	if (verbose && !quiet){
  fprintf(stderr, "iter %2zu: A = %.4f, B = %.4f, C = %.4f, |A|/|v| = %.4f cond(J) = %8.4f, |f(x)| = %.4f\n",
          iter,
          gsl_vector_get(x, 0),
          gsl_vector_get(x, 1),
          gsl_vector_get(x, 2),
          avratio,
          1.0 / rcond,
          gsl_blas_dnrm2(f));
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
  gsl_multifit_nlinear_workspace *work =
    gsl_multifit_nlinear_alloc(T, params, n, p);
  gsl_vector * f = gsl_multifit_nlinear_residual(work);
  gsl_vector * y = gsl_multifit_nlinear_position(work);
  int info;
  double chisq0, chisq, rcond;
	
	
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
  fprintf(stderr, "final x       = (%.12e, %.12e)\n",
          gsl_vector_get(x, 0), gsl_vector_get(x, 1));
  fprintf(stderr, "final cond(J) = %.12e\n", 1.0 / rcond);
	}
  gsl_multifit_nlinear_free(work);
}

int
main (int argc, char **argv)
{
  struct arguments arguments;

  /* Default values. */
  arguments.silent = 0;
  arguments.verbose = 0;
  arguments.ntail = 200;
  arguments.output_file = "Sig.mat.out";


  /* Parse our arguments; every option seen by parse_opt will
     be reflected in arguments. */
  argp_parse (&argp, argc, argv, 0, 0, &arguments);
       
  int N;
  complex<double>* Sigma;
  double* omega;
  char * FNAME = arguments.args[0];
  double T = atof(arguments.args[1]);
  int nmat = atoi(arguments.args[2]);
  bool quiet = arguments.silent;
  bool verbose  =arguments.verbose;
  
	if (verbose && !quiet) 
  printf ("INPUT = %s\nOUTPUT_FILE = %s\n"
  				"T = %f\n"
          "VERBOSE = %s\nSILENT = %s\n",
          arguments.args[0],
          arguments.output_file,
          T,
          arguments.verbose ? "yes" : "no",
          arguments.silent ? "yes" : "no");
  
  if (verbose && !quiet) printf("=== Reading Input File ===\n");
  
		
	ReadFunc(FNAME, N,Sigma,omega);
	
  if (verbose && !quiet) printf("=== Finish Reading Input File , N:%d ===\n",N);
  
  if (verbose && !quiet) printf("=== Fitting the tail ===\n");
  
  const size_t n = arguments.ntail;  /* number of data points to fit */
  const size_t p = 3;    /* number of model parameters */
  
  gsl_vector *f = gsl_vector_alloc(n);
  gsl_vector *x = gsl_vector_alloc(p);
  
  gsl_multifit_nlinear_fdf fdf;
  gsl_multifit_nlinear_parameters fdf_params =
    gsl_multifit_nlinear_default_parameters();
    
  struct fit_data fit_data;
  
	double* w = new double[n];
	double* y = new double[n];
  
  for (int i = 0; i < n/2; ++i)
  {
      w[i] = omega[i];
      y[i] = real(Sigma[i]);
  }
  
  for (int i = 0; i < n/2; ++i)
  {
      w[i+n/2] = omega[N-1-i];
      y[i+n/2] = real(Sigma[N-1-i]);
  }
  fit_data.w = w;
  fit_data.y = y;
  fit_data.n = n;
    
  /* define function to be minimized */
  fdf.f = func_Sigmatail;
  fdf.df = NULL;
  fdf.fvv = NULL;
  fdf.n = n;
  fdf.p = p;
  fdf.params = &fit_data;

  /* starting point */
  gsl_vector_set(x, 0, 1.0);
  gsl_vector_set(x, 1, 1.0);
  gsl_vector_set(x, 2, 1.0);
  
  struct fit_params fparams;
  fparams.max_iter = 10000;
  fparams.xtol = 1e-14;
  fparams.gtol = 1e-14;
  fparams.ftol = 1e-14;
  fparams.verbose = verbose;
  fparams.quiet = quiet;

  fdf_params.trs = gsl_multifit_nlinear_trs_lmaccel;
  solve_system(x, &fdf, &fdf_params, &fparams);
  
  
  /* print data and model */
  /*
  {
    double A = gsl_vector_get(x, 0);
    double B = gsl_vector_get(x, 1);
    double C = gsl_vector_get(x, 2);
    
    for (i = 0; i < n; ++i)
      {
        double wi = fit_data.w[i];
        double yi = fit_data.y[i];
        double fi = reSigma(A, B,C, wi);

        printf("%f %f %f\n", wi, yi, fi);
      }
  }
	*/
  gsl_vector_free(f);
  gsl_vector_free(x);
  
  if (verbose && !quiet) printf("=== Finished Fitting the tail ===\n");
  
  double Sigmainf = gsl_vector_get(x, 2);
  
  
  complex<double> * Sigmamat = new complex<double>[nmat];
  
  if (verbose && !quiet) printf("Const term = %f\n",Sigmainf);
  
  if (verbose && !quiet) printf("=== Begin analytic continuation ===\n");
  
  #pragma omp parallel for
  for (int i=0;i<nmat;i++){
  	complex<double> *integrand = new complex<double>[N];
		double wmat = (2*i+1)*M_PI*T;
		for (int j=0;j<N;j++)	integrand[j] = imag(Sigma[j])
																				/(omega[j]-complex<double>(0.0,wmat))
																				/M_PI;
		Sigmamat[i] = TrapezIntegral(N, integrand, omega)+Sigmainf;
  	delete [] integrand;
	}
	
	
  if (verbose && !quiet) printf("=== Finished analytic continuation ===\n");
	
  if (verbose && !quiet) printf("=== Printing results to file ===\n");
	{
	FILE * out;
  out = fopen(arguments.output_file, "w");
	for (int i=0;i<nmat;i++){
		fprintf(out,"%lf %lf %lf\n",(2*i+1)*M_PI*T,real(Sigmamat[i]),imag(Sigmamat[i]));
	}
  fclose(out);
	}
  if (verbose && !quiet) printf("=== Finished Printing results to file ===\n");
  
  delete [] Sigmamat;
  delete [] omega;
  delete [] Sigma;
  delete [] w;
  delete [] y;

  return 0;
}
