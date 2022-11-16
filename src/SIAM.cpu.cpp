/*
	IPT Anderson Impurity Solver CPU-only code
	Adapted from code by Jaksha Vuchichevicc https://github.com/JaksaVucicevic/DMFT
*/

#include "SIAM.h"
#include "routines.h"
#include "Grid.h"
#include "dinterpl.h"

//GSL Libraries for Adaptive Cauchy Integration
#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>

using namespace std;

struct KK_params
    {
      double* omega;
      double* Y;
      int N;
      dinterpl* spline;
    };

double imSOCSigmafc(double om, void *params);

double imSOCSigmafl(double om, void *params);

double imSOCSigmafc(double om, void *params)
{
  struct KK_params *p= (struct KK_params *)params;
  return p->spline->cspline_eval(om);
}

double imSOCSigmafl(double om, void *params)
{
  struct KK_params *p= (struct KK_params *)params;
  return dinterpl::linear_eval(om,p->omega, p->Y , p->N);
}

void SIAM::get_Ps()
{
	if (!usecubicspline){
		#pragma omp parallel for
		for (int i=0; i<N; i++) 
		{ 
		    double *p1 = new double[N];
		    double *p2 = new double[N];
		    #pragma ivdep
		    for (int j=0; j<N; j++)
		    {  
		       p1[j] = g->Am[j] * dinterpl::linear_eval(g->omega[j] - g->omega[i],g->omega,g->Ap,N);
		       p2[j] = g->Ap[j] * dinterpl::linear_eval(g->omega[j] - g->omega[i],g->omega,g->Am,N);
		    }

		    //get Ps by integrating                           
		    g->P1[i] = M_PI * TrapezIntegral(N, p1, g->omega);
		    g->P2[i] = M_PI * TrapezIntegral(N, p2, g->omega);

		    delete [] p1;
		    delete [] p2;
		}
  }
  else{
		#pragma omp parallel for
		for (int i=0; i<N; i++) 
		{ 
		    double *p1 = new double[N];
		    double *p2 = new double[N];
		    #pragma ivdep
		    for (int j=0; j<N; j++)
		    {  
		       p1[j] = g->Am[j] * Ap_spline->cspline_eval(g->omega[j] - g->omega[i]);
		       p2[j] = g->Ap[j] * Am_spline->cspline_eval(g->omega[j] - g->omega[i]);
		    }

		    //get Ps by integrating                           
		    g->P1[i] = M_PI * TrapezIntegral(N, p1, g->omega);
		    g->P2[i] = M_PI * TrapezIntegral(N, p2, g->omega);

		    delete [] p1;
		    delete [] p2;
		}
  
  }
}

void SIAM::get_SOCSigma()
{
  double *imSOCSigma = new double[N];
	if (!usecubicspline){
		#pragma omp parallel for
		for (int i=0; i<N; i++) 
		{ 
		  double *s = new double[N];
		  #pragma ivdep
		  for (int j=0; j<N; j++) 
		  {  
		     s[j] =   dinterpl::linear_eval(g->omega[i] - g->omega[j],g->omega,g->Ap,N) * g->P2[j] 
		               + dinterpl::linear_eval(g->omega[i] - g->omega[j],g->omega,g->Am,N) * g->P1[j];
		  }
		                     
		  //integrate 
		  imSOCSigma[i] = - U*U * TrapezIntegral(N, s, g->omega);
		  if (ClipOff( imSOCSigma[i] )) Clipped = true ;
		  delete [] s;
		}
	}
	else{
		#pragma omp parallel for
		for (int i=0; i<N; i++) 
		{ 
		  double *s = new double[N];
		  #pragma ivdep
		  for (int j=0; j<N; j++) 
		  {  
		     s[j] =   Ap_spline->cspline_eval(g->omega[i] - g->omega[j]) * g->P2[j] 
		               + Am_spline->cspline_eval(g->omega[i] - g->omega[j]) * g->P1[j];
		  }
		                     
		  //integrate 
		  imSOCSigma[i] = - U*U * TrapezIntegral(N, s, g->omega);
		  if (ClipOff( imSOCSigma[i] )) Clipped = true ;
		  delete [] s;
		}	
	}
  if (Clipped) sprintf(ibuffer + strlen(ibuffer),"SIAM::run::(Warning) !!!Clipping SOCSigma!!!!\n");  
  

  // The KramersKonig function is not used to compute the Cauchy Integral, this allows arbitrary grid to be used.
  
  
  if (usecubicspline) imSOCSigmaspline = new dinterpl(g->omega, imSOCSigma , N);
  gsl_set_error_handler_off();
  #pragma omp parallel for schedule(dynamic)
  for (int i=1; i<N-1; i++)
  { 
    const double a = g->omega[0], b = g->omega[N-1]; // limits of integration
    const double epsabs = 0, epsrel = KKAccr; // requested errors
    double result; // the integral value
    double error; // the error estimate

    double c = g->omega[i];

    struct KK_params params;
    gsl_function F;
    
    if (!usecubicspline){
		  params.omega = g->omega;
		  params.Y = imSOCSigma;
		  params.N = N;
    	F.function = &imSOCSigmafl;
    }
    else{
    	params.spline = imSOCSigmaspline;
    	F.function = &imSOCSigmafc;
    }

    //F.function = &imSOCSigmafc;
    F.params = &params;

    
    size_t limit = QUADLIMIT;// work area size
    gsl_integration_workspace *ws = gsl_integration_workspace_alloc (limit);

    gsl_integration_qawc (&F, a, b , c , epsabs, epsrel, limit, ws, &result, &error);

    gsl_integration_workspace_free (ws);

    g->SOCSigma[i] = complex<double>(result/M_PI,imSOCSigma[i]);
  }
  
  g->SOCSigma[0] = g->SOCSigma[1];
  g->SOCSigma[N-1] = g->SOCSigma[N-2];

	if (usecubicspline) delete imSOCSigmaspline;
  delete [] imSOCSigma;
}
