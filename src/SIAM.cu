/*
	IPT Anderson Impurity Solver GPU code for CUDA devices
*/



//Math Libraries
#include <complex>
#include <cmath>
#include <cfloat>

//CUDA libraries
#include <cuda.h>
#include <cuda_runtime.h>

//OpenMP
#include <omp.h>

//IPT Solver
#include "SIAM.h"
#include "Grid.h"
#include "dinterpl.h"

//GSL Libraries for Adaptive Cauchy Integration
#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>


// Device constants header //
//__constant__ <type> <variable>;

// Constants
const int MAX_GPU_COUNT=8;
const int CUDA_threadcount = 250;

// Typedef

// Structs

// Interpolation Function
template<typename T> __device__ int d_interp_bsearch (const T x_array[], T x, int index_lo, int index_hi);
template<typename T> __device__ T d_interpl(T x, const T y_array[],  const T x_array[], int size);
template<typename T> __device__ T d_cspline_eval(double x, const T cspline_a[], const T cspline_b[], const T cspline_c[], const T cspline_d[],  const T x_array[], int size);

struct KK_params
    {
      double* omega;
      double* Y;
      int N;
      dinterpl* spline;
    };

template<typename T> struct TGPUplan
{
    //Host-side input data
    int N_LOCAL,i0;
    int block_count;
    
    //Host buffers
    T *h_P1,*h_P2,*h_imSOCSigma;

    //Device buffers
    T *d_omega,*d_Ap,*d_Am,*d_P1,*d_P2,*d_imSOCSigma;
    
    T *d_Ap_spline_a, *d_Ap_spline_b, *d_Ap_spline_c, *d_Ap_spline_d;
    T *d_Am_spline_a, *d_Am_spline_b, *d_Am_spline_c, *d_Am_spline_d;

    //Stream for asynchronous command execution
    cudaStream_t stream;

} ;

/* GPU Function Headers */

// Polarization Functions
template<typename T> __global__ void d_get_Ps(T P1[], T P2[], const T Ap[], const T Am[], const T omega[], int i0, int N_LOCAL, int N);

template<typename T> __global__ void d_get_Psc(T P1[], T P2[]
, const T Ap[], const T Am[]
, const T Ap_spline_a[], const T Ap_spline_b[], const T Ap_spline_c[], const T Ap_spline_d[]
, const T Am_spline_a[], const T Am_spline_b[], const T Am_spline_c[], const T Am_spline_d[]
, const T omega[], int i0, int N_LOCAL, int N);

// Imaginary Part of Second-order correction to self energy
template<typename T> __global__ void d_get_imSOCSigma(T imSOCSigma[],const T P1[], const T P2[], const T Ap[] , const T Am[] , const T omega[], T U, int i0, int N_LOCAL, int N);

template<typename T> __global__ void d_get_imSOCSigma(T imSOCSigma[]
, const T P1[], const T P2[]
, const T Ap_spline_a[], const T Ap_spline_b[], const T Ap_spline_c[], const T Ap_spline_d[]
, const T Am_spline_a[], const T Am_spline_b[], const T Am_spline_c[], const T Am_spline_d[]
, const T omega[], T U, int i0, int N_LOCAL, int N);


template<typename T> __device__ inline int d_interp_bsearch (const T x_array[], T x, int index_lo, int index_hi)
{
  int min = index_lo, max = index_hi;
  while (min + 1 < max)
  {
    int i = (min + max) >> 1;
    min = x > x_array[i] ? i : min;
    max = x > x_array[i] ? max : i;
  }
  return min;
}

template<typename T> __device__ T d_interpl(T x, const T y_array[],  const T x_array[], int size)
{
  T x_lo, x_hi;
  T y_lo, y_hi;
  T y;
  int index;
  const T xmin = x_array[0];
  const T xmax = x_array[size-1];

  if (x > xmax or x < xmin) {
    y = 0.0;
  }
  else {
    index = d_interp_bsearch<T>(x_array, x, 0, size - 1);

    /* evaluate */
    x_lo = x_array[index];
    x_hi = x_array[index + 1];
    y_lo = y_array[index];
    y_hi = y_array[index + 1];

    y = y_lo + (x - x_lo) / (x_hi - x_lo) * (y_hi - y_lo);
  }
  return y;
}

template<typename T> __device__ T d_cspline_eval(double x, const T cspline_a[], const T cspline_b[], const T cspline_c[], const T cspline_d[],  const T x_array[], int size){

  T x_lo, x_hi;
  //T y_lo, y_hi;
  T t;
  T y;
  size_t index;
  const T xmin = x_array[0];
  const T xmax = x_array[size-1];

  if (x > xmax or x < xmin) {
    y = 0.0;
  }
  else {
    index = d_interp_bsearch<T>(x_array, x, 0, size - 1);
    /* evaluate */
    x_lo = x_array[index];
    x_hi = x_array[index + 1];
    t = (x-x_lo)/(x_hi-x_lo);

    y = cspline_a[index]+cspline_b[index]*t+cspline_c[index]*t*t+cspline_d[index]*t*t*t;
  }
  
  return y;
}

/* CPU Function Headers */

//Interpolation
/*
size_t interp_bsearch (const double x_array[], double x, int index_lo, int index_hi);
double linear_interpl(double x, const double y_array[],  const double x_array[], int size);
*/

//Clipoff
bool h_ClipOff(double &X);
double imSOCSigmafc(double om, void *params);
double imSOCSigmafl(double om, void *params);


//Interpolation Function

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

bool h_ClipOff(double &X)
{
  if (X>0) 
  {
    X = -CLIPVAL;
    return true;
  }
  else
    return false;
}

/*=================================== DEVICE SIAM FUNCTIONS (Linear interpolation) ===================================*/

template<typename T> __global__ void d_get_Ps(T P1[], T P2[], const T Ap[], const T Am[], const T omega[], int i0, int N_LOCAL, int N)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int i = i0+tid;

  if (tid < N_LOCAL){

		//Trapezoidal Integral (range from 0 to N-1)
		
		T p1sum = 0.0;
		T p2sum = 0.0;
		
		const T p1_0 = Am[0]* d_interpl<T>(omega[0] - omega[i], Ap, omega,N);
		const T p2_0 = Ap[0]* d_interpl<T>(omega[0] - omega[i], Am, omega,N);
		const T p1_N = Am[N]* d_interpl<T>(omega[N-1] - omega[i], Ap, omega,N);
		const T p2_N = Ap[N]* d_interpl<T>(omega[N-1] - omega[i], Am, omega,N);
		
		p1sum += p1_0*(omega[1]-omega[0])+p1_N*(omega[N-1]-omega[N-2]);
		p1sum += p2_0*(omega[1]-omega[0])+p2_N*(omega[N-1]-omega[N-2]);

		for (int j=1;j<N-1;j++){
			const T p1 = Am[j]* d_interpl<T>(omega[j] - omega[i], Ap, omega,N);
			const T p2 = Ap[j]* d_interpl<T>(omega[j] - omega[i], Am, omega,N);
			const T domega = (omega[j+1]-omega[j-1]);
			p1sum += domega*p1;
			p2sum += domega*p2;
		}

		//Store result (to be pieced together if multi-GPU)
		P1[tid] = M_PI * p1sum * 0.5;
		P2[tid] = M_PI * p2sum * 0.5;
		
  }
}

template<typename T> __global__ void d_get_imSOCSigma(T imSOCSigma[],const T P1[], const T P2[], const T Ap[] , const T Am[] , const T omega[], T U, int i0, int N_LOCAL, int N)
{

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int i = i0+tid;

  if (tid < N_LOCAL)
  {
		//Trapezoidal Integral (range from 0 to N-1)
  	T ssum = 0.0;
  	
    const T s_0 = d_interpl<T>(omega[i] - omega[0], Ap, omega,N) * P2[0] + d_interpl<T>(omega[i] - omega[0], Am, omega,N) * P1[0];
    const T s_N = d_interpl<T>(omega[i] - omega[N-1], Ap, omega,N) * P2[N-1] + d_interpl<T>(omega[i] - omega[N-1], Am, omega,N) * P1[N-1];
  	
  	ssum += s_0*(omega[1]-omega[0])+s_N*(omega[N-1]-omega[N-2]);
  	
  	
    for (int j=1; j<N-1; j++) 
    { 
				const T s = d_interpl<T>(omega[i] - omega[j], Ap, omega,N) * P2[j] + d_interpl<T>(omega[i] - omega[j], Am, omega,N) * P1[j];
				ssum += (omega[j+1]-omega[j-1])*s;
    }            
      
    imSOCSigma[tid] = - U*U * ssum * 0.5;
  }
}

/*=================================== DEVICE SIAM FUNCTIONS (Cubic Spline interpolation) ===================================*/

template<typename T> __global__ void d_get_Psc(T P1[], T P2[]
, const T Ap[], const T Am[]
, const T Ap_spline_a[], const T Ap_spline_b[], const T Ap_spline_c[], const T Ap_spline_d[]
, const T Am_spline_a[], const T Am_spline_b[], const T Am_spline_c[], const T Am_spline_d[]
, const T omega[], int i0, int N_LOCAL, int N)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int i = i0+tid;

  if (tid < N_LOCAL){

		//Trapezoidal Integral (range from 0 to N-1)
		
		T p1sum = 0.0;
		T p2sum = 0.0;
		const T p1_0 = Am[0]* d_cspline_eval<T>(omega[0] - omega[i], Ap_spline_a , Ap_spline_b ,Ap_spline_c , Ap_spline_d, omega,N);
		const T p2_0 = Ap[0]* d_cspline_eval<T>(omega[0] - omega[i], Am_spline_a , Am_spline_b ,Am_spline_c , Am_spline_d, omega,N);
		const T p1_N = Am[N]* d_cspline_eval<T>(omega[N-1] - omega[i], Ap_spline_a , Ap_spline_b ,Ap_spline_c , Ap_spline_d, omega,N);
		const T p2_N = Ap[N]* d_cspline_eval<T>(omega[N-1] - omega[i], Am_spline_a , Am_spline_b ,Am_spline_c , Am_spline_d, omega,N);
		
		p1sum += p1_0*(omega[1]-omega[0])+p1_N*(omega[N-1]-omega[N-2]);
		p1sum += p2_0*(omega[1]-omega[0])+p2_N*(omega[N-1]-omega[N-2]);

		for (int j=1;j<N-1;j++){
			const T p1 = Am[j]* d_cspline_eval<T>(omega[j] - omega[i], Ap_spline_a , Ap_spline_b ,Ap_spline_c , Ap_spline_d, omega,N);
			const T p2 = Ap[j]* d_cspline_eval<T>(omega[j] - omega[i], Am_spline_a , Am_spline_b ,Am_spline_c , Am_spline_d, omega,N);
			const T domega = (omega[j+1]-omega[j-1]);
			p1sum += domega*p1;
			p2sum += domega*p2;
		}

		//Store result (to be pieced together if multi-GPU)
		P1[tid] = M_PI * p1sum * 0.5;
		P2[tid] = M_PI * p2sum * 0.5;
		
  }
}

template<typename T> __global__ void d_get_imSOCSigmac(T imSOCSigma[]
, const T P1[], const T P2[]
, const T Ap_spline_a[], const T Ap_spline_b[], const T Ap_spline_c[], const T Ap_spline_d[]
, const T Am_spline_a[], const T Am_spline_b[], const T Am_spline_c[], const T Am_spline_d[]
, const T omega[], T U, int i0, int N_LOCAL, int N)
{

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int i = i0+tid;

  if (tid < N_LOCAL)
  {
		//Trapezoidal Integral (range from 0 to N-1)
  	T ssum = 0.0;
  	
    const T s_0 = d_cspline_eval<T>(omega[i] - omega[0], Ap_spline_a , Ap_spline_b ,Ap_spline_c , Ap_spline_d, omega,N) * P2[0] 
    + d_cspline_eval<T>(omega[i] - omega[0], Am_spline_a , Am_spline_b ,Am_spline_c , Am_spline_d, omega,N) * P1[0];
    const T s_N = d_cspline_eval<T>(omega[i] - omega[N-1], Ap_spline_a , Ap_spline_b ,Ap_spline_c , Ap_spline_d, omega,N) * P2[N-1]
     + d_cspline_eval<T>(omega[i] - omega[N-1], Am_spline_a , Am_spline_b ,Am_spline_c , Am_spline_d, omega,N) * P1[N-1];
  	
  	ssum += s_0*(omega[1]-omega[0])+s_N*(omega[N-1]-omega[N-2]);
  	
  	
    for (int j=1; j<N-1; j++) 
    { 
				const T s = d_cspline_eval<T>(omega[i] - omega[j], Ap_spline_a , Ap_spline_b ,Ap_spline_c , Ap_spline_d, omega,N) * P2[j] 
				+ d_cspline_eval<T>(omega[i] - omega[j], Am_spline_a , Am_spline_b ,Am_spline_c , Am_spline_d, omega,N) * P1[j];
				ssum += (omega[j+1]-omega[j-1])*s;
    }            
      
    imSOCSigma[tid] = - U*U * ssum * 0.5;
  }
}

//=================================== HOST SIAM FUNCTIONS (FP64) ===================================//

void SIAM::get_Ps()
{

  TGPUplan<double> plan[MAX_GPU_COUNT];
    
  //Initialize multi-GPUs
  int GPU_N;
  cudaGetDeviceCount(&GPU_N); //Get amount of GPUs, assuming to be > 0;

  if (GPU_N > MAX_GPU_COUNT)
  {
    	GPU_N = MAX_GPU_COUNT;
  }
  
	//Subdividing input data across GPUs
	//Get data sizes for each GPU
	for (int i = 0; i < GPU_N; i++)
	{
			plan[i].N_LOCAL = N / GPU_N;
	}
	
	//Take into account "odd" data sizes
	for (int i = 0; i < N % GPU_N; i++)
	{
			plan[i].N_LOCAL++;
	}
	
	//Begining omega-point for each GPU
	int cumN=0;
	for (int i = 0; i<GPU_N ;i++)
	{
			plan[i].i0 = cumN;
			cumN += plan[i].N_LOCAL;
	}
	
	for (int i = 0; i<GPU_N ;i++)
	{
	    cudaSetDevice(i);
	    cudaStreamCreate(&plan[i].stream);
	    
    	//Allocate memory
	    cudaMalloc((void **)&plan[i].d_omega, N * sizeof(double));
	    cudaMalloc((void **)&plan[i].d_Ap, N * sizeof(double));
	    cudaMalloc((void **)&plan[i].d_Am, N * sizeof(double));
	    if (usecubicspline){
	    cudaMalloc((void **)&plan[i].d_Ap_spline_a, (N-1) * sizeof(double));
	    cudaMalloc((void **)&plan[i].d_Ap_spline_b, (N-1) * sizeof(double));
	    cudaMalloc((void **)&plan[i].d_Ap_spline_c, (N-1) * sizeof(double));
	    cudaMalloc((void **)&plan[i].d_Ap_spline_d, (N-1) * sizeof(double));
	    cudaMalloc((void **)&plan[i].d_Am_spline_a, (N-1) * sizeof(double));
	    cudaMalloc((void **)&plan[i].d_Am_spline_b, (N-1) * sizeof(double));
	    cudaMalloc((void **)&plan[i].d_Am_spline_c, (N-1) * sizeof(double));
	    cudaMalloc((void **)&plan[i].d_Am_spline_d, (N-1) * sizeof(double));
	    }
	    cudaMalloc((void **)&plan[i].d_P1, plan[i].N_LOCAL * sizeof(double));
	    cudaMalloc((void **)&plan[i].d_P2, plan[i].N_LOCAL * sizeof(double));
	    cudaMallocHost((void **)&plan[i].h_P1, plan[i].N_LOCAL * sizeof(double));
	    cudaMallocHost((void **)&plan[i].h_P2, plan[i].N_LOCAL * sizeof(double));
	    
			//initialize block count
			plan[i].block_count = (plan[i].N_LOCAL + CUDA_threadcount) / CUDA_threadcount;
	}
	
  for (int i = 0; i < GPU_N; i++)
  {
  		//Create streams for issuing GPU command asynchronously
	    cudaSetDevice(i);
	   
			
			//Copy omega-grid from CPU
			cudaMemcpyAsync(plan[i].d_omega, g->omega, N * sizeof(double), cudaMemcpyHostToDevice, plan[i].stream);
      
			//Copy Ap,Am from CPU
			cudaMemcpyAsync(plan[i].d_Ap, g->Ap, N * sizeof(double), cudaMemcpyHostToDevice, plan[i].stream);
			cudaMemcpyAsync(plan[i].d_Am, g->Am, N * sizeof(double), cudaMemcpyHostToDevice, plan[i].stream);
			
			if (!usecubicspline){
      	//Perform GPU computations
  			d_get_Ps<double><<<plan[i].block_count,CUDA_threadcount,0,plan[i].stream>>>(plan[i].d_P1, plan[i].d_P2
  																																								, plan[i].d_Ap, plan[i].d_Am
  																																								, plan[i].d_omega,plan[i].i0,plan[i].N_LOCAL, N);
			}
			else {
				cudaMemcpyAsync(plan[i].d_Ap_spline_a, Ap_spline->cspline_a, (N-1) * sizeof(double), cudaMemcpyHostToDevice, plan[i].stream);
				cudaMemcpyAsync(plan[i].d_Ap_spline_b, Ap_spline->cspline_b, (N-1) * sizeof(double), cudaMemcpyHostToDevice, plan[i].stream);
				cudaMemcpyAsync(plan[i].d_Ap_spline_c, Ap_spline->cspline_c, (N-1) * sizeof(double), cudaMemcpyHostToDevice, plan[i].stream);
				cudaMemcpyAsync(plan[i].d_Ap_spline_d, Ap_spline->cspline_d, (N-1) * sizeof(double), cudaMemcpyHostToDevice, plan[i].stream);
				cudaMemcpyAsync(plan[i].d_Am_spline_a, Am_spline->cspline_a, (N-1) * sizeof(double), cudaMemcpyHostToDevice, plan[i].stream);
				cudaMemcpyAsync(plan[i].d_Am_spline_b, Am_spline->cspline_b, (N-1) * sizeof(double), cudaMemcpyHostToDevice, plan[i].stream);
				cudaMemcpyAsync(plan[i].d_Am_spline_c, Am_spline->cspline_c, (N-1) * sizeof(double), cudaMemcpyHostToDevice, plan[i].stream);
				cudaMemcpyAsync(plan[i].d_Am_spline_d, Am_spline->cspline_d, (N-1) * sizeof(double), cudaMemcpyHostToDevice, plan[i].stream);
      	//Perform GPU computations
  			d_get_Psc<double><<<plan[i].block_count,CUDA_threadcount,0,plan[i].stream>>>(plan[i].d_P1, plan[i].d_P2
																																									, plan[i].d_Ap, plan[i].d_Am
  																																								, plan[i].d_Ap_spline_a, plan[i].d_Ap_spline_b, plan[i].d_Ap_spline_c, plan[i].d_Ap_spline_d
  																																								, plan[i].d_Am_spline_a, plan[i].d_Am_spline_b, plan[i].d_Am_spline_c, plan[i].d_Am_spline_d
  																																								, plan[i].d_omega
  																																								, plan[i].i0,plan[i].N_LOCAL, N);
			}
      //Read back GPU results
      cudaMemcpyAsync(plan[i].h_P1, plan[i].d_P1, plan[i].N_LOCAL *sizeof(double), cudaMemcpyDeviceToHost, plan[i].stream);
      cudaMemcpyAsync(plan[i].h_P2, plan[i].d_P2, plan[i].N_LOCAL *sizeof(double), cudaMemcpyDeviceToHost, plan[i].stream);
  }
  
  //Process GPU results
  for (int i = 0; i < GPU_N; i++)
  {
      //Set device
      cudaSetDevice(i);

      //Wait for all operations to finish
      cudaStreamSynchronize(plan[i].stream);

			//printf("Finished...\n");
      //Piece together results from GPU

      for (int j = 0; j < plan[i].N_LOCAL; j++)
      {
          g->P1[plan[i].i0+j] = plan[i].h_P1[j];
          g->P2[plan[i].i0+j] = plan[i].h_P2[j];
      }
      
			//printf("Freeing memory...\n");
			
      cudaFreeHost(plan[i].h_P1);
      cudaFreeHost(plan[i].h_P2);
      cudaFree(plan[i].d_omega);
      cudaFree(plan[i].d_Ap);
      cudaFree(plan[i].d_Am);
      
	    if (usecubicspline){
      cudaFree(plan[i].d_Ap_spline_a);
      cudaFree(plan[i].d_Ap_spline_b);
      cudaFree(plan[i].d_Ap_spline_c);
      cudaFree(plan[i].d_Ap_spline_d);
      cudaFree(plan[i].d_Am_spline_a);
      cudaFree(plan[i].d_Am_spline_b);
      cudaFree(plan[i].d_Am_spline_c);
      cudaFree(plan[i].d_Am_spline_d);
      }
      cudaFree(plan[i].d_P1);
      cudaFree(plan[i].d_P2);
      cudaStreamDestroy(plan[i].stream);
  }
  
  cudaDeviceSynchronize();
  
}


void SIAM::get_SOCSigma(){

  double *imSOCSigma = new double[N];
  
  TGPUplan<double> plan[MAX_GPU_COUNT];
    
  //Initialize multi-GPUs
  int GPU_N;
  cudaGetDeviceCount(&GPU_N); //Get amount of GPUs, assuming to be > 0;

  if (GPU_N > MAX_GPU_COUNT)
  {
    	GPU_N = MAX_GPU_COUNT;
  }
  
	//Subdividing input data across GPUs
	//Get data sizes for each GPU
	for (int i = 0; i < GPU_N; i++)
	{
			plan[i].N_LOCAL = N / GPU_N;
	}
	
	//Take into account "odd" data sizes
	for (int i = 0; i < N % GPU_N; i++)
	{
			plan[i].N_LOCAL++;
	}
	
	//Begining omega-point for each GPU
	int cumN=0;
	for (int i = 0; i<GPU_N ;i++)
	{
			plan[i].i0 = cumN;
			cumN += plan[i].N_LOCAL;
			//printf("GPU %d will work on %d points from %d\n",i,plan[i].N_LOCAL,plan[i].i0);
	}

  //allocate memory (GPU and System page-locked)
  for (int i = 0; i < GPU_N; i++)
  {
      cudaSetDevice(i);
      
	    cudaStreamCreate(&plan[i].stream);
      
      //Allocate memory
	    cudaMalloc((void **)&plan[i].d_omega, N * sizeof(double));
	    cudaMalloc((void **)&plan[i].d_Ap, N * sizeof(double));
	    cudaMalloc((void **)&plan[i].d_Am, N * sizeof(double));
	    
	    if (usecubicspline){
	    cudaMalloc((void **)&plan[i].d_Ap_spline_a, (N-1) * sizeof(double));
	    cudaMalloc((void **)&plan[i].d_Ap_spline_b, (N-1) * sizeof(double));
	    cudaMalloc((void **)&plan[i].d_Ap_spline_c, (N-1) * sizeof(double));
	    cudaMalloc((void **)&plan[i].d_Ap_spline_d, (N-1) * sizeof(double));
	    cudaMalloc((void **)&plan[i].d_Am_spline_a, (N-1) * sizeof(double));
	    cudaMalloc((void **)&plan[i].d_Am_spline_b, (N-1) * sizeof(double));
	    cudaMalloc((void **)&plan[i].d_Am_spline_c, (N-1) * sizeof(double));
	    cudaMalloc((void **)&plan[i].d_Am_spline_d, (N-1) * sizeof(double));
	    }
	    
      cudaMalloc((void **)&plan[i].d_P1, N * sizeof(double));
      cudaMalloc((void **)&plan[i].d_P2, N * sizeof(double));
      cudaMalloc((void **)&plan[i].d_imSOCSigma, plan[i].N_LOCAL * sizeof(double));
      cudaMallocHost((void **)&plan[i].h_imSOCSigma, plan[i].N_LOCAL * sizeof(double));
      
			//initialize block count
			plan[i].block_count = (plan[i].N_LOCAL + CUDA_threadcount) / CUDA_threadcount;
  }
  
  //Copy data to GPU, launch the kernel and copy data back. All asynchronously
  for (int i = 0; i < GPU_N; i++)
  {
      //Set device
      cudaSetDevice(i);
      
      
      //Copy pieced-together P1,P2 from CPU
      cudaMemcpyAsync(plan[i].d_omega, g->omega, N * sizeof(double), cudaMemcpyHostToDevice, plan[i].stream);
      cudaMemcpyAsync(plan[i].d_Ap, g->Ap, N * sizeof(double), cudaMemcpyHostToDevice, plan[i].stream);
      cudaMemcpyAsync(plan[i].d_Am, g->Am, N * sizeof(double), cudaMemcpyHostToDevice, plan[i].stream);
      cudaMemcpyAsync(plan[i].d_P1, g->P1, N * sizeof(double), cudaMemcpyHostToDevice, plan[i].stream);
      cudaMemcpyAsync(plan[i].d_P2, g->P2, N * sizeof(double), cudaMemcpyHostToDevice, plan[i].stream);

			if (!usecubicspline){
		    //Perform GPU computations
				d_get_imSOCSigma<double><<<plan[i].block_count,CUDA_threadcount,0,plan[i].stream>>>(plan[i].d_imSOCSigma
																																													, plan[i].d_P1, plan[i].d_P2
																																													, plan[i].d_Ap , plan[i].d_Am 
																																													, plan[i].d_omega
																																													, U ,plan[i].i0, plan[i].N_LOCAL, N);
			}
			else{
				cudaMemcpyAsync(plan[i].d_Ap_spline_a, Ap_spline->cspline_a, (N-1) * sizeof(double), cudaMemcpyHostToDevice, plan[i].stream);
				cudaMemcpyAsync(plan[i].d_Ap_spline_b, Ap_spline->cspline_b, (N-1) * sizeof(double), cudaMemcpyHostToDevice, plan[i].stream);
				cudaMemcpyAsync(plan[i].d_Ap_spline_c, Ap_spline->cspline_c, (N-1) * sizeof(double), cudaMemcpyHostToDevice, plan[i].stream);
				cudaMemcpyAsync(plan[i].d_Ap_spline_d, Ap_spline->cspline_d, (N-1) * sizeof(double), cudaMemcpyHostToDevice, plan[i].stream);
				cudaMemcpyAsync(plan[i].d_Am_spline_a, Am_spline->cspline_a, (N-1) * sizeof(double), cudaMemcpyHostToDevice, plan[i].stream);
				cudaMemcpyAsync(plan[i].d_Am_spline_b, Am_spline->cspline_b, (N-1) * sizeof(double), cudaMemcpyHostToDevice, plan[i].stream);
				cudaMemcpyAsync(plan[i].d_Am_spline_c, Am_spline->cspline_c, (N-1) * sizeof(double), cudaMemcpyHostToDevice, plan[i].stream);
				cudaMemcpyAsync(plan[i].d_Am_spline_d, Am_spline->cspline_d, (N-1) * sizeof(double), cudaMemcpyHostToDevice, plan[i].stream);
		    //Perform GPU computations
				d_get_imSOCSigmac<double><<<plan[i].block_count,CUDA_threadcount,0,plan[i].stream>>>(plan[i].d_imSOCSigma
																																													, plan[i].d_P1, plan[i].d_P2
  																																												, plan[i].d_Ap_spline_a, plan[i].d_Ap_spline_b, plan[i].d_Ap_spline_c, plan[i].d_Ap_spline_d
  																																												, plan[i].d_Am_spline_a, plan[i].d_Am_spline_b, plan[i].d_Am_spline_c, plan[i].d_Am_spline_d
																																													, plan[i].d_omega
																																													, U ,plan[i].i0, plan[i].N_LOCAL, N);
			}
      //Read back GPU results
      cudaMemcpyAsync(plan[i].h_imSOCSigma, plan[i].d_imSOCSigma, plan[i].N_LOCAL * sizeof(double), cudaMemcpyDeviceToHost, plan[i].stream);

  }
  
  //Process GPU results
  for (int i = 0; i < GPU_N; i++)
  {
      //Set device
      cudaSetDevice(i);

      //Wait for all operations to finish
      cudaStreamSynchronize(plan[i].stream);

      //Piece together results from GPU

      for (int j = 0; j < plan[i].N_LOCAL; j++)
      {
          imSOCSigma[plan[i].i0+j] = plan[i].h_imSOCSigma[j];
          if (h_ClipOff( imSOCSigma[plan[i].i0+j] )) Clipped = true ;
      }
      
      cudaFreeHost(plan[i].h_imSOCSigma);
      cudaFree(plan[i].d_omega);
      cudaFree(plan[i].d_Ap);
      cudaFree(plan[i].d_Am);
      
	    if (usecubicspline){
      cudaFree(plan[i].d_Ap_spline_a);
      cudaFree(plan[i].d_Ap_spline_b);
      cudaFree(plan[i].d_Ap_spline_c);
      cudaFree(plan[i].d_Ap_spline_d);
      cudaFree(plan[i].d_Am_spline_a);
      cudaFree(plan[i].d_Am_spline_b);
      cudaFree(plan[i].d_Am_spline_c);
      cudaFree(plan[i].d_Am_spline_d);
      }
      
      cudaFree(plan[i].d_P1);
      cudaFree(plan[i].d_P2);
      cudaFree(plan[i].d_imSOCSigma);
      cudaStreamDestroy(plan[i].stream);
  }
  if (Clipped) sprintf(ibuffer + strlen(ibuffer),"SIAM::run::(Warning) !!!Clipping SOCSigma!!!!\n");
  
  cudaDeviceSynchronize();  
  
  // The KramersKonig function is not used to compute the Cauchy Integral, this allows arbitrary grid to be used.
  
  gsl_set_error_handler_off();
  
  if (usecubicspline) imSOCSigmaspline = new dinterpl(g->omega, imSOCSigma , N);
  
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


//=================================== HOST SIAM FUNCTIONS (FP32, use if FP64 is too slow on consumer grade GPUs) ===================================//
/*
void SIAM::get_Ps()
{

  TGPUplan<float> plan[MAX_GPU_COUNT];
  
  //Type cast double to float
  float *omega32 = new float[N];
  float *Ap32 = new float[N];
  float *Am32 = new float[N];
  std::copy(g->omega,(g->omega)+N,omega32);
  std::copy(g->Ap,(g->Ap)+N,Ap32);
  std::copy(g->Am,(g->Am)+N,Am32);
    
  //Initialize multi-GPUs
  int GPU_N;
  cudaGetDeviceCount(&GPU_N); //Get amount of GPUs, assuming to be > 0;

  if (GPU_N > MAX_GPU_COUNT)
  {
    	GPU_N = MAX_GPU_COUNT;
  }
  
	//Subdividing input data across GPUs
	//Get data sizes for each GPU
	for (int i = 0; i < GPU_N; i++)
	{
			plan[i].N_LOCAL = N / GPU_N;
	}
	
	//Take into account "odd" data sizes
	for (int i = 0; i < N % GPU_N; i++)
	{
			plan[i].N_LOCAL++;
	}
	
	//Begining omega-point for each GPU
	int cumN=0;
	for (int i = 0; i<GPU_N ;i++)
	{
			plan[i].i0 = cumN;
			cumN += plan[i].N_LOCAL;
			//printf("GPU %d will work on %d points from %d\n",i,plan[i].N_LOCAL,plan[i].i0);
	}
	
	for (int i = 0; i<GPU_N ;i++)
	{
	    cudaSetDevice(i);
	    cudaStreamCreate(&plan[i].stream);
	    
    	//Allocate memory
	    cudaMalloc((void **)&plan[i].d_omega, N * sizeof(float));
	    cudaMalloc((void **)&plan[i].d_Ap, N * sizeof(float));
	    cudaMalloc((void **)&plan[i].d_Am, N * sizeof(float));
	    cudaMalloc((void **)&plan[i].d_P1, plan[i].N_LOCAL * sizeof(float));
	    cudaMalloc((void **)&plan[i].d_P2, plan[i].N_LOCAL * sizeof(float));
	    cudaMallocHost((void **)&plan[i].h_P1, plan[i].N_LOCAL * sizeof(float));
	    cudaMallocHost((void **)&plan[i].h_P2, plan[i].N_LOCAL * sizeof(float));
	    
			//initialize block count
			plan[i].block_count = (plan[i].N_LOCAL + CUDA_threadcount) / CUDA_threadcount;
	}
	
  for (int i = 0; i < GPU_N; i++)
  {
  		//Create streams for issuing GPU command asynchronously
	    cudaSetDevice(i);
	   
			
			//Copy omega-grid from CPU
			cudaMemcpyAsync(plan[i].d_omega, omega32, N * sizeof(float), cudaMemcpyHostToDevice, plan[i].stream);
      
			//Copy Ap,Am from CPU
			cudaMemcpyAsync(plan[i].d_Ap, Ap32, N * sizeof(float), cudaMemcpyHostToDevice, plan[i].stream);
			cudaMemcpyAsync(plan[i].d_Am, Am32, N * sizeof(float), cudaMemcpyHostToDevice, plan[i].stream);
			
      //Perform GPU computations
  		d_get_Ps<float><<<plan[i].block_count,CUDA_threadcount,0,plan[i].stream>>>(plan[i].d_P1, plan[i].d_P2, plan[i].d_Ap, plan[i].d_Am, plan[i].d_omega,plan[i].i0,plan[i].N_LOCAL, N);

      //Read back GPU results
      cudaMemcpyAsync(plan[i].h_P1, plan[i].d_P1, plan[i].N_LOCAL *sizeof(float), cudaMemcpyDeviceToHost, plan[i].stream);
      cudaMemcpyAsync(plan[i].h_P2, plan[i].d_P2, plan[i].N_LOCAL *sizeof(float), cudaMemcpyDeviceToHost, plan[i].stream);
  }
  
  //Process GPU results
  for (int i = 0; i < GPU_N; i++)
  {
      //Set device
      cudaSetDevice(i);

      //Wait for all operations to finish
      cudaStreamSynchronize(plan[i].stream);

      //Piece together results from GPU (typecast to double)
      for (int j = 0; j < plan[i].N_LOCAL; j++)
      {
          g->P1[plan[i].i0+j] = (double) plan[i].h_P1[j];
          g->P2[plan[i].i0+j] = (double) plan[i].h_P2[j];
      }
      
      cudaFreeHost(plan[i].h_P1);
      cudaFreeHost(plan[i].h_P2);
      cudaFree(plan[i].d_omega);
      cudaFree(plan[i].d_Ap);
      cudaFree(plan[i].d_Am);
      cudaFree(plan[i].d_P1);
      cudaFree(plan[i].d_P2);
      cudaStreamDestroy(plan[i].stream);
  }
  cudaDeviceSynchronize();
  delete [] omega32;
  delete [] Ap32;
  delete [] Am32;
}


void SIAM::get_SOCSigma(){

  double *imSOCSigma = new double[N];
  
  TGPUplan<float> plan[MAX_GPU_COUNT];
  
  //Type cast double to float
  float *omega32 = new float[N];
  float *Ap32 = new float[N];
  float *Am32 = new float[N];
  float *P132 = new float[N];
  float *P232 = new float[N];
  std::copy(g->omega,(g->omega)+N,omega32);
  std::copy(g->Ap,(g->Ap)+N,Ap32);
  std::copy(g->Am,(g->Am)+N,Am32);
  std::copy(g->P1,(g->P1)+N,P132);
  std::copy(g->P2,(g->P2)+N,P232);
    
  //Initialize multi-GPUs
  int GPU_N;
  cudaGetDeviceCount(&GPU_N); //Get amount of GPUs, assuming to be > 0;

  if (GPU_N > MAX_GPU_COUNT)
  {
    	GPU_N = MAX_GPU_COUNT;
  }
  
	//Subdividing input data across GPUs
	//Get data sizes for each GPU
	for (int i = 0; i < GPU_N; i++)
	{
			plan[i].N_LOCAL = N / GPU_N;
	}
	
	//Take into account "odd" data sizes
	for (int i = 0; i < N % GPU_N; i++)
	{
			plan[i].N_LOCAL++;
	}
	
	//Begining omega-point for each GPU
	int cumN=0;
	for (int i = 0; i<GPU_N ;i++)
	{
			plan[i].i0 = cumN;
			cumN += plan[i].N_LOCAL;
			//printf("GPU %d will work on %d points from %d\n",i,plan[i].N_LOCAL,plan[i].i0);
	}

  //allocate memory (GPU and System page-locked)
  for (int i = 0; i < GPU_N; i++)
  {
      cudaSetDevice(i);
      
	    cudaStreamCreate(&plan[i].stream);
      
      //Allocate memory
	    cudaMalloc((void **)&plan[i].d_omega, N * sizeof(float));
	    cudaMalloc((void **)&plan[i].d_Ap, N * sizeof(float));
	    cudaMalloc((void **)&plan[i].d_Am, N * sizeof(float));
      cudaMalloc((void **)&plan[i].d_P1, N * sizeof(float));
      cudaMalloc((void **)&plan[i].d_P2, N * sizeof(float));
      cudaMalloc((void **)&plan[i].d_imSOCSigma, plan[i].N_LOCAL * sizeof(float));
      cudaMallocHost((void **)&plan[i].h_imSOCSigma, plan[i].N_LOCAL * sizeof(float));
      
			//initialize block count
			plan[i].block_count = (plan[i].N_LOCAL + CUDA_threadcount) / CUDA_threadcount;
  }
  
  //Copy data to GPU, launch the kernel and copy data back. All asynchronously
  for (int i = 0; i < GPU_N; i++)
  {
      //Set device
      cudaSetDevice(i);
      
      
      //Copy pieced-together P1,P2 from CPU
      cudaMemcpyAsync(plan[i].d_omega, omega32, N * sizeof(float), cudaMemcpyHostToDevice, plan[i].stream);
      cudaMemcpyAsync(plan[i].d_Ap, Ap32, N * sizeof(float), cudaMemcpyHostToDevice, plan[i].stream);
      cudaMemcpyAsync(plan[i].d_Am, Am32, N * sizeof(float), cudaMemcpyHostToDevice, plan[i].stream);
      cudaMemcpyAsync(plan[i].d_P1, P132, N * sizeof(float), cudaMemcpyHostToDevice, plan[i].stream);
      cudaMemcpyAsync(plan[i].d_P2, P232, N * sizeof(float), cudaMemcpyHostToDevice, plan[i].stream);

      //Perform GPU computations
  		d_get_imSOCSigma<float><<<plan[i].block_count,CUDA_threadcount,0,plan[i].stream>>>(plan[i].d_imSOCSigma, plan[i].d_P1, plan[i].d_P2, plan[i].d_Ap , plan[i].d_Am , plan[i].d_omega, U,
  																																								plan[i].i0, plan[i].N_LOCAL, N);

      //Read back GPU results
      cudaMemcpyAsync(plan[i].h_imSOCSigma, plan[i].d_imSOCSigma, plan[i].N_LOCAL * sizeof(float), cudaMemcpyDeviceToHost, plan[i].stream);

  }
  
  //Process GPU results
  for (int i = 0; i < GPU_N; i++)
  {
      //Set device
      cudaSetDevice(i);

      //Wait for all operations to finish
      cudaStreamSynchronize(plan[i].stream);

      //Piece together results from GPU (typecast to double)
      for (int j = 0; j < plan[i].N_LOCAL; j++)
      {
          imSOCSigma[plan[i].i0+j] = (double) plan[i].h_imSOCSigma[j];
          if (h_ClipOff( imSOCSigma[plan[i].i0+j] )) Clipped = true ;
      }
      
      cudaFreeHost(plan[i].h_imSOCSigma);
      cudaFree(plan[i].d_omega);
      cudaFree(plan[i].d_Ap);
      cudaFree(plan[i].d_Am);
      cudaFree(plan[i].d_P1);
      cudaFree(plan[i].d_P2);
      cudaFree(plan[i].d_imSOCSigma);
      cudaStreamDestroy(plan[i].stream);
  }
  if (Clipped) printf("    !!!Clipping SOCSigma!!!!\n");
  
  cudaDeviceSynchronize();  
  
  // The KramersKonig function is not used to compute the Cauchy Integral, this allows arbitrary grid to be used.
  
  gsl_set_error_handler_off();
  #pragma omp parallel for schedule(dynamic)
  for (int i=1; i<N-1; i++)
  { 
    const double a = g->omega[0], b = g->omega[N-1]; // limits of integration
    const double epsabs = 0., epsrel = 1.e-12; // requested errors
    double result; // the integral value
    double error; // the error estimate

    double c = g->omega[i];

    struct KK_params params;
    params.omega = g->omega;
    params.Y = imSOCSigma;
    params.N = N;

    gsl_function F;
    F.function = &imSOCSigmafc;
    F.params = &params;

    
    size_t limit = 100;// work area size
    gsl_integration_workspace *ws = gsl_integration_workspace_alloc (limit);

    gsl_integration_qawc (&F, a, b , c , epsabs, epsrel, limit, ws, &result, &error);

    gsl_integration_workspace_free (ws);

    g->SOCSigma[i] = complex<double>(result/M_PI,imSOCSigma[i]);
  }
  
  g->SOCSigma[0] = g->SOCSigma[1];
  g->SOCSigma[N-1] = g->SOCSigma[N-2];

  delete [] imSOCSigma;
  
  delete [] omega32;
  delete [] Ap32;
  delete [] Am32;
  delete [] P132;
  delete [] P232;
}

*/
