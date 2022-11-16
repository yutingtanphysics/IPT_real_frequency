/*
	The class grid stores the grid that IPT read and writes, called by SIAM class.
*/

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "Grid.h"

// Constructor and Destructors

Grid::Grid(const double omega_in[],size_t N)
{
	//This constructor assign all memory needed in this class
  
  //Grid
  this->N = N;
  
  omega = new double[N];
  
  //Input
  Delta = new complex<double>[N];
  //Output
  Sigma = new complex<double>[N];
  G = new complex<double>[N];
  
  //copy omega-grid
	for (int i=0;i<N;i++){
		omega[i] = omega_in[i]; 
	}

	//Intermediate steps
  fermi = new double[N];
  G0 = new complex<double>[N];
  Ap = new double[N];
  Am = new double[N];
  P1 = new double[N];
  P2 = new double[N];
  SOCSigma = new complex<double>[N];
}

Grid::~Grid()
{
  ReleaseMemory();
}

void Grid::ReleaseMemory()
{
  delete [] Sigma;
  delete [] G;
  
  delete [] fermi;          
  delete [] G0;  
  delete [] Ap;          
  delete [] Am;
  delete [] P1;         
  delete [] P2;
  delete [] SOCSigma;
  
  
  delete [] omega;
  delete [] Delta;
}

void Grid::PrintFullResult(const char* ResultFN) //this prints everything, including intermediate step
{ 
  FILE *f;
  f = fopen(ResultFN, "w");

  for (int i=0; i<N; i++)
  { 
     // loop through and store the numbers into the file
    fprintf(f, "%.15le %.15le %.15le %.15le %.15le %.15le %.15le %.15le %.15le %.15le %.15le %.15le %.15le %.15le %.15le %.15le\n", 
                   omega[i], fermi[i],					//1 2
                   real(Delta[i]), imag(Delta[i]),			//3 4
                   real(G0[i]), imag(G0[i]), 				//5 6
                   Ap[i], Am[i], P1[i], P2[i],				//7 8 9 10 
                   real(SOCSigma[i]), imag(SOCSigma[i]), 		//11 12
                   real(Sigma[i]), imag(Sigma[i]),			//13 14
                   real(G[i]), imag(G[i]));				//15 16
                   
                
  }
  fclose(f);
}

void Grid::PrintResult(const char* Gffile,const char* Sigfile) //Non-debug version to save disk space
{ 
  FILE *fGf;
  fGf = fopen(Gffile, "w");
  for (int i=0; i<N; i++) fprintf(fGf, "%.15le %.15le %.15le\n", omega[i], real(G[i]), imag(G[i]));  
  fclose(fGf);
  
  FILE *fSigma;
  fSigma = fopen(Sigfile, "w");
  for (int i=0; i<N; i++) fprintf(fSigma, "%.15le %.15le %.15le\n", omega[i], real(Sigma[i]), imag(Sigma[i]));  
  fclose(fSigma);
}

