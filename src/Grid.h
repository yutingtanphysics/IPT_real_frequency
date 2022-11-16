/*
	The class grid stores the grid that IPT read and writes, called by SIAM class.
	Adapted from code by Jaksha Vuchichevicc https://github.com/JaksaVucicevic/DMFT
	in the original code this class is named "Result"
*/


#include <complex>
using namespace std;

class Grid
{
  public:
  
  	//Constructor-destructor
    Grid(const double omega_in[],size_t N);
    ~Grid();
    
    //Grid
    int N; //length of grid
    double* omega;		//omega grid
    
    //Input
    complex<double>* Delta;	//Bath
    
    //Output
    complex<double>* Sigma;	//Sigma interpolating between exact limiting cases
    complex<double>* G;		//Greens function on real axis
    
    //Occupation
    double n,n0;
    
    void PrintFullResult(const char* ResultFN); //Debug version - basically prints everything
    void PrintResult(const char* Gffile,const char* Sigfile); //Non-debug version to save disk space
    

  	//Intermediate steps (except the fermi function none of these things can be "recycled")
    double* fermi;		//fermi function
    double* Ap;			//spectral functions
    double* Am;
    double* P1;			//polarizations
    double* P2;
    complex<double>* SOCSigma;	//Second order contribution in sigma
    complex<double>* G0;	//auxillary Green's function
    
  private:
    void ReleaseMemory();
};
