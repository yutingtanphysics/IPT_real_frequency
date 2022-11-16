#include <iostream>
#include <complex>


/*

*/

//Buffer size for printouts  (It should be sufficient unless AmoebaMaxIts is increased to absurdly large values)
#define BUFFERSIZE 65536

//Maximum subdivision for adaptive quadrature
#define QUADLIMIT 200

//epsrel for adaptive quadrature
#define QUADACCR 1e-9

//
#define CLIPVAL 1e-5


class Grid;
class dinterpl;

using namespace std;

//Siam params

struct siamparams{
	double mu,mu0,U,T,epsilon,eta;
	double KKAccr,Accr;
	double * L;
	double * R;
	int fitorder;
	bool SymmetricCase,Fixmu0,usecubicspline,tailcorrection;
	
	bool broadenSigma;
	
	bool CheckSpectralWeight;
	bool verbose;
	int HybridBisectMaxIts;
	double HybridBisectStart,HybridBisectEnd;
};

//======================= SIAM Class ==========================================//

class SIAM
{
  public:
  
    //--Constructors/destructors--//
    SIAM(const double omega[],size_t N, void * params);
    ~SIAM();
    
    //--------RUN SIAM--------//
    int Run(const complex<double> Delta[]); 
    
    //--Result IO--//
    void PrintResult(); //print to "Gf.out" and "Sig.out"
    void PrintResult(const char* Gffile,const char* Sigfile);
    void PrintFullResult(const char* ResultFN);
    
    //--Buffer IO--//
    const char* buffer();
    void PrintBuffer(const char* FN);
    void PrintBuffer(const char* FN,bool quiet);

    //No meaning so far
    int status;
  private:
  	//For MPI use only (not active in openmp/gpu mode)
  	int rank,size;
  
  	//verbose output (currently not useful as we print all output)
  	bool verbose;
  	
  	//print buffer
  	char * ibuffer;
  	
  	//parameters
  	struct siamparams *p;
  	
    //--Grid--//
    Grid* g;
    int N;
    
    //--impurity parameters--//
    double U;			//on-site repulsion
    double T;			//temperature
    double epsilon;		//impurity energy level

    //--bath parameters--// 
    double mu;			//global chemical potential
    double mu0;			//fictious chemical potential
    
    //---BROADENING---//
    double eta;
    complex<double> ieta;
    complex<double> ietaS;
    

    //----tail correction----//
    bool tailcorrection;
    bool broadenSigma;
    int fitorder;
    double *L;
    double *R;
    
		double getn0corr();
		double getwG0corr();

    //----kramers kronig----//
    double KKAccr;
    bool usecubicspline;
  	dinterpl* imSOCSigmaspline;
  	
  	//----splines----//
  	dinterpl* Ap_spline;
  	dinterpl* Am_spline;
    
     //--get functions--//
    double get_fermi(int i);
    double get_n(complex<double> X[]);

    //--get procedures--//
    void get_fermi();
    void get_G0();
    void get_As();
    void get_Ps();  
    void get_SOCSigma();
    double get_b();
    void get_Sigma();
    void get_G();

		//--Clipping of ImSigma and ImG--//
    bool ClipOff(complex<double> &X);
    bool ClipOff(double &X);
    bool Clipped;
    double ClippingValue = CLIPVAL;

    //--- SIAM solver ---//
    void SolveSiam(double* V);
    void Hybrid_Bisection(double accr, double* V);
  
    //-- mu0 search --//
    bool SymmetricCase;
    bool HalfFilling;
    bool Fixmu0;
    
    int MAX_ITS; 
    
    double Accr;
    
    double HybridBisectStart;	//before amoeba starts, the equation is solved roughly (with accuracy AmobeScanStep) by scanning from AmoebaScanStart to AmoebaScanEnd.
    double HybridBisectEnd; 	//make sure AmoebaScanStart and AmoebaScanEnd are far enough apart (when U or W is large).
    int HybridBisectMaxIts;		//maximum number of Amoeba iterations
    
    
    
    //------ OPTIONS -------//
    bool CheckSpectralWeight;   //if true program prints out spectral weights of G and G0 after each iteration
    
    
   
};
