#include <stdlib.h>
#include <argp.h>
#include "routines.h"
#include "SIAM.h"
#include "Params.h"
#include "dinterpl.h"
#include "tail.h"
#include <cmath>
#include <complex>
#include <fstream>
#include <mpi.h>

using namespace std;


const char *argp_program_version =
  "IPT-cuda Aug-2020";
const char *argp_program_bug_address =
  "<henrytsang222@gmail.com>";
  
/* TODO Program documentation. */
static char doc[] =
  "Iterative Pertubation Theory (IPT) Solver for Anderson Impurity Model on the real axis";
  
/* INPUT DESCRIPTION */
static char args_doc[] = "[INPUT]";

/* Options */
static struct argp_option options[] = {
  {"params",  'p',  "FILE",      0,  "Read params from FILE instead of standard input" },
  {"log",  'l',  "FILE",      0,  "Output log to FILE instead of standard output" },
  {"grid",  'w', "FILE",      0,  "Read grid from FILE" },
  {"Gf",  'g', "FILE",      0,  "Output Gf to FILE" },
  {"debug",  'd', 0,      0,  "Output intermediate steps" },
  {"verbose",  'v', 0,      0,  "Produce verbose output" },
  {"quiet",    'q', 0,      0,  "Don't produce any output" },
  {"silent",   's', 0,      OPTION_ALIAS },
  {"output",   'o', "FILE", 0,  "Output to FILE instead of standard output" },
  { 0 }
};


/* Used by main to communicate with parse_opt. */
struct arguments
{
  char *args[1];                /* arg1 */
  int silent, verbose,debug,Gf;
  char *Sigma_file,*Gf_file,*grid_file,*log_file,*params_file;
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
    case 'd':
      arguments->debug = 1;
      break;
    case 'o':
      arguments->Sigma_file = arg;
      break;
    case 'g':
      arguments->Gf_file = arg;
      break;
    case 'w':
      arguments->grid_file = arg;
      break;

    case ARGP_KEY_ARG:
      if (state->arg_num >= 1)
        /* Too many arguments. */
        argp_usage (state);

      arguments->args[state->arg_num] = arg;

      break;

    case ARGP_KEY_END:
      if (state->arg_num < 1)
        /* Not enough arguments. */
        argp_usage (state);
      break;

    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

static struct argp argp = { options, parse_opt, args_doc, doc };


int main(int argc, char* argv[])
{

	//START MPI
	MPI_Init(&argc, &argv);
	
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);  //Rank
  MPI_Comm_size(MPI_COMM_WORLD, &size);  //Size
	
  struct arguments arguments;

  /* Default values. */
  arguments.debug = 0;
  arguments.silent = 0;
  arguments.verbose = 0;
  arguments.Sigma_file = "\0";
  arguments.Gf_file = "\0";
  arguments.grid_file = "\0";
  arguments.log_file = "IPT.log";
  arguments.params_file = "PARAMS";

  argp_parse (&argp, argc, argv, 0, 0, &arguments);
  
  //IPT parameters
  bool quiet = arguments.silent;
  bool verbose = arguments.verbose;
  bool defaultgrid = true;
  bool debug = arguments.debug;
  string logfile = arguments.log_file;
  string paramsfile = arguments.params_file;
  char* deltafile = arguments.args[0];
  string gridfile;
  string Gffile;
  string Sigfile;
  bool tailcorrection = true;
  
  int N;
  double * omega;
  complex<double> * Delta;
  
  //IPT
	double mu,mu0,U,T,epsilon,eta;
	double KKAccr,Accr;
	
	//Tail
	double * L;
	double * R;
	int fitorder;
	int ntail;
	bool SymmetricCase,Fixmu0,usecubicspline;
	
	bool CheckSpectralWeight;
	int HybridBisectMaxIts;
	double HybridBisectStart, HybridBisectEnd;
  

	//Begin Root IO process
	if (rank==0){
	
	//Init Log file
	{
		FILE* LOGFILE = fopen(logfile.c_str(),"w");
		if ( LOGFILE==NULL )
		{
		  fprintf(LOGFILE,"-- ERROR -- Cannot open log file!\n");
		  if (!quiet) printf("-- ERROR -- Cannot open log file!\n");
		  return -1;
		}
		fclose(LOGFILE);
		if (!quiet && verbose) printf("-- INFO -- Program log will be output to %s\n",logfile.c_str());
	}
	
	//Init Params readin
  Params params(paramsfile.c_str(),logfile.c_str(),!verbose || quiet);
  
  //Output-Gf
  if (arguments.Gf_file=="\0")
  { 
    char* pstring = "'Gf-file'";
		int S = params.ReadParam(Gffile,pstring);
		
		if (Gffile=="default") Gffile = "Gf.out";
		
  	FILE* flog = fopen(logfile.c_str(), "a");
		fprintf(flog,"-- INFO -- Gf will be output to %s\n",Gffile.c_str());
		fclose(flog);
  }
  else {Gffile.assign(arguments.Gf_file);}
  if (!quiet && verbose) printf("-- INFO -- Gf will be output to %s\n",Gffile.c_str());
  
  //Output-Sigma
  if (arguments.Sigma_file=="\0")
  { 
    char* pstring = "'Sig-file'";
		int S = params.ReadParam(Sigfile,pstring); 
		
		if (Sigfile=="default") Sigfile = "Sig.out";
		
  	FILE* flog = fopen(logfile.c_str(), "a");
		fprintf(flog,"-- INFO -- Sigma will be output to %s\n",Sigfile.c_str());
		fclose(flog);
  }
  else {Sigfile.assign(arguments.Sigma_file);}
	if (!quiet && verbose) printf("-- INFO -- Sigma will be output to %s\n",Sigfile.c_str());
  
  {
  	string str(arguments.grid_file);
    char* pstring = "'Grid'";
    if (arguments.grid_file=="\0"){
			int S = params.ReadParam(gridfile,pstring);
    }
    else if (str=="default"){
    	gridfile = "default";
		}
		else{
			gridfile.assign(arguments.grid_file);
		}
		
		//string str(gridfile);
		if (gridfile=="default"){
			defaultgrid = true; 
			
			if (!quiet && verbose) printf("-- INFO -- Use grid supplied by Delta (default)\n");
			
			FILE* flog = fopen(logfile.c_str(), "a");
			fprintf(flog,"-- INFO -- Use grid supplied by Delta (default)\n");
			fclose(flog);
		}
		else{
			//check if omega file is readable
			FILE * fgrid = fopen(gridfile.c_str(), "r");
			if (fgrid == NULL) { 
				char buff[BUFFERSIZE]; 
				snprintf(buff, sizeof(buff), "-- ERROR -- Failed to open file %s :",gridfile.c_str()); 
				perror(buff);
				
				FILE* flog = fopen(logfile.c_str(), "a");
				fprintf(flog,"-- ERROR -- Failed to open file : %s\n",gridfile.c_str()); 
				fclose(flog);
				
				return -1;
			}
			fclose(fgrid);
			
			{
				if (!quiet && verbose) printf("-- INFO -- Reading in grid from file : %s\n",gridfile.c_str());
				FILE* flog = fopen(logfile.c_str(), "a");
				fprintf(flog,"-- INFO -- Reading in grid from file : %s\n",gridfile.c_str());
				fclose(flog);
			}
			
			ReadFunc(gridfile.c_str(), N, omega);
			{
				if (!quiet && verbose) printf("-- INFO -- %d omega points read in from %s\n",N,gridfile.c_str()); 
				FILE* flog = fopen(logfile.c_str(), "a");
				fprintf(flog,"-- INFO -- %d omega points read in from %s\n",N,gridfile.c_str()); 
				fclose(flog);
			}
			defaultgrid = false;
		}
	}	
	
	//Delta
	{
		if (!quiet && verbose) printf("-- INFO -- deltafile set to %s\n",deltafile);
		FILE* flog = fopen(logfile.c_str(), "a");
		fprintf(flog,"-- INFO -- deltafile set to %s\n",deltafile);
		fclose(flog);
	}
  
	
  //Check if Delta file is readable
  FILE * fDelta = fopen(deltafile, "r");
  if (fDelta == NULL) { 
  	char buff[BUFFERSIZE]; 
  	snprintf(buff, sizeof(buff), "-- ERROR -- Failed to open file %s : ",deltafile); 
  	perror(buff); 
		
		FILE* flog = fopen(logfile.c_str(), "a");
		fprintf(flog,"-- ERROR -- Failed to open file %s\n",deltafile); 
		fclose(flog);
  	return -1;
  }
  fclose(fDelta);
  
  
  //Read-in Delta
	if (defaultgrid == true){ //Use grid provided by Delta file
		ReadFunc(deltafile, N,Delta,omega); //alloc memory for Delta, omega. Also set N to dimension of input
	}
	else{ //Read-in Delta and interpolate to the omega-grid
		int N_in;
		
		double * omega_in;
		double * reDelta_in;
		double * imDelta_in;
		
		ReadFunc(deltafile, N_in,reDelta_in,imDelta_in,omega_in); 
		
		Delta = new complex<double>[N];	//Alloc memory for Delta, size is equal to that of the input grid
		
		for (int i = 0; i<N; i++){
		  Delta[i] = complex<double>( 
				dinterpl::linear_eval (omega[i], omega_in, reDelta_in, N_in) ,
				dinterpl::linear_eval (omega[i], omega_in, imDelta_in, N_in) );}
  	delete [] omega_in;
  	delete [] reDelta_in;
  	delete [] imDelta_in;
	}


	if (!quiet && verbose) printf("-- INFO -- %d points of Delta read in\n",N);
	{
	FILE* flog = fopen(logfile.c_str(), "a");
	fprintf(flog,"-- INFO -- %d points of Delta read in\n",N);
	fclose(flog);
	}
	
	//impurity parameters
  params.ReadParam(U,"'U'");
  params.ReadParam(T,"'T'");
  params.ReadParam(epsilon,"'epsilon'");
  params.ReadParam(mu,"'mu'");
	params.ReadParam(mu0,"'mu0'");
	
  //broadening of G0
  params.ReadParam(eta,"'eta'");
	  
  //PH symmetry
  params.ReadParam(SymmetricCase,"'SymmetricCase'"); //Force PH symmetry, sample only positive frequencies
  params.ReadParam(Fixmu0,"'Fixmu0'");
  
  params.ReadParam(Accr,"'Accr'");
  params.ReadParam(HybridBisectStart,"'HybridBisectStart'");
  params.ReadParam(HybridBisectEnd,"'HybridBisectEnd'");
  params.ReadParam(HybridBisectMaxIts,"'HybridBisectMaxIts'");
  
  //Kramers Kronig
  params.ReadParam(KKAccr,"'KKAccr'");
  params.ReadParam(usecubicspline,"'UseCubicSpline'");
  
  //G0 integral tail correction
  params.ReadParam(fitorder,"'fitorder'");
  
  params.ReadParam(ntail,"'ntail'");
  
  L = new double[fitorder];
  R = new double[fitorder];
  
  struct fit_params fparams;
  
  fparams.setdefault();
  fparams.ntail = ntail;
  fparams.p = fitorder;
  fparams.verbose = verbose;
  fparams.quiet = quiet;
  
  if (!quiet && verbose) printf("FO = %d, ntail = %d\n",fitorder,ntail);
  
  fit_tail(omega,Delta, N,L,R,fitorder, &fparams);
  
  if (!quiet && verbose) { for (int i=0;i<fitorder;i++) printf("Fitted L[%d] : %f , R[%d] : %f\n",i,L[i],i,R[i]); }
  
  params.ReadParam(tailcorrection,"'TailCorrection'");

  //options
  params.ReadParam(CheckSpectralWeight,"'CheckSpectralWeight'"); //default false
  
	if (!quiet && verbose) printf("-- INFO -- Finish initializing paramters\n");
	{
	FILE* flog = fopen(logfile.c_str(), "a");
	fprintf(flog,"-- INFO -- Finish initializing paramters\n");
	fclose(flog);
	}
	//End root IO process
	}
	
	MPI_Bcast(&N,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&fitorder,1,MPI_INT,0,MPI_COMM_WORLD);
	
	//---Alloc memory for workers---//
	if (rank != 0){
		omega = new double[N];
		Delta = new complex<double>[N];
		L = new double[fitorder];
		R = new double[fitorder];
	}
	
	//---Begin parallel process---//
	MPI_Bcast(&mu,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&mu0,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&U,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&T,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&epsilon,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&eta,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	
	MPI_Bcast(&KKAccr,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&Accr,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	
	MPI_Bcast(L,fitorder,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(R,fitorder,MPI_DOUBLE,0,MPI_COMM_WORLD);
	
	MPI_Bcast(&SymmetricCase,1,MPI_C_BOOL,0,MPI_COMM_WORLD);
	MPI_Bcast(&Fixmu0,1,MPI_C_BOOL,0,MPI_COMM_WORLD);
	MPI_Bcast(&usecubicspline,1,MPI_C_BOOL,0,MPI_COMM_WORLD);
	MPI_Bcast(&tailcorrection,1,MPI_C_BOOL,0,MPI_COMM_WORLD);
	
	MPI_Bcast(&CheckSpectralWeight,1,MPI_C_BOOL,0,MPI_COMM_WORLD);
	MPI_Bcast(&tailcorrection,1,MPI_C_BOOL,0,MPI_COMM_WORLD);
	MPI_Bcast(&verbose,1,MPI_C_BOOL,0,MPI_COMM_WORLD);
	
	MPI_Bcast(&HybridBisectMaxIts,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&HybridBisectStart,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&HybridBisectEnd,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	
	MPI_Bcast(omega,N,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(Delta,N,MPI_DOUBLE,0,MPI_COMM_WORLD);
	
  //Initialize SIAM solver
  struct siamparams solverparams;
  
  solverparams.mu = mu;
  solverparams.mu0 = mu0;
  solverparams.U = U;
  solverparams.T = T;
  solverparams.epsilon = epsilon;
  solverparams.eta = eta;
  
  solverparams.KKAccr = KKAccr;
  solverparams.Accr = Accr;
  
  solverparams.L = L;
  solverparams.R = R;
  
	solverparams.fitorder = fitorder;
	solverparams.SymmetricCase = SymmetricCase;
	solverparams.Fixmu0 = Fixmu0;
	solverparams.usecubicspline = usecubicspline;
	solverparams.tailcorrection = tailcorrection;
	
	solverparams.CheckSpectralWeight = CheckSpectralWeight;
  solverparams.verbose = verbose;
	solverparams.HybridBisectMaxIts = HybridBisectMaxIts;
	solverparams.HybridBisectStart = HybridBisectStart;
	solverparams.HybridBisectEnd = HybridBisectEnd;
	
  MPI_Barrier(MPI_COMM_WORLD);
	
	//Initialize the solver using supplied arguments
	SIAM Solver(omega,N,&solverparams);
	
	if (rank==0) { 
		if (debug==true) Solver.PrintFullResult("input.res"); //for debug
		if (!quiet && verbose) printf("-- INFO -- Finish printing full input to input.res\n");
	}
	//run the impurity solver
	Solver.Run(Delta); 
	
	if (rank==0) { 
	
		Solver.PrintBuffer(logfile.c_str(),quiet);
	
		Solver.PrintResult(Gffile.c_str(),Sigfile.c_str());
	
		if (debug==true) Solver.PrintFullResult("output.res"); //for debug
		if (!quiet && verbose) printf("-- INFO -- Finish printing full output to output.res\n");
	}
  //Free heap
  
  delete [] L;
  delete [] R;
  
  delete [] Delta;
  delete [] omega;

	//EXIT MPI
	MPI_Finalize();
	  
  return 0;
}
