# IPT
Iterative Perturbation Theory Solver for Anderson Impurity Model

	Maintained by Yuting Tan
	Email: yutingtan.physics@gmail.com
	Affiliation: National High Magnetic Field Laboratory, Tallahassee, FL, USA
	
	Last Update : 2021

	Adapted from DMFT code by Jaksha Vuchichevicc https://github.com/JaksaVucicevic/DMFT
	 


To install:

	run 'make gpu' to compile the gpu binary  
	run 'make cpu' to compile the cpu binary  
	run 'make mpi' to compile the mpi binary  
	run 'make' to compile both cpu and gpu binaries.  
	run 'make install' to install to ~/bin/  
  
To run:
  
	call 'IPT-mpi' for MPI binary  
	call 'IPT-cpu' for CPU-only binary  
	call 'IPT-gpu' for GPU-accelerated binary  
	call 'IPT-gpu --help' or 'IPT-gpu --help' for help
  
Note: 

	This current code edited by the author is meant to be a standalone binary without interface/api options.
	This code supports optional Nvidia GPU acceleration in the computation process to speedup for pure-CPU evaluations in the original code.
	
Input:
	
	1. supply a file containing the hybridization function on size-N grid in the format
	
	omega1 Re(Delta1) Im(Delta1)  
	omega2 Re(Delta2) Im(Delta2)  
	...  
	omegaN Re(DeltaN) Im(DeltaN)  
	
	2. supply a file called PARAMS (Edit the one in PARAMS.example and save to PARAMS)
	
	3. (optional) supply a file called omega.inp, this will override the grid provided by the input hybridization function in 1
	
Output:

	1. Sig.out for self-energy and Gf.out for Green's function by default
	
	2. If -d flag is supplied all steps in the computation are output to output.res and input.res
	
	3. log file is output to IPT.log by default
	
	
Usage:
 
	IPT-gpu [-dqsv?V] [-g FILE] [-l FILE] [-o FILE] [-p FILE] [-w FILE]  
		[--debug] [--Gf=FILE] [--log=FILE] [--output=FILE] [--params=FILE]  
		[--quiet] [--silent] [--verbose] [--grid=FILE] [--help] [--usage]  
		[--version] [INPUT]  

Flags:

	-d, --debug                Output intermediate steps

	-g, --Gf=FILE              Output Gf to FILE

	-l, --log=FILE             Output log to FILE instead of standard output

	-o, --output=FILE          Output to FILE instead of standard output

	-p, --params=FILE          Read params from FILE instead of standard input

	-q, -s, --quiet, --silent  Don't produce any output

	-v, --verbose              Produce verbose output

	-w, --grid=FILE            Read grid from FILE

	-?, --help                 Give this help list

			--usage                Give a short usage message
		  
	-V, --version              Print program version


  
