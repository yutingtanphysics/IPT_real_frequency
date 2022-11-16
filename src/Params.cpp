/*
	The Params class read params from file, but you can choose not to use it
	Adapted from code by Jaksha Vuchichevicc https://github.com/JaksaVucicevic/DMFT
	in the original code this class is named "Input"
*/

#include <cstdio>
#include <string>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include "Params.h"
using namespace std;


Params::Params(const char* InputFN,const char* FN,bool quiet)
{
	this->quiet = quiet;
	this->LOGFN = FN;
  SetInputFN(InputFN);
}

Params::~Params()
{
  
}

void Params::SetInputFN(const char* InputFN)
{ 
	FILE* LOGFILE = fopen(this->LOGFN.c_str(),"a");
	
  this->InputFN.assign(InputFN);
  FILE* InputFile = fopen(this->InputFN.c_str(),"r");
  if ( InputFile==NULL )
  {
    fprintf(LOGFILE,"-- WARNING -- Params: Params File does not exist!\n");
    printf("-- WARNING -- Params: Params File does not exist!\n");
    abort();
  }
  else
  {
  	if (!quiet) printf("-- INFO -- Params: Params File name set to: %s\n",this->InputFN.c_str());
    fprintf(LOGFILE,"-- INFO -- Params: Params File name set to: %s\n",this->InputFN.c_str());
    fclose(InputFile);
  }
  
  fclose(LOGFILE);
}

template <typename T> int Params::ReadArray(int N, T* Param,const char* ParamName)
{
  char* line = ReadParam(ParamName);
  if (line==NULL) return -1;
  stringstream ss;
  ss << line;
  for(int i = 0; i < N; i++)
    ss >> Param[i];
  return 0;
}

int Params::ReadArray(int N, double* Param, const char* ParamName)
{
  int Err;
  Err = ReadArray<double>(N, Param, ParamName);
  return Err;
}

int Params::ReadArray(int N, int* Param, const char* ParamName)
{
  int Err;
  Err = ReadArray<int>(N, Param, ParamName);
  return Err;
}

int Params::ReadParam(int& Param, const char* ParamName)
{
	FILE* LOGFILE = fopen(this->LOGFN.c_str(),"a");
	
  char* line = ReadParam(ParamName);
  if (line==NULL) return -1;

  if ( sscanf(line,"%d", &Param) == EOF ) 
  {  
     fprintf(LOGFILE,"-- ERROR -- Params: Param %s can not be read\n",ParamName);
     printf("-- ERROR -- Params: Param %s can not be read\n",ParamName);
     abort();
     //return -1;    
  }
  fclose(LOGFILE);
  
  return 0;
}

int Params::ReadParam(double& Param, const char* ParamName)
{
	FILE* LOGFILE = fopen(this->LOGFN.c_str(),"a");
	
  char* line = ReadParam(ParamName);
  if (line==NULL) return -1;

  if ( sscanf(line,"%le", &Param) == EOF ) 
  {
     fprintf(LOGFILE,"-- ERROR -- Params: Param %s can not be read\n",ParamName);  
     printf("-- ERROR -- Params: Param %s can not be read\n",ParamName);
     abort();
     //return -1;    
  }
  fclose(LOGFILE);
  return 0;
}

int Params::ReadParam(char& Param, const char* ParamName)
{
	FILE* LOGFILE = fopen(this->LOGFN.c_str(),"a");
	
  char* line = ReadParam(ParamName);
  if (line==NULL) abort();

  Param = line[0]; 
  fclose(LOGFILE);
  return 0;
}

int Params::ReadParam(char* Param, const char* ParamName)
{
	FILE* LOGFILE = fopen(this->LOGFN.c_str(),"a");
	
  char* line = ReadParam(ParamName);
  if (line==NULL) abort();

  sscanf(line,"%s",Param);
  fclose(LOGFILE);
  return 0;

}

int Params::ReadParam(string &Param, const char* ParamName)
{
	FILE* LOGFILE = fopen(this->LOGFN.c_str(),"a");
	
  char* line = ReadParam(ParamName);
  if (line==NULL) abort();

	char buffer [2048];
  sscanf(line,"%s",buffer);
  Param.assign (buffer);
  fclose(LOGFILE);
  return 0;

}

int Params::ReadParam(bool& Param, const char* ParamName)
{
	FILE* LOGFILE = fopen(this->LOGFN.c_str(),"a");
	
  char* line = ReadParam(ParamName);
  if (line==NULL) abort();

  if (line[0]=='T')
    Param = true;
  else 
    if (line[0]=='F')
      Param = false;
    else
   	{
		  printf("-- ERROR -- Params: Param %s can not be read\n",ParamName);
		  fprintf(LOGFILE,"-- ERROR -- Params: Param %s can not be read\n",ParamName);
		}
  fclose(LOGFILE);
  return 0;
}


char* Params::ReadParam(const char* ParamName)
{ 
	FILE* LOGFILE = fopen(this->LOGFN.c_str(),"a");
	
  FILE* InputFile = fopen(InputFN.c_str(),"r");
  if ( InputFile==NULL ) return NULL;
    
  char* line = new char[128];
  while ( fgets(line, 128, InputFile ) != NULL ) 
  { if (line[0]=='#') continue;
    string sline(line);
    if ( sline.find(ParamName) != string::npos )
      return line;
  }
  delete [] line;
  fclose (InputFile);

  printf("-- ERROR -- Params: Param %s can not be read\n",ParamName);
  fprintf(LOGFILE,"-- ERROR -- Params: Param %s can not be read\n",ParamName);
  
  fclose(LOGFILE);
  return NULL;
}
