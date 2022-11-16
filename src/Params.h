/*
	The Params class read params from file, but you can choose not to use it
	Adapted from code by Jaksha Vuchichevicc https://github.com/JaksaVucicevic/DMFT
	in the original code this class is named "Input"
*/

#include <iostream>
using namespace std;


class Params
{  
   public:
     Params(const char* InputFN,const char* FN,bool quiet);
     ~Params();

     void SetInputFN(const char* InputFN);   

     int ReadParam(double& Param, const char* ParamName);
     int ReadParam(int& Param, const char* ParamName);
     int ReadParam(char& Param, const char* ParamName);
     int ReadParam(char* Param, const char* ParamName);
     int ReadParam(string &Param, const char* ParamName);
     int ReadParam(bool& Param, const char* ParamName); 
     int ReadArray(int N, double* Param, const char* ParamName);
     int ReadArray(int N, int* Param, const char* ParamName);

   private:
     string InputFN,LOGFN;
     template <typename T> int ReadArray(int N, T* Param,const char* ParamName);
     char* ReadParam(const char* ParamName);
     bool quiet;
     
     
};
