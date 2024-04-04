#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <dlfcn.h>

using namespace std;

typedef vector<pair<uint64_t, float>> Compressed_Coo;

string log_prefix = "./log";

int roundup(int num, int multiple)
{
	return ((num + multiple - 1) / multiple) * multiple;
}

int get_ceil_log2(int x) 
{ 
    return (int)ceil(log2(x)); 
}

/**
 * Extract compssed coo coordinate.
 */
int extract(uint64_t coords, int start, int len)
{
	uint64_t result = (coords >> start);
	result = result & ((1 << len) - 1);
	return (int)result;
}

/**
 * extract compressed coo corrdinate, but don't remove upper 
 * coordinate.
 */
uint64_t extract_upper_coords(uint64_t coords, int start)
{
	return coords >> start;
}

/* Whether power of 2 */
bool is_power_of_2(int n) { return (n > 0) && ((n & (n - 1)) == 0); }

/* Debug func. */
void fwrite2file(float* val, int data_size, string title = "*******")
{
	string filename = log_prefix + "/Debug_data.txt";
	ofstream outputFile(filename);
	outputFile << title << endl;
	for(int tt = 0; tt < data_size; tt++) 
	{
		outputFile << fixed << setprecision(2) << val[tt] << endl;
	}
	outputFile << endl;
	outputFile.close();
	cout<< "[Debug] Debug data write successed." <<endl;
}

void fwrite2file(vector<int> val, int data_size, string title = "*******")
{
	string filename = log_prefix + "/Debug_data.txt";
	ofstream outputFile(filename);
	outputFile << title << endl;
	for(int tt = 0; tt < data_size; tt++) 
	{
		outputFile << fixed << setprecision(2) << val[tt] << endl;
	}
	outputFile << endl;
	outputFile.close();
	cout<< "[Debug] Debug data write successed." <<endl;
}

void fwrite2file(Compressed_Coo val, int data_size, string title = "*******")
{
	string filename = log_prefix + "/Debug_data.txt";
	ofstream outputFile(filename);
	outputFile << title << endl;
	for(int tt = 0; tt < data_size; tt++) 
	{
		outputFile << fixed << setprecision(2) << val[tt].first << endl;
	}
	outputFile << endl;
	outputFile.close();
	cout<< "[Debug] Debug data write successed." <<endl;
    outputFile.close();
}

void fwrite2file(string val, string filename, string title = "*******")
{
	filename = log_prefix + "/" + filename;
	ofstream outputFile(filename, ios::app);
	outputFile << title << endl;
	outputFile << val << endl;
	outputFile << endl;
	outputFile.close();
	cout<< "[Debug]" << filename << " write successed." << endl;
}

/* Safe func to execute shell command. */
bool executeCommand(const string cmd) {
    FILE* pipe = popen((cmd + " 2>&1").c_str(), "r"); // Open a pipe to execute the command
    if (!pipe) 
    {
        stringstream ss;
        ss << "[ERROR][executeCommand] " << cmd << endl;
		throw std::runtime_error(ss.str());
	}
    char buffer[128];
    string result = "";
    while (!feof(pipe)) 
    {
        if (fgets(buffer, 128, pipe) != nullptr)
            result += buffer;
    }
    pclose(pipe);
	bool excution_success = 1; // Flag to indicate successful compilation.

	if (result.find("error") != string::npos || result.find("Error") != string::npos) 
    {
        cerr << "[ERROR][executeCommand] " << result << endl;
		excution_success = 0; // Set flag to indicate compilation failure
    }
    return excution_success;
}

#endif // UTILS_HPP