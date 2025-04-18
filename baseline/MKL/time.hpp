#ifndef TIME_HPP
#define TIME_HPP

#include <chrono>
using namespace std;
using Clock = ::chrono::high_resolution_clock;

/**
 * microsecond time diff. 
 * Input type can be time_point and duration in <chrono> lib
 * 
 * Parameters
 * ----------
 * arg1 : t2
 *   End time.
 * arg2 : t1
 * 	 Start time.
 */
float inline compute_clock_micro(chrono::steady_clock::time_point t2, chrono::steady_clock::time_point t1)
{
	return (double)(chrono::duration_cast<chrono::microseconds>(t2 - t1).count());
}

float inline compute_clock_micro(chrono::system_clock::time_point t2, chrono::system_clock::time_point t1)
{
	return (double)(chrono::duration_cast<chrono::microseconds>(t2 - t1).count());
}

/**
 * millisecond time diff. 
 * Input type can be time_point and duration in <chrono> lib
 * 
 * Parameters
 * ----------
 * arg1 : t2
 *   End time.
 * arg2 : t1
 * 	 Start time.
 */
double inline compute_clock(chrono::steady_clock::time_point t2, chrono::steady_clock::time_point t1)
{
	return (double)(chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count()) / 1000000.0;
}

double inline compute_clock(chrono::system_clock::time_point t2, chrono::system_clock::time_point t1)
{
	return (double)(chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count()) / 1000000.0;
}

#endif
