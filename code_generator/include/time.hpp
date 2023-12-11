#ifndef TIME_HPP
#define TIME_HPP

#include <chrono>
using namespace std;
using Clock = ::chrono::high_resolution_clock;

/**
 * 两个时间点之间的微秒时间差
 * 输入可以是 <chrono> 库中的 time_point 和 duration 类型
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
 * 两个时间点之间的毫秒时间差
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
