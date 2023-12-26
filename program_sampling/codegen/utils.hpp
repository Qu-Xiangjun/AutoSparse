#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

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


#endif // UTILS_HPP