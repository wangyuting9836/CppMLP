//
// Created by wangy on 2023/12/2.
//

#ifndef _UTILS_H_
#define _UTILS_H_

#include <random>
inline std::mt19937& rand_generator()
{
	static thread_local std::mt19937 gen(std::random_device{}());
	return gen;
}

#endif //_UTILS_H_
