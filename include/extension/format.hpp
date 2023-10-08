#pragma once

#include <map>
#include <string>
#include <vector>

// just hack a very simple (C++20) std::format method that not available on gcc 
template <typename ...Args>
inline auto format(std::string const& pattern, Args...) -> std::string
{
	return ""; // TODO
}