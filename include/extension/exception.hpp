#pragma once

#include <stdexcept>

namespace einops {
	
class Exception : public std::runtime_error
{
public:
	Exception(std::string const& what)
		: std::runtime_error(what.c_str())
	{}
};

} // namespace einops