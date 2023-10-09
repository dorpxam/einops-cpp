#pragma once

#include <algorithm>
#include <map>
#include <string>
#include <vector>

template <typename ...Args>
inline auto format(std::string_view const& pattern, const Args&... args) -> std::string
{
	return ""; // TODO
}