#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>
#include <list>
#include <map>
#include <string>
#include <vector>

template<typename ...L>
struct multilambda : L... 
{
	using L::operator()...;
	constexpr multilambda(L...lambda) 
		: L(std::move(lambda))... 
	{}
};

// very simple format helper that mimics the C++20 std::format with {} pattern.
// currently, this code only support std::string and double parameters.

template <typename ...Args>
inline auto format(std::string const& str, const Args&... args) -> std::string
{
	const std::string::difference_type op = std::count(str.begin(), str.end(), '{');
	const std::string::difference_type cl = std::count(str.begin(), str.end(), '}');
	assert(op == cl);
	
	const std::size_t nargs = sizeof...(Args);
	assert(op == nargs); 

	std::list<std::string> arguments;

	auto remove_trailing_zeros = [](std::string str) -> std::string
	{
		while (str[str.size() - 1] == '0' || str[str.size() - 1] == '.')
			str.resize(str.size() - 1);
		return str;
	};

	multilambda action
	{
		[&](double d) { arguments.push_back(d == 0 ? "0.0" : remove_trailing_zeros(std::to_string(d))); },
		[&](std::string s) { arguments.push_back(s); },
	};
	std::apply([action](auto ...v) { (action(v), ...); }, std::tuple<Args...>(args...));

	std::string output;
	std::string accumulator;
	bool needclose = false;
	for (auto&& c : str)
	{
		if (c == '{')
		{
			output += accumulator;
			accumulator = "";
			needclose = true;
		}
		else
		if (c == '}')
		{
			if (!needclose || !accumulator.empty())
				throw std::runtime_error("format: malformed pattern, only support {} without contents.");
			else
				needclose = false;

			output += arguments.front();
					  arguments.pop_front();
		}
		else
			accumulator += c;
	}
	if (!accumulator.empty())
		output += accumulator;

	assert(arguments.empty());

	return output;
}