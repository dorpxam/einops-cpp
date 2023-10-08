#pragma once

#include <cmath>
#include <map>
#include <string>
#include <type_traits>

using Hash = std::size_t;
using HashMap = std::map<std::string, Hash>;

template <typename T, typename... Rest>
void hash_combine(Hash& seed, const T& v, const Rest&... rest)
{
	seed ^= std::hash<T>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	(hash_combine(seed, rest), ...);
}

struct HashBuilder
{
	template <typename... Args>
	Hash operator()(const Args&... args) const noexcept
	{
		Hash hash = 0;
		hash_combine(hash, args...);
		return hash;
	}
	
	template <typename... Args>
	Hash operator()(Hash hash, const Args&... args) const noexcept
	{
		hash_combine(hash, args...);
		return hash;
	}
};